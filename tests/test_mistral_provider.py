"""Unit tests for Mistral provider retry and response handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.models.registry import MODEL_REGISTRY
from src.providers.mistral_provider import MistralProvider


MODEL_CONFIG = MODEL_REGISTRY["mistral-small"]


def make_message(
    output: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Create a minimal mock Mistral completion response."""

    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=output,
                )
            )
        ],
    )


def make_exception(
    status_code: int,
    message: str | None = None,
    headers: dict | None = None,
):
    """Create an exception that resembles a Mistral SDK HTTP error."""

    response = SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
    )

    return Exception(message or f"API error: HTTP {status_code}"), response


def attach_response(exc: Exception, response: object) -> Exception:
    """Attach a response object to a normal Python exception."""

    exc.response = response
    exc.status_code = response.status_code
    return exc


def test_successful_request():
    """A successful request returns normalized output and token usage."""

    provider = MistralProvider()

    provider._client = MagicMock()

    provider._client.chat.complete.return_value = make_message(
        output="Hello from Mistral",
        prompt_tokens=20,
        completion_tokens=10,
    )

    response = provider.call(
        "Say hello",
        MODEL_CONFIG,
    )

    assert response.error is None
    assert response.output_text == "Hello from Mistral"
    assert response.input_tokens == 20
    assert response.output_tokens == 10
    assert response.cost_usd > 0
    assert response.provider == "mistral"
    assert response.model_name == "mistral-small"

    provider._client.chat.complete.assert_called_once()


def test_429_retries_then_succeeds():
    """A 429 response is retried and succeeds on the next attempt."""

    provider = MistralProvider()

    provider._client = MagicMock()

    first_error, first_response = make_exception(
        429,
        "Rate limit exceeded",
    )

    first_error = attach_response(
        first_error,
        first_response,
    )

    provider._client.chat.complete.side_effect = [
        first_error,
        make_message("Recovered"),
    ]

    with patch("src.providers.mistral_provider.time.sleep") as mock_sleep:
        response = provider.call(
            "Retry me",
            MODEL_CONFIG,
        )

    assert response.error is None
    assert response.output_text == "Recovered"

    assert provider._client.chat.complete.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_429_exhausts_retries():
    """Repeated 429 responses eventually return an error response."""

    provider = MistralProvider()

    provider._client = MagicMock()

    errors = []

    for _ in range(provider.MAX_RETRIES + 1):
        error, error_response = make_exception(
            429,
            "Rate limit exceeded",
        )

        errors.append(
            attach_response(
                error,
                error_response,
            )
        )

    provider._client.chat.complete.side_effect = errors

    with patch("src.providers.mistral_provider.time.sleep") as mock_sleep:
        response = provider.call(
            "Always fail",
            MODEL_CONFIG,
        )

    assert response.error is not None
    assert "Rate limit exceeded" in response.error

    assert provider._client.chat.complete.call_count == provider.MAX_RETRIES + 1

    assert mock_sleep.call_count == provider.MAX_RETRIES


def test_503_retries_then_succeeds():
    """A transient 503 response is retried."""

    provider = MistralProvider()

    provider._client = MagicMock()

    error, error_response = make_exception(
        503,
        "Service unavailable",
    )

    error = attach_response(
        error,
        error_response,
    )

    provider._client.chat.complete.side_effect = [
        error,
        make_message("Recovered after 503"),
    ]

    with patch("src.providers.mistral_provider.time.sleep") as mock_sleep:
        response = provider.call(
            "Retry service failure",
            MODEL_CONFIG,
        )

    assert response.error is None
    assert response.output_text == "Recovered after 503"

    assert provider._client.chat.complete.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_400_does_not_retry():
    """A client-side 400 error should fail immediately."""

    provider = MistralProvider()

    provider._client = MagicMock()

    error, error_response = make_exception(
        400,
        "Bad request",
    )

    error = attach_response(
        error,
        error_response,
    )

    provider._client.chat.complete.side_effect = error

    with patch("src.providers.mistral_provider.time.sleep") as mock_sleep:
        response = provider.call(
            "Invalid request",
            MODEL_CONFIG,
        )

    assert response.error is not None
    assert "Bad request" in response.error

    provider._client.chat.complete.assert_called_once()
    mock_sleep.assert_not_called()


def test_retry_after_header_is_honored():
    """Retry-After is preferred over exponential backoff."""

    provider = MistralProvider()

    provider._client = MagicMock()

    error, error_response = make_exception(
        429,
        "Rate limit exceeded",
        headers={"Retry-After": "3"},
    )

    error = attach_response(
        error,
        error_response,
    )

    provider._client.chat.complete.side_effect = [
        error,
        make_message("Recovered"),
    ]

    with patch("src.providers.mistral_provider.time.sleep") as mock_sleep:
        response = provider.call(
            "Respect retry-after",
            MODEL_CONFIG,
        )

    assert response.error is None
    assert response.output_text == "Recovered"

    mock_sleep.assert_called_once_with(3.0)


def test_retry_after_is_bounded():
    """Retry-After cannot exceed the provider's maximum backoff."""

    provider = MistralProvider()

    error, error_response = make_exception(
        429,
        "Rate limit exceeded",
        headers={"Retry-After": "100"},
    )

    error = attach_response(
        error,
        error_response,
    )

    delay = provider._backoff_seconds(
        attempt=0,
        exc=error,
    )

    assert delay == provider.MAX_BACKOFF_S


@pytest.mark.parametrize(
    "status_code",
    [500, 502, 503, 504],
)
def test_server_errors_are_retryable(status_code):
    """Common 5xx server errors are classified as retryable."""

    error, error_response = make_exception(
        status_code,
        f"Server error {status_code}",
    )

    error = attach_response(
        error,
        error_response,
    )

    assert MistralProvider._is_retryable(error) is True


def test_client_errors_are_not_retryable():
    """Non-transient 4xx errors are not retried."""

    for status_code in [400, 401, 403, 404, 422]:
        error, error_response = make_exception(
            status_code,
            f"Client error {status_code}",
        )

        error = attach_response(
            error,
            error_response,
        )

        assert MistralProvider._is_retryable(error) is False
