from unittest.mock import patch

from src.models.registry import get_model
from src.models.response import Response
from src.resilience import CircuitRegistry, CircuitState
from src.routing import route_request


def _success_response(model_name: str, provider: str) -> Response:
    return Response(
        output_text="successful response",
        input_tokens=10,
        output_tokens=20,
        latency_s=0.1,
        cost_usd=0.001,
        model_name=model_name,
        provider=provider,
    )


def _failure_response(
    model_name: str,
    provider: str,
    error_type: str,
) -> Response:
    return Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=0.1,
        cost_usd=0.0,
        model_name=model_name,
        provider=provider,
        error=f"simulated {error_type}",
        error_type=error_type,
    )


def test_success_closes_primary_circuit():
    with patch(
        "src.routing._call_model",
        return_value=_success_response(
            "groq-gpt-oss-20b",
            "groq",
        ),
    ):
        result = route_request("Explain what an API is.")

    assert result.response.error is None
    assert result.used_fallback is False


def test_primary_failure_triggers_fallback():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    responses = [
        _failure_response(
            primary.name,
            "mistral",
            "rate_limit",
        ),
        _success_response(
            fallback.name,
            "groq",
        ),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ):
        result = route_request("Write a short explanation of Python.")

    assert result.used_fallback is True
    assert result.routed_model == "groq-gpt-oss-20b"
    assert result.response.error is None


def test_timeout_triggers_fallback():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    from src.resilience import ProviderTimeoutError

    responses = [
        ProviderTimeoutError("simulated timeout"),
        _success_response(
            fallback.name,
            "groq",
        ),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ):
        result = route_request("Explain recursion.")

    assert result.used_fallback is True
    assert result.routed_model == "groq-gpt-oss-20b"
    assert result.response.error is None


def test_circuit_open_triggers_fallback():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    from src.resilience import CircuitOpenError

    responses = [
        CircuitOpenError("simulated open circuit"),
        _success_response(
            fallback.name,
            "groq",
        ),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ):
        result = route_request("What is machine learning?")

    assert result.used_fallback is True
    assert result.routed_model == "groq-gpt-oss-20b"
    assert result.response.error is None
