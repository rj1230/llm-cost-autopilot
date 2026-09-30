"""Tests for provider error classification."""

from types import SimpleNamespace

import pytest

from src.providers.error_utils import (
    AUTHENTICATION_ERROR,
    NETWORK_ERROR,
    RATE_LIMIT,
    SERVER_ERROR,
    TIMEOUT,
    UNKNOWN,
    classify_provider_error,
)


def make_error(
    status_code: int | None = None,
    message: str = "provider error",
):
    exc = Exception(message)

    if status_code is not None:
        response = SimpleNamespace(
            status_code=status_code,
            headers={},
        )

        exc.status_code = status_code
        exc.response = response

    return exc


@pytest.mark.parametrize(
    "status_code",
    [429],
)
def test_rate_limit_error(status_code):
    assert classify_provider_error(make_error(status_code)) == RATE_LIMIT


@pytest.mark.parametrize(
    "status_code",
    [500, 502, 503, 504],
)
def test_server_error(status_code):
    assert classify_provider_error(make_error(status_code)) == SERVER_ERROR


@pytest.mark.parametrize(
    "status_code",
    [401, 403],
)
def test_authentication_error(status_code):
    assert classify_provider_error(make_error(status_code)) == AUTHENTICATION_ERROR


@pytest.mark.parametrize(
    "message",
    [
        "Request timed out",
        "Connection timeout",
        "deadline exceeded",
    ],
)
def test_timeout_error(message):
    assert classify_provider_error(make_error(message=message)) == TIMEOUT


@pytest.mark.parametrize(
    "message",
    [
        "Connection error",
        "Connection refused",
        "Network is unreachable",
        "DNS resolution failed",
    ],
)
def test_network_error(message):
    assert classify_provider_error(make_error(message=message)) == NETWORK_ERROR


def test_rate_limit_message_without_status_code():
    assert (
        classify_provider_error(make_error(message="Rate limit exceeded")) == RATE_LIMIT
    )


def test_unknown_error():
    assert (
        classify_provider_error(make_error(message="Something completely unexpected"))
        == UNKNOWN
    )
