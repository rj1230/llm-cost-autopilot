"""Shared provider error classification."""

from __future__ import annotations

RATE_LIMIT = "rate_limit"
TIMEOUT = "timeout"
SERVER_ERROR = "server_error"
AUTHENTICATION_ERROR = "authentication_error"
NETWORK_ERROR = "network_error"
CIRCUIT_OPEN = "circuit_open"
UNKNOWN = "unknown"


def _status_code(exc: Exception) -> int | None:
    """Extract an HTTP status code from a provider exception."""

    status_code = getattr(exc, "status_code", None)

    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)

    if response is not None:
        response_status = getattr(
            response,
            "status_code",
            None,
        )

        if isinstance(response_status, int):
            return response_status

    return None


def classify_provider_error(exc: Exception) -> str:
    """
    Convert a provider exception into a stable operational category.

    The returned value is intentionally coarse-grained so provider-specific
    exception text does not become part of the analytics contract.
    """

    status_code = _status_code(exc)

    if status_code == 429:
        return RATE_LIMIT

    if status_code in {401, 403}:
        return AUTHENTICATION_ERROR

    if status_code is not None and 500 <= status_code <= 599:
        return SERVER_ERROR

    message = str(exc).lower()

    timeout_markers = (
        "timeout",
        "timed out",
        "time out",
        "deadline exceeded",
    )

    if any(marker in message for marker in timeout_markers):
        return TIMEOUT

    authentication_markers = (
        "authentication",
        "unauthorized",
        "invalid api key",
        "invalid_api_key",
        "api key",
        "forbidden",
    )

    if any(marker in message for marker in authentication_markers):
        return AUTHENTICATION_ERROR

    network_markers = (
        "connection error",
        "connection refused",
        "connection reset",
        "connection aborted",
        "network error",
        "network is unreachable",
        "dns",
        "name resolution",
        "temporary failure in name resolution",
    )

    if any(marker in message for marker in network_markers):
        return NETWORK_ERROR

    if "429" in message or "rate limit" in message or "rate_limited" in message:
        return RATE_LIMIT

    if any(
        marker in message
        for marker in (
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    ):
        return SERVER_ERROR

    return UNKNOWN
