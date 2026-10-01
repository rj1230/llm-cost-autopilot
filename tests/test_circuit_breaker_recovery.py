"""Integration tests for circuit breaker recovery."""

import time
from unittest.mock import patch

from src.models.registry import get_model
from src.models.response import Response
from src.resilience import CircuitRegistry, CircuitState
from src.routing import _call_model


def _success_response() -> Response:
    """Build a deterministic successful provider response."""

    model = get_model("mistral-small")

    return Response(
        output_text="Provider recovered successfully.",
        input_tokens=10,
        output_tokens=8,
        latency_s=0.2,
        cost_usd=0.000003,
        model_name=model.name,
        provider=model.provider.value,
    )


def test_provider_circuit_recovers_after_failures_and_cooldown():
    """Validate OPEN -> HALF_OPEN -> CLOSED recovery."""

    registry = CircuitRegistry(
        failure_threshold=3,
        cooldown_s=0.05,
    )

    circuit = registry.get("mistral")
    model = get_model("mistral-small")

    with patch(
        "src.routing._CIRCUITS",
        registry,
    ), patch(
        "src.routing.send_request",
        return_value=_success_response(),
    ) as mock_send:
        circuit.record_failure()
        circuit.record_failure()

        assert circuit.state == CircuitState.CLOSED

        circuit.record_failure()

        assert circuit.state == CircuitState.OPEN
        assert circuit.allow_request() is False

        time.sleep(0.07)

        assert circuit.state == CircuitState.HALF_OPEN

        response = _call_model(
            "Explain circuit breaker recovery.",
            model,
        )

        assert response.error is None
        assert response.output_text == "Provider recovered successfully."

        assert circuit.state == CircuitState.CLOSED
        assert circuit.snapshot().consecutive_failures == 0
        assert circuit.allow_request() is True

        assert mock_send.call_count == 1
