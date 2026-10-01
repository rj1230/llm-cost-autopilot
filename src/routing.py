"""
Request routing for LLM Cost Autopilot.

route_request(prompt):
    classify complexity -> apply confidence-aware routing ->
    select primary model -> check circuit ->
    call provider with timeout -> record provider health ->
    optionally fall back -> persist audit metadata -> return result.

Confidence-aware routing keeps the raw classifier prediction separate
from the final routing tier used for provider selection.
"""

import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import yaml

from src import config
from src.classifier.predict import predict_complexity
from src.client import send_request
from src.logging_db import log_request
from src.models.registry import get_model
from src.models.response import Response
from src.resilience import (
    CircuitOpenError,
    CircuitRegistry,
    ProviderTimeoutError,
    call_with_timeout,
)


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"


_CIRCUITS = CircuitRegistry(
    failure_threshold=config.CIRCUIT_FAILURE_THRESHOLD,
    cooldown_s=config.CIRCUIT_COOLDOWN_S,
)


@dataclass
class RoutingResult:
    """
    Result of one routing attempt.

    `tier` is the final routing tier used for provider selection.

    `classifier_tier` is the raw tier predicted by the ML classifier
    before confidence-aware safety adjustment.

    `classification_confidence` is the classifier probability assigned
    to the raw predicted tier.

    `low_confidence` indicates whether the confidence policy adjusted
    the routing decision.
    """

    request_id: str
    response: Response
    tier: int
    primary_model: str
    routed_model: str
    used_fallback: bool
    classifier_tier: int
    classification_confidence: float
    low_confidence: bool


def _load_config() -> dict:
    """Load routing and fallback configuration."""

    with CONFIG_PATH.open() as file:
        return yaml.safe_load(file)


def _provider_key(model_config) -> str:
    """Return the stable provider key used by the circuit registry."""

    return model_config.provider.value


def _call_model(
    prompt: str,
    model_config,
) -> Response:
    """
    Execute one provider call through the resilience layer.

    The circuit is checked before execution and updated according
    to the result of the provider call.
    """

    provider = _provider_key(model_config)

    circuit = _CIRCUITS.get(provider)

    if not circuit.allow_request():
        raise CircuitOpenError(f"Circuit for provider '{provider}' is open.")

    try:
        response = call_with_timeout(
            lambda: send_request(
                prompt,
                model_config,
            ),
            timeout_s=config.PROVIDER_TIMEOUT_S,
        )

    except ProviderTimeoutError:
        circuit.record_failure()
        raise

    except Exception:
        circuit.record_failure()
        raise

    if response.error:
        circuit.record_failure()
    else:
        circuit.record_success()

    return response


def _timeout_response(
    model_config,
    error: Exception,
) -> Response:
    """Create a standardized timeout response."""

    return Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=config.PROVIDER_TIMEOUT_S,
        cost_usd=0.0,
        model_name=model_config.name,
        provider=model_config.provider.value,
        error=str(error),
        error_type="timeout",
    )


def _circuit_open_response(
    model_config,
    error: Exception,
) -> Response:
    """Create a standardized circuit-open response."""

    return Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
        cost_usd=0.0,
        model_name=model_config.name,
        provider=model_config.provider.value,
        error=str(error),
        error_type="circuit_open",
    )


def _unknown_error_response(
    model_config,
    error: Exception,
) -> Response:
    """Create a standardized response for unexpected routing errors."""

    return Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
        cost_usd=0.0,
        model_name=model_config.name,
        provider=model_config.provider.value,
        error=str(error),
        error_type="unknown",
    )


def _circuit_state_for(model_config) -> str:
    """Return the current circuit state for a provider."""

    provider = _provider_key(model_config)

    return _CIRCUITS.get(provider).state.value


def route_request(prompt: str) -> RoutingResult:
    """
    Route one request with confidence-aware tier selection,
    timeout protection, circuit breaking, provider fallback,
    and resilience audit metadata.
    """

    config_data = _load_config()

    prediction = predict_complexity(prompt)

    classifier_tier = prediction.tier
    classification_confidence = prediction.confidence
    low_confidence = prediction.is_low_confidence

    tier = prediction.routing_tier

    primary_model = config_data["routing"][tier]
    primary_config = get_model(primary_model)

    routed_model = primary_model
    used_fallback = False
    primary_error_type = None

    try:
        response = _call_model(
            prompt,
            primary_config,
        )

    except ProviderTimeoutError as exc:
        response = _timeout_response(
            primary_config,
            exc,
        )

    except CircuitOpenError as exc:
        response = _circuit_open_response(
            primary_config,
            exc,
        )

    except Exception as exc:  # noqa: BLE001
        response = _unknown_error_response(
            primary_config,
            exc,
        )

    primary_circuit_state = _circuit_state_for(primary_config)

    if response.error:
        primary_error_type = response.error_type

        fallback_name = config_data.get(
            "fallback",
            {},
        ).get(tier)

        if fallback_name and fallback_name != primary_model:
            fallback_config = get_model(fallback_name)

            try:
                response = _call_model(
                    prompt,
                    fallback_config,
                )

            except ProviderTimeoutError as exc:
                response = _timeout_response(
                    fallback_config,
                    exc,
                )

            except CircuitOpenError as exc:
                response = _circuit_open_response(
                    fallback_config,
                    exc,
                )

            except Exception as exc:  # noqa: BLE001
                response = _unknown_error_response(
                    fallback_config,
                    exc,
                )

            routed_model = fallback_name
            used_fallback = True

    request_id = str(uuid.uuid4())

    log_request(
        request_id=request_id,
        prompt=prompt,
        tier=tier,
        primary_model=primary_model,
        routed_model=routed_model,
        used_fallback=used_fallback,
        response=response,
        classifier_tier=classifier_tier,
        classification_confidence=classification_confidence,
        low_confidence=low_confidence,
        primary_error_type=primary_error_type,
        circuit_state=primary_circuit_state,
    )

    return RoutingResult(
        request_id=request_id,
        response=response,
        tier=tier,
        primary_model=primary_model,
        routed_model=routed_model,
        used_fallback=used_fallback,
        classifier_tier=classifier_tier,
        classification_confidence=classification_confidence,
        low_confidence=low_confidence,
    )


def route_request_with_verification(
    prompt: str,
    synchronous: bool = False,
) -> tuple[
    RoutingResult,
    "Future | VerificationOutcome | None",
]:
    """
    Route a request and optionally run quality verification.

    Asynchronous verification returns immediately and updates the
    audit record in the background.

    Synchronous verification completes before returning.

    Verification eligibility follows the raw classifier tier rather
    than the confidence-adjusted routing tier.
    """

    from src.verification.queue import (
        handle_outcome,
        submit_verification,
    )
    from src.verification.verifier import (
        VerificationOutcome,
        verify_response,
    )

    result = route_request(prompt)

    # T3 raw classifier predictions skip automatic verification.
    if result.classifier_tier == 3:
        return result, None

    if synchronous:
        outcome = verify_response(
            prompt,
            result.response,
            result.routed_model,
            result.request_id,
        )

        handle_outcome(outcome)

        return result, outcome

    future = submit_verification(
        prompt,
        result.response,
        result.routed_model,
        result.request_id,
    )

    return result, future
