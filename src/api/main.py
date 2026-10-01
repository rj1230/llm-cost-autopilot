"""
FastAPI service for LLM Cost Autopilot.

Public endpoints:
    GET  /healthz       - liveness probe
    GET  /readyz        - readiness probe

Authenticated endpoints:
    POST /v1/completions
    GET  /v1/models
    GET  /v1/stats
    PUT /v1/routing-config

Run with:
    uvicorn src.api.main:app --reload
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src import config
from src.api.auth import require_api_key
from src.api.schemas import (
    ChatChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    RoutingConfigUpdate,
    RoutingMetadata,
)
from src.logging_db import ensure_schema
from src.models.registry import MODEL_REGISTRY
from src.routing import CONFIG_PATH, route_request_with_verification
from src.stats import get_summary


logger = logging.getLogger("llm_cost_autopilot.api")


app = FastAPI(
    title="LLM Cost Autopilot",
    description=(
        "Provider-agnostic LLM routing with complexity-based model "
        "selection, fallback, quality verification, cost tracking, "
        "and resilience controls."
    ),
    version="0.6.0",
)


ERROR_STATUS_CODES = {
    "rate_limit": 503,
    "timeout": 504,
    "server_error": 503,
    "authentication_error": 502,
    "network_error": 503,
    "circuit_open": 503,
    "unknown": 500,
}


ERROR_MESSAGES = {
    "rate_limit": "The selected provider is rate limited.",
    "timeout": "The selected provider timed out.",
    "server_error": "The selected provider is temporarily unavailable.",
    "authentication_error": (
        "The selected provider is unavailable due to an upstream authentication error."
    ),
    "network_error": "The selected provider is temporarily unreachable.",
    "circuit_open": "The selected provider is temporarily unavailable.",
    "unknown": "An unexpected internal error occurred.",
}


def _error_response(error_type: str | None) -> JSONResponse:
    """
    Convert an internal provider error into a stable public API response.

    Raw provider or SDK exception details are intentionally never returned.
    """

    normalized_type = error_type or "unknown"

    status_code = ERROR_STATUS_CODES.get(
        normalized_type,
        500,
    )

    message = ERROR_MESSAGES.get(
        normalized_type,
        ERROR_MESSAGES["unknown"],
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": message,
            "error_type": normalized_type,
        },
    )


def _log_event(
    event: str,
    *,
    request_id: str | None = None,
    **fields: object,
) -> None:
    """
    Emit a structured JSON operational event.

    Secrets, API keys, raw prompts, and provider exception details
    must never be included in the logged fields.
    """

    payload = {
        "event": event,
        "request_id": request_id,
        **fields,
    }

    logger.info(
        json.dumps(
            payload,
            separators=(",", ":"),
            default=str,
        )
    )


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe."""

    return {
        "status": "ok",
    }


@app.get("/readyz")
def readyz() -> JSONResponse:
    """
    Readiness probe.

    Verifies the local persistence layer, classifier artifact,
    routing configuration, and required provider configuration.

    Returns HTTP 200 when all checks pass and HTTP 503 otherwise.
    """

    checks: dict[str, str] = {}

    # SQLite
    try:
        ensure_schema()
        checks["database"] = "ok"
    except Exception:  # noqa: BLE001
        checks["database"] = "error"

    # Classifier artifact
    classifier_path = CONFIG_PATH.parent.parent / "data" / "classifier_v4_final.joblib"

    checks["classifier"] = "ok" if classifier_path.is_file() else "error"

    # Routing configuration
    try:
        with CONFIG_PATH.open(
            encoding="utf-8",
        ) as file:
            routing_config = yaml.safe_load(file)

        if not isinstance(routing_config, dict):
            raise ValueError("routing configuration is not an object")

        routing = routing_config.get(
            "routing",
            {},
        )

        if set(routing) != {1, 2, 3}:
            raise ValueError("routing configuration must contain tiers 1, 2, and 3")

        unknown_routing_models = [
            model for model in routing.values() if model not in MODEL_REGISTRY
        ]

        if unknown_routing_models:
            raise ValueError(f"unknown routing models: {unknown_routing_models}")

        fallback = routing_config.get(
            "fallback",
            {},
        )

        if not isinstance(fallback, dict):
            raise ValueError("fallback configuration is not an object")

        unsupported_fallback_tiers = sorted(set(fallback) - {1, 2, 3})

        if unsupported_fallback_tiers:
            raise ValueError("fallback configuration contains unsupported tiers")

        unknown_fallback_models = [
            model for model in fallback.values() if model not in MODEL_REGISTRY
        ]

        if unknown_fallback_models:
            raise ValueError(f"unknown fallback models: {unknown_fallback_models}")

        for tier, fallback_model in fallback.items():
            primary_model = routing.get(tier)

            if primary_model is not None and fallback_model == primary_model:
                raise ValueError(
                    f"fallback model for tier {tier} must differ from the primary model"
                )

        checks["routing_config"] = "ok"

    except Exception:  # noqa: BLE001
        checks["routing_config"] = "error"

    # Provider configuration
    checks["mistral_provider"] = "ok" if config.MISTRAL_API_KEY else "error"

    checks["groq_provider"] = "ok" if config.GROQ_API_KEY else "error"

    ready = all(value == "ok" for value in checks.values())

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checks": checks,
        },
    )


@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    dependencies=[Depends(require_api_key)],
)
def create_completion(
    request: CompletionRequest,
) -> CompletionResponse | JSONResponse:
    """Create a routed completion."""

    user_messages = [message for message in request.messages if message.role == "user"]

    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="messages must include at least one role='user' message",
        )

    prompt = user_messages[-1].content

    result, verification = route_request_with_verification(
        prompt,
        synchronous=request.wait_for_verification,
    )

    request_id = result.request_id

    # A provider/resilience failure survived the fallback path.
    if result.response.error:
        _log_event(
            "completion_failed",
            request_id=request_id,
            tier=result.tier,
            classifier_tier=result.classifier_tier,
            classification_confidence=result.classification_confidence,
            low_confidence=result.low_confidence,
            primary_model=result.primary_model,
            routed_model=result.routed_model,
            used_fallback=result.used_fallback,
            latency_s=result.response.latency_s,
            error_type=result.response.error_type,
        )

        return _error_response(
            result.response.error_type,
        )

    if verification is None:
        verification_status = "skipped (tier 3)"
        final_text = result.response.output_text

    elif request.wait_for_verification:
        verification_status = "escalated" if verification.escalated else "passed"
        final_text = verification.final_response.output_text

    else:
        verification_status = "queued"
        final_text = result.response.output_text

    reasoning = {
        1: "classified as simple - routed to the cheapest model",
        2: "classified as moderate complexity - routed to a mid-tier model",
        3: "classified as complex - routed straight to the highest-quality model",
    }[result.tier]

    if result.classifier_tier != result.tier:
        reasoning += (
            " (low-confidence classifier prediction was promoted "
            "to a safer routing tier)"
        )

    if result.used_fallback:
        reasoning += (
            " (primary model for this tier was unavailable - used the fallback)"
        )

    _log_event(
        "completion_succeeded",
        request_id=request_id,
        tier=result.tier,
        classifier_tier=result.classifier_tier,
        classification_confidence=result.classification_confidence,
        low_confidence=result.low_confidence,
        primary_model=result.primary_model,
        routed_model=result.routed_model,
        used_fallback=result.used_fallback,
        input_tokens=result.response.input_tokens,
        output_tokens=result.response.output_tokens,
        total_tokens=result.response.total_tokens,
        cost_usd=result.response.cost_usd,
        latency_s=result.response.latency_s,
        verification=verification_status,
        error_type=None,
    )

    return CompletionResponse(
        id=request_id,
        choices=[
            ChatChoice(
                message=ChatMessage(
                    role="assistant",
                    content=final_text,
                )
            )
        ],
        routing=RoutingMetadata(
            request_id=request_id,
            tier=result.tier,
            classifier_tier=result.classifier_tier,
            classification_confidence=result.classification_confidence,
            low_confidence=result.low_confidence,
            selected_model=result.routed_model,
            reasoning=reasoning,
            used_fallback=result.used_fallback,
            cost_usd=result.response.cost_usd,
            latency_s=result.response.latency_s,
            verification=verification_status,
        ),
    )


@app.get(
    "/v1/models",
    response_model=list[ModelInfo],
    dependencies=[Depends(require_api_key)],
)
def list_models() -> list[ModelInfo]:
    """Return registered model metadata."""

    return [
        ModelInfo(
            name=model.name,
            provider=model.provider.value,
            cost_per_input_token=model.cost_per_input_token,
            cost_per_output_token=model.cost_per_output_token,
            quality_tier=model.quality_tier.value,
        )
        for model in MODEL_REGISTRY.values()
    ]


@app.get(
    "/v1/stats",
    dependencies=[Depends(require_api_key)],
)
def stats() -> dict:
    """Return operational cost, routing, and classifier statistics."""

    summary = get_summary()

    return {
        "total_requests": summary.total_requests,
        "verified_requests": summary.verified_requests,
        "total_cost_usd": round(
            summary.total_cost_usd,
            6,
        ),
        "hypothetical_all_gpt4o_cost_usd": round(
            summary.hypothetical_cost_usd,
            6,
        ),
        "savings_usd": round(
            summary.savings_usd,
            6,
        ),
        "savings_pct": round(
            summary.savings_pct,
            2,
        ),
        "routing_distribution": summary.routing_distribution,
        "primary_model_distribution": summary.primary_model_distribution,
        "fallback_count": summary.fallback_count,
        "fallback_rate": round(
            summary.fallback_rate,
            4,
        ),
        "escalation_count": summary.escalation_count,
        "escalation_rate_of_verified": round(
            summary.escalation_rate_of_verified,
            4,
        ),
        "avg_quality_score": summary.avg_quality_score,
        "classification_observability": {
            "average_confidence": (
                round(
                    summary.average_classification_confidence,
                    4,
                )
                if summary.average_classification_confidence is not None
                else None
            ),
            "low_confidence_count": summary.low_confidence_count,
            "low_confidence_rate": round(
                summary.low_confidence_rate,
                4,
            ),
            "promotion_count": summary.promotion_count,
            "promotion_rate": round(
                summary.promotion_rate,
                4,
            ),
            "raw_tier_distribution": summary.raw_tier_distribution,
            "routing_transitions": summary.routing_transitions,
        },
    }


@app.put(
    "/v1/routing-config",
    dependencies=[Depends(require_api_key)],
)
def update_routing_config(
    update: RoutingConfigUpdate,
) -> dict:
    """Update routing policy without redeploying the service."""

    unknown = [
        model
        for model in {
            **update.routing,
            **update.fallback,
        }.values()
        if model not in MODEL_REGISTRY
    ]

    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model(s) not in the registry: {unknown}",
        )

    for tier, fallback_model in update.fallback.items():
        primary_model = update.routing.get(tier)

        if primary_model is not None and fallback_model == primary_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"fallback model for tier {tier} must differ from the primary model"
                ),
            )

    new_config = {
        "routing": update.routing,
        "fallback": update.fallback,
    }

    CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=CONFIG_PATH.parent,
            prefix=f".{CONFIG_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            yaml.safe_dump(
                new_config,
                file,
                default_flow_style=False,
                sort_keys=False,
            )

            file.flush()
            os.fsync(file.fileno())

            temporary_path = Path(file.name)

        os.replace(
            temporary_path,
            CONFIG_PATH,
        )

        temporary_path = None

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to persist routing configuration.",
        ) from exc

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(
                    missing_ok=True,
                )
            except OSError:
                pass

    _log_event(
        "routing_config_updated",
        routing=update.routing,
        fallback=update.fallback,
    )

    return {
        "status": "updated",
        "config": new_config,
    }