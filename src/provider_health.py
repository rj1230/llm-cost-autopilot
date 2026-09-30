"""
Provider health metrics for LLM Cost Autopilot.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.logging_db import DB_PATH, ensure_schema


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    failure_rate: float
    average_latency_s: float
    total_cost_usd: float
    fallback_requests: int


def _provider_from_model(model_name: str) -> str:
    """Map registry model names to provider names."""

    if model_name == "mistral-small":
        return "mistral"

    if model_name == "groq-gpt-oss-20b":
        return "groq"

    if model_name == "gpt-4o":
        return "openai"

    return model_name


def _row_failed(row: sqlite3.Row) -> bool:
    """
    Determine whether a request failed.

    New rows use explicit error_type.

    Historical rows may predate error_type and therefore fall back
    to the legacy token/cost inference.
    """

    if row["error_type"] is not None:
        return True

    return not (
        row["input_tokens"] > 0 or row["output_tokens"] > 0 or row["cost_usd"] > 0
    )


def _empty_health(provider: str) -> ProviderHealth:
    """Return zero-valued health metrics for a provider."""

    return ProviderHealth(
        provider=provider,
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        failure_rate=0.0,
        average_latency_s=0.0,
        total_cost_usd=0.0,
        fallback_requests=0,
    )


def _empty() -> dict[str, ProviderHealth]:
    """Return an empty provider health mapping."""

    return {}


def _calculate_health(
    rows: list[sqlite3.Row],
) -> ProviderHealth | None:
    """Calculate health metrics for a collection of provider rows."""

    if not rows:
        return None

    provider = _provider_from_model(rows[0]["routed_model"])

    total = len(rows)

    failed = sum(1 for row in rows if _row_failed(row))

    successful = total - failed

    total_latency = sum(row["latency_s"] for row in rows)

    total_cost = sum(row["cost_usd"] for row in rows)

    fallback_requests = sum(int(row["used_fallback"] or 0) for row in rows)

    return ProviderHealth(
        provider=provider,
        total_requests=total,
        successful_requests=successful,
        failed_requests=failed,
        failure_rate=(failed / total if total else 0.0),
        average_latency_s=(total_latency / total if total else 0.0),
        total_cost_usd=total_cost,
        fallback_requests=fallback_requests,
    )


def _load_rows() -> list[sqlite3.Row]:
    """Load provider-health rows from the audit database."""

    if not Path(DB_PATH).exists():
        return []

    ensure_schema(DB_PATH)

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    conn.row_factory = sqlite3.Row

    try:
        return conn.execute(
            """
            SELECT
                routed_model,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_s,
                used_fallback,
                error_type
            FROM request_log
            """
        ).fetchall()
    finally:
        conn.close()


def get_provider_health_for(
    provider: str,
) -> ProviderHealth:
    """
    Return health metrics for one provider.

    Known providers return their calculated metrics.

    Unknown providers return zero-valued metrics rather than None,
    keeping the function safe for dashboard and API consumers.
    """

    rows = _load_rows()

    provider_rows = [
        row for row in rows if _provider_from_model(row["routed_model"]) == provider
    ]

    health = _calculate_health(provider_rows)

    if health is not None:
        return health

    return _empty_health(provider)


def get_provider_health() -> dict[str, ProviderHealth]:
    """Calculate provider-level health metrics from the audit database."""

    rows = _load_rows()

    if not rows:
        return _empty()

    grouped: dict[str, list[sqlite3.Row]] = {}

    for row in rows:
        provider = _provider_from_model(row["routed_model"])

        grouped.setdefault(
            provider,
            [],
        ).append(row)

    result: dict[str, ProviderHealth] = {}

    for provider, provider_rows in grouped.items():
        health = _calculate_health(provider_rows)

        if health is not None:
            result[provider] = health

    return result
