"""
Dashboard statistics for LLM Cost Autopilot.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.logging_db import DB_PATH, ensure_schema
from src.models.registry import get_model


ALWAYS_GPT4O = "gpt-4o"


@dataclass
class DashboardSummary:
    total_requests: int
    verified_requests: int
    total_cost_usd: float
    hypothetical_cost_usd: float
    savings_usd: float
    savings_pct: float
    routing_distribution: dict[str, int]
    primary_model_distribution: dict[str, int]
    fallback_count: int
    fallback_rate: float
    escalation_count: int
    escalation_rate_of_verified: float
    avg_quality_score: float | None
    daily_cost: list[dict] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)


def _connect() -> sqlite3.Connection:
    """Open the database after ensuring the current schema exists."""

    ensure_schema(DB_PATH)

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    conn.row_factory = sqlite3.Row

    return conn


def _empty_summary() -> DashboardSummary:
    """Return an empty dashboard summary."""

    return DashboardSummary(
        total_requests=0,
        verified_requests=0,
        total_cost_usd=0.0,
        hypothetical_cost_usd=0.0,
        savings_usd=0.0,
        savings_pct=0.0,
        routing_distribution={},
        primary_model_distribution={},
        fallback_count=0,
        fallback_rate=0.0,
        escalation_count=0,
        escalation_rate_of_verified=0.0,
        avg_quality_score=None,
        daily_cost=[],
        quality_scores=[],
    )


def get_summary() -> DashboardSummary:
    """Calculate dashboard routing, cost, fallback, and quality metrics."""

    if not Path(DB_PATH).exists():
        return _empty_summary()

    gpt4o = get_model(ALWAYS_GPT4O)

    conn = _connect()

    try:
        rows = conn.execute(
            """
            SELECT
                timestamp,
                primary_model,
                routed_model,
                input_tokens,
                output_tokens,
                cost_usd,
                quality_score,
                escalated,
                verified,
                used_fallback
            FROM request_log
            """
        ).fetchall()
    finally:
        conn.close()

    total_requests = len(rows)

    total_cost = sum(row["cost_usd"] for row in rows)

    hypothetical_cost = sum(
        gpt4o.cost_for(
            row["input_tokens"],
            row["output_tokens"],
        )
        for row in rows
    )

    savings = hypothetical_cost - total_cost

    savings_pct = savings / hypothetical_cost * 100 if hypothetical_cost > 0 else 0.0

    routing_distribution: dict[str, int] = {}

    for row in rows:
        model = row["routed_model"]

        routing_distribution[model] = routing_distribution.get(model, 0) + 1

    primary_model_distribution: dict[str, int] = {}

    for row in rows:
        model = row["primary_model"]

        primary_model_distribution[model] = primary_model_distribution.get(model, 0) + 1

    fallback_count = sum(int(row["used_fallback"] or 0) for row in rows)

    fallback_rate = fallback_count / total_requests if total_requests else 0.0

    verified_rows = [row for row in rows if row["verified"]]

    escalated_rows = [row for row in verified_rows if row["escalated"]]

    quality_scores = [
        row["quality_score"]
        for row in verified_rows
        if row["quality_score"] is not None
    ]

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None

    daily: dict[str, dict] = {}

    for row in rows:
        day = row["timestamp"][:10]

        bucket = daily.setdefault(
            day,
            {
                "date": day,
                "cost_usd": 0.0,
                "requests": 0,
            },
        )

        bucket["cost_usd"] += row["cost_usd"]
        bucket["requests"] += 1

    daily_cost = sorted(
        daily.values(),
        key=lambda bucket: bucket["date"],
    )

    return DashboardSummary(
        total_requests=total_requests,
        verified_requests=len(verified_rows),
        total_cost_usd=total_cost,
        hypothetical_cost_usd=hypothetical_cost,
        savings_usd=savings,
        savings_pct=savings_pct,
        routing_distribution=routing_distribution,
        primary_model_distribution=primary_model_distribution,
        fallback_count=fallback_count,
        fallback_rate=fallback_rate,
        escalation_count=len(escalated_rows),
        escalation_rate_of_verified=(
            len(escalated_rows) / len(verified_rows) if verified_rows else 0.0
        ),
        avg_quality_score=avg_quality,
        daily_cost=daily_cost,
        quality_scores=quality_scores,
    )


def get_daily_escalation_rate() -> list[dict]:
    """
    Return daily escalation rates for verified requests.

    Tier-3 requests are excluded because they are not verified.
    """

    if not Path(DB_PATH).exists():
        return []

    conn = _connect()

    try:
        rows = conn.execute(
            """
            SELECT
                timestamp,
                escalated
            FROM request_log
            WHERE verified = 1
            """
        ).fetchall()
    finally:
        conn.close()

    daily: dict[str, dict] = {}

    for row in rows:
        day = row["timestamp"][:10]

        bucket = daily.setdefault(
            day,
            {
                "date": day,
                "verified": 0,
                "escalated": 0,
            },
        )

        bucket["verified"] += 1
        bucket["escalated"] += int(row["escalated"] or 0)

    result = []

    for day in sorted(daily):
        bucket = daily[day]

        result.append(
            {
                "date": bucket["date"],
                "verified": bucket["verified"],
                "escalation_rate": (
                    bucket["escalated"] / bucket["verified"]
                    if bucket["verified"]
                    else 0.0
                ),
            }
        )

    return result
