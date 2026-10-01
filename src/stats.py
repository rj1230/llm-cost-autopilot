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

    # Classification observability.
    average_classification_confidence: float | None
    low_confidence_count: int
    low_confidence_rate: float
    promotion_count: int
    promotion_rate: float
    raw_tier_distribution: dict[str, int]
    routing_transitions: dict[str, int]

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
        average_classification_confidence=None,
        low_confidence_count=0,
        low_confidence_rate=0.0,
        promotion_count=0,
        promotion_rate=0.0,
        raw_tier_distribution={},
        routing_transitions={},
        daily_cost=[],
        quality_scores=[],
    )


def get_summary() -> DashboardSummary:
    """Calculate dashboard routing, cost, fallback, quality, and classifier metrics."""

    if not Path(DB_PATH).exists():
        return _empty_summary()

    gpt4o = get_model(ALWAYS_GPT4O)

    conn = _connect()

    try:
        rows = conn.execute(
            """
            SELECT
                timestamp,
                tier,
                primary_model,
                routed_model,
                input_tokens,
                output_tokens,
                cost_usd,
                quality_score,
                escalated,
                verified,
                used_fallback,
                classifier_tier,
                classification_confidence,
                low_confidence
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

    savings_pct = (
        savings / hypothetical_cost * 100
        if hypothetical_cost > 0
        else 0.0
    )

    routing_distribution: dict[str, int] = {}

    for row in rows:
        model = row["routed_model"]

        routing_distribution[model] = (
            routing_distribution.get(model, 0) + 1
        )

    primary_model_distribution: dict[str, int] = {}

    for row in rows:
        model = row["primary_model"]

        primary_model_distribution[model] = (
            primary_model_distribution.get(model, 0) + 1
        )

    fallback_count = sum(
        int(row["used_fallback"] or 0)
        for row in rows
    )

    fallback_rate = (
        fallback_count / total_requests
        if total_requests
        else 0.0
    )

    verified_rows = [
        row
        for row in rows
        if row["verified"]
    ]

    escalated_rows = [
        row
        for row in verified_rows
        if row["escalated"]
    ]

    quality_scores = [
        row["quality_score"]
        for row in verified_rows
        if row["quality_score"] is not None
    ]

    avg_quality = (
        sum(quality_scores) / len(quality_scores)
        if quality_scores
        else None
    )

    # ------------------------------------------------------------------
    # Classification observability
    # ------------------------------------------------------------------
    #
    # Only rows with classifier_tier and confidence are included.
    # This keeps historical rows from pre-Phase-3C migrations from
    # being assigned invented classifier metadata.
    #
    classifier_rows = [
        row
        for row in rows
        if row["classifier_tier"] is not None
        and row["classification_confidence"] is not None
    ]

    classification_confidences = [
        float(row["classification_confidence"])
        for row in classifier_rows
    ]

    average_classification_confidence = (
        sum(classification_confidences)
        / len(classification_confidences)
        if classification_confidences
        else None
    )

    low_confidence_count = sum(
        int(row["low_confidence"] or 0)
        for row in classifier_rows
    )

    classification_count = len(classifier_rows)

    low_confidence_rate = (
        low_confidence_count / classification_count
        if classification_count
        else 0.0
    )

    # A promotion is an actual change in routing tier caused by the
    # confidence-aware policy. This is intentionally different from
    # low_confidence_count because T3 cannot be promoted beyond T3.
    promotion_count = sum(
        1
        for row in classifier_rows
        if int(row["classifier_tier"]) != int(row["tier"])
    )

    promotion_rate = (
        promotion_count / classification_count
        if classification_count
        else 0.0
    )

    raw_tier_distribution: dict[str, int] = {}

    for row in classifier_rows:
        tier = str(int(row["classifier_tier"]))

        raw_tier_distribution[tier] = (
            raw_tier_distribution.get(tier, 0) + 1
        )

    routing_transitions: dict[str, int] = {}

    for row in classifier_rows:
        transition = (
            f"{int(row['classifier_tier'])}"
            f"_to_{int(row['tier'])}"
        )

        routing_transitions[transition] = (
            routing_transitions.get(transition, 0) + 1
        )

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
            len(escalated_rows) / len(verified_rows)
            if verified_rows
            else 0.0
        ),
        avg_quality_score=avg_quality,
        average_classification_confidence=(
            average_classification_confidence
        ),
        low_confidence_count=low_confidence_count,
        low_confidence_rate=low_confidence_rate,
        promotion_count=promotion_count,
        promotion_rate=promotion_rate,
        raw_tier_distribution=raw_tier_distribution,
        routing_transitions=routing_transitions,
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
