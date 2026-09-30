"""Cost accuracy tests for LLM Cost Autopilot."""

import sqlite3
from pathlib import Path

import pytest

from src import stats as stats_module
from src.logging_db import log_request
from src.models.registry import get_model
from src.models.response import Response


def _response(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
) -> Response:
    """Build a deterministic response for cost validation."""

    model_config = get_model(model)

    return Response(
        output_text="Cost validation response.",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=0.1,
        cost_usd=cost,
        model_name=model_config.name,
        provider=model_config.provider.value,
    )


@pytest.mark.parametrize(
    ("model_name", "input_tokens", "output_tokens"),
    [
        ("mistral-small", 1_000, 500),
        ("groq-gpt-oss-20b", 1_000, 500),
        ("gpt-4o", 1_000, 500),
    ],
)
def test_model_cost_calculation_matches_registered_pricing(
    model_name,
    input_tokens,
    output_tokens,
):
    """Verify cost_for() exactly follows registered token pricing."""

    model = get_model(model_name)

    expected = (
        input_tokens * model.cost_per_input_token
        + output_tokens * model.cost_per_output_token
    )

    assert model.cost_for(
        input_tokens,
        output_tokens,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("model_name", "expected_cost"),
    [
        (
            "mistral-small",
            0.00045,
        ),
        (
            "groq-gpt-oss-20b",
            0.000225,
        ),
        (
            "gpt-4o",
            0.0075,
        ),
    ],
)
def test_known_1000_input_500_output_costs(
    model_name,
    expected_cost,
):
    """Verify known pricing examples against the registry."""

    model = get_model(model_name)

    assert model.cost_for(
        1_000,
        500,
    ) == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    "model_name",
    [
        "mistral-small",
        "groq-gpt-oss-20b",
        "gpt-4o",
    ],
)
def test_zero_tokens_have_zero_cost(model_name):
    """Verify zero-token requests produce zero estimated cost."""

    model = get_model(model_name)

    assert model.cost_for(
        0,
        0,
    ) == pytest.approx(0.0)


def test_stats_uses_persisted_cost_and_correct_gpt4o_baseline(
    tmp_path,
    monkeypatch,
):
    """Verify dashboard economics use persisted costs and token counts."""

    db_path = Path(tmp_path) / "requests.db"

    monkeypatch.setattr(
        "src.logging_db.DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        "src.logging_db.DATA_DIR",
        Path(tmp_path),
    )

    monkeypatch.setattr(
        "src.stats.DB_PATH",
        db_path,
    )

    mistral = get_model("mistral-small")
    groq = get_model("groq-gpt-oss-20b")
    gpt4o = get_model("gpt-4o")

    requests = [
        (
            "cost-1",
            "request one",
            "mistral-small",
            1_000,
            500,
        ),
        (
            "cost-2",
            "request two",
            "groq-gpt-oss-20b",
            2_000,
            1_000,
        ),
    ]

    total_actual_cost = 0.0
    total_baseline_cost = 0.0

    for request_id, prompt, model_name, input_tokens, output_tokens in requests:
        model = mistral if model_name == "mistral-small" else groq

        actual_cost = model.cost_for(
            input_tokens,
            output_tokens,
        )

        baseline_cost = gpt4o.cost_for(
            input_tokens,
            output_tokens,
        )

        total_actual_cost += actual_cost
        total_baseline_cost += baseline_cost

        log_request(
            request_id=request_id,
            prompt=prompt,
            tier=1,
            primary_model=model_name,
            routed_model=model_name,
            used_fallback=False,
            response=_response(
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
            ),
        )

        assert actual_cost >= 0
        assert baseline_cost >= actual_cost

    summary = stats_module.get_summary()

    expected_savings = total_baseline_cost - total_actual_cost

    expected_savings_pct = expected_savings / total_baseline_cost * 100

    assert summary.total_requests == 2

    assert summary.total_cost_usd == pytest.approx(total_actual_cost)

    assert summary.hypothetical_cost_usd == pytest.approx(total_baseline_cost)

    assert summary.savings_usd == pytest.approx(expected_savings)

    assert summary.savings_pct == pytest.approx(expected_savings_pct)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                request_id,
                input_tokens,
                output_tokens,
                cost_usd
            FROM request_log
            ORDER BY request_id
            """
        ).fetchall()

    assert len(rows) == 2

    assert rows[0][0] == "cost-1"
    assert rows[0][1] == 1_000
    assert rows[0][2] == 500
    assert rows[0][3] == pytest.approx(mistral.cost_for(1_000, 500))

    assert rows[1][0] == "cost-2"
    assert rows[1][1] == 2_000
    assert rows[1][2] == 1_000
    assert rows[1][3] == pytest.approx(groq.cost_for(2_000, 1_000))
