"""Tests for the deterministic Phase 2C routing benchmark."""

from pathlib import Path

import pytest

from benchmark.benchmark import (
    RESULTS_PATH,
    build_benchmark_cases,
    run_benchmark,
)


def test_build_benchmark_cases_has_30_requests():
    cases = build_benchmark_cases()

    assert len(cases) == 30
    assert sum(case.expected_tier == 1 for case in cases) == 10
    assert sum(case.expected_tier == 2 for case in cases) == 10
    assert sum(case.expected_tier == 3 for case in cases) == 10


def test_benchmark_is_deterministic():
    first = run_benchmark()
    second = run_benchmark()

    assert first["metrics"] == second["metrics"]
    assert first["by_tier"] == second["by_tier"]


def test_benchmark_has_expected_metrics():
    result = run_benchmark()

    metrics = result["metrics"]

    assert metrics["requests"] == 30
    assert 0.0 <= metrics["routing_accuracy"] <= 1.0
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["failure_rate"] <= 1.0
    assert 0.0 <= metrics["fallback_rate"] <= 1.0

    latency = metrics["latency_s"]

    assert latency["average"] > 0
    assert latency["p50"] > 0
    assert latency["p95"] > 0
    assert latency["p99"] > 0

    tokens = metrics["tokens"]

    assert tokens["input"] > 0
    assert tokens["output"] > 0
    assert tokens["total"] > 0

    cost = metrics["cost_usd"]

    assert cost["actual"] >= 0
    assert cost["average_per_request"] >= 0
    assert cost["gpt4o_baseline"] > 0


def test_benchmark_contains_all_tiers():
    result = run_benchmark()

    assert set(result["by_tier"]) == {
        "1",
        "2",
        "3",
    }

    for tier in ("1", "2", "3"):
        assert result["by_tier"][tier]["requests"] == 10


def test_benchmark_contains_per_request_records():
    result = run_benchmark()

    records = result["records"]

    assert len(records) == 30

    for record in records:
        assert record["request_id"]
        assert record["prompt"]

        assert record["expected_tier"] in {
            1,
            2,
            3,
        }

        assert record["predicted_tier"] in {
            1,
            2,
            3,
        }

        assert record["primary_model"]
        assert record["routed_model"]

        assert isinstance(
            record["used_fallback"],
            bool,
        )

        assert isinstance(
            record["success"],
            bool,
        )

        assert record["input_tokens"] >= 0
        assert record["output_tokens"] >= 0

        assert record["total_tokens"] == (
            record["input_tokens"] + record["output_tokens"]
        )

        assert record["latency_s"] >= 0
        assert record["cost_usd"] >= 0
        assert record["baseline_cost_usd"] >= 0


def test_benchmark_writes_results_file():
    result = run_benchmark()

    assert result["benchmark"]["provider_calls"] == "mocked"

    assert result["benchmark"]["deterministic"] is True

    assert result["benchmark"]["cases"] == 30

    assert Path(RESULTS_PATH).exists()


def test_empty_cases_are_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        run_benchmark([])
