"""Regression tests for Phase 7 production evaluation gates."""

from copy import deepcopy

import pytest

from benchmark.benchmark import run_benchmark
from evals.evaluator import (
    DEFAULT_THRESHOLDS,
    EvaluationThresholds,
    evaluate_benchmark,
)


@pytest.fixture(scope="module")
def benchmark_result():
    """Run the deterministic benchmark once for this test module."""

    return run_benchmark()


def test_production_evaluation_passes_current_baseline(benchmark_result):
    """The current deterministic benchmark must pass all production gates."""

    result = evaluate_benchmark(benchmark_result)

    assert result["summary"]["overall_status"] == "PASS"
    assert result["summary"]["failed_gates"] == 0
    assert result["summary"]["passed_gates"] == result["summary"]["total_gates"]


def test_all_expected_gates_are_present(benchmark_result):
    """Verify the evaluation contains every required production gate."""

    result = evaluate_benchmark(benchmark_result)

    gate_names = {gate["name"] for gate in result["gates"]}

    assert gate_names == {
        "routing_accuracy",
        "success_rate",
        "fallback_rate",
        "p95_latency",
        "p99_latency",
        "savings_rate",
    }


def test_routing_accuracy_regression_fails(benchmark_result):
    """A routing regression below the threshold must fail the gate."""

    degraded = deepcopy(benchmark_result)

    degraded["metrics"]["routing_accuracy"] = 0.49

    result = evaluate_benchmark(degraded)

    assert result["summary"]["overall_status"] == "FAIL"

    routing_gate = next(
        gate for gate in result["gates"] if gate["name"] == "routing_accuracy"
    )

    assert routing_gate["passed"] is False


def test_success_rate_regression_fails(benchmark_result):
    """A reliability regression below the threshold must fail."""

    degraded = deepcopy(benchmark_result)

    degraded["metrics"]["success_rate"] = 0.98

    result = evaluate_benchmark(degraded)

    assert result["summary"]["overall_status"] == "FAIL"

    success_gate = next(
        gate for gate in result["gates"] if gate["name"] == "success_rate"
    )

    assert success_gate["passed"] is False


def test_fallback_rate_regression_fails(benchmark_result):
    """An excessive fallback rate must fail."""

    degraded = deepcopy(benchmark_result)

    degraded["metrics"]["fallback_rate"] = 0.21

    result = evaluate_benchmark(degraded)

    assert result["summary"]["overall_status"] == "FAIL"

    fallback_gate = next(
        gate for gate in result["gates"] if gate["name"] == "fallback_rate"
    )

    assert fallback_gate["passed"] is False


def test_p95_latency_regression_fails(benchmark_result):
    """A P95 latency regression must fail."""

    degraded = deepcopy(benchmark_result)

    degraded["metrics"]["latency_s"]["p95"] = 2.01

    result = evaluate_benchmark(degraded)

    assert result["summary"]["overall_status"] == "FAIL"

    latency_gate = next(
        gate for gate in result["gates"] if gate["name"] == "p95_latency"
    )

    assert latency_gate["passed"] is False


def test_p99_latency_regression_fails(benchmark_result):
    """A P99 latency regression must fail."""

    degraded = deepcopy(benchmark_result)

    degraded["metrics"]["latency_s"]["p99"] = 3.01

    result = evaluate_benchmark(degraded)

    assert result["summary"]["overall_status"] == "FAIL"

    latency_gate = next(
        gate for gate in result["gates"] if gate["name"] == "p99_latency"
    )

    assert latency_gate["passed"] is False


def test_cost_savings_regression_fails(benchmark_result):
    """A savings regression below the minimum must fail."""

    degraded = deepcopy(benchmark_result)

    degraded["metrics"]["cost_usd"]["estimated_savings_rate"] = 0.49

    result = evaluate_benchmark(degraded)

    assert result["summary"]["overall_status"] == "FAIL"

    savings_gate = next(
        gate for gate in result["gates"] if gate["name"] == "savings_rate"
    )

    assert savings_gate["passed"] is False


def test_custom_thresholds_are_supported(benchmark_result):
    """The evaluator must support explicit production thresholds."""

    thresholds = EvaluationThresholds(
        min_success_rate=1.0,
        max_fallback_rate=0.0,
        min_savings_rate=0.90,
        max_p95_latency_s=1.50,
        max_p99_latency_s=1.50,
        min_routing_accuracy=0.60,
    )

    result = evaluate_benchmark(
        benchmark_result,
        thresholds=thresholds,
    )

    assert result["summary"]["overall_status"] == "PASS"


def test_invalid_benchmark_result_is_rejected():
    """Malformed benchmark input must fail explicitly."""

    with pytest.raises(ValueError, match="metrics"):
        evaluate_benchmark({})


def test_default_thresholds_are_stable():
    """Verify the project's documented default evaluation thresholds."""

    assert DEFAULT_THRESHOLDS.min_success_rate == 0.99
    assert DEFAULT_THRESHOLDS.max_fallback_rate == 0.20
    assert DEFAULT_THRESHOLDS.min_savings_rate == 0.50
    assert DEFAULT_THRESHOLDS.max_p95_latency_s == 2.00
    assert DEFAULT_THRESHOLDS.max_p99_latency_s == 3.00
    assert DEFAULT_THRESHOLDS.min_routing_accuracy == 0.50
