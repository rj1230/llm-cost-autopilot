"""
Production evaluation and deployment gates for LLM Cost Autopilot.

This module evaluates the existing deterministic routing benchmark against
explicit production thresholds. It does not make external LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationThresholds:
    """Production thresholds used by the evaluation gate."""

    min_success_rate: float = 0.99
    max_fallback_rate: float = 0.20
    min_savings_rate: float = 0.50
    max_p95_latency_s: float = 2.00
    max_p99_latency_s: float = 3.00
    min_routing_accuracy: float = 0.50


DEFAULT_THRESHOLDS = EvaluationThresholds()


def _gate(
    name: str,
    passed: bool,
    actual: float,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    """Create a normalized evaluation gate result."""

    return {
        "name": name,
        "passed": passed,
        "actual": round(actual, 8),
        "threshold": round(threshold, 8),
        "comparison": comparison,
    }


def evaluate_benchmark(
    benchmark_result: dict[str, Any],
    thresholds: EvaluationThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """
    Evaluate benchmark metrics against production thresholds.

    Returns a machine-readable evaluation report containing individual
    gates and an overall PASS/FAIL status.
    """

    if not isinstance(benchmark_result, dict):
        raise TypeError("benchmark_result must be a dictionary.")

    metrics = benchmark_result.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError("Benchmark result is missing 'metrics'.")

    latency = metrics.get("latency_s")
    cost = metrics.get("cost_usd")

    if not isinstance(latency, dict):
        raise ValueError("Benchmark result is missing 'latency_s'.")

    if not isinstance(cost, dict):
        raise ValueError("Benchmark result is missing 'cost_usd'.")

    routing_accuracy = float(metrics.get("routing_accuracy", 0.0))
    success_rate = float(metrics.get("success_rate", 0.0))
    fallback_rate = float(metrics.get("fallback_rate", 1.0))
    p95_latency = float(latency.get("p95", float("inf")))
    p99_latency = float(latency.get("p99", float("inf")))
    savings_rate = float(cost.get("estimated_savings_rate", 0.0))

    gates = [
        _gate(
            "routing_accuracy",
            routing_accuracy >= thresholds.min_routing_accuracy,
            routing_accuracy,
            thresholds.min_routing_accuracy,
            ">=",
        ),
        _gate(
            "success_rate",
            success_rate >= thresholds.min_success_rate,
            success_rate,
            thresholds.min_success_rate,
            ">=",
        ),
        _gate(
            "fallback_rate",
            fallback_rate <= thresholds.max_fallback_rate,
            fallback_rate,
            thresholds.max_fallback_rate,
            "<=",
        ),
        _gate(
            "p95_latency",
            p95_latency <= thresholds.max_p95_latency_s,
            p95_latency,
            thresholds.max_p95_latency_s,
            "<=",
        ),
        _gate(
            "p99_latency",
            p99_latency <= thresholds.max_p99_latency_s,
            p99_latency,
            thresholds.max_p99_latency_s,
            "<=",
        ),
        _gate(
            "savings_rate",
            savings_rate >= thresholds.min_savings_rate,
            savings_rate,
            thresholds.min_savings_rate,
            ">=",
        ),
    ]

    passed = all(gate["passed"] for gate in gates)

    return {
        "evaluation": {
            "name": "LLM Cost Autopilot Production Evaluation",
            "version": "1.0",
            "provider_calls": "mocked",
            "deterministic": True,
        },
        "gates": gates,
        "summary": {
            "total_gates": len(gates),
            "passed_gates": sum(gate["passed"] for gate in gates),
            "failed_gates": sum(not gate["passed"] for gate in gates),
            "overall_status": "PASS" if passed else "FAIL",
        },
    }
