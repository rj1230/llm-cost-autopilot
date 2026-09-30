"""
Command-line production evaluation for LLM Cost Autopilot.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark.benchmark import run_benchmark
from evals.evaluator import evaluate_benchmark


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evals" / "results"
RESULTS_PATH = RESULTS_DIR / "latest.json"


def run_evaluation() -> dict:
    """Run the deterministic benchmark and evaluate production gates."""

    benchmark_result = run_benchmark()

    evaluation = evaluate_benchmark(benchmark_result)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": benchmark_result["benchmark"],
        "metrics": benchmark_result["metrics"],
        "by_tier": benchmark_result["by_tier"],
        "evaluation": evaluation["evaluation"],
        "gates": evaluation["gates"],
        "summary": evaluation["summary"],
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def main() -> None:
    """Run the production evaluation and print the gate report."""

    result = run_evaluation()

    metrics = result["metrics"]
    latency = metrics["latency_s"]
    cost = metrics["cost_usd"]
    summary = result["summary"]

    print()
    print("=" * 64)
    print("LLM COST AUTOPILOT — PHASE 7 PRODUCTION EVALUATION")
    print("=" * 64)

    print()
    print("Benchmark")
    print(f"  Requests:             {metrics['requests']}")
    print(f"  Provider calls:       {result['benchmark']['provider_calls']}")
    print(f"  Deterministic:        {result['benchmark']['deterministic']}")

    print()
    print("Routing")
    print(f"  Routing accuracy:     {metrics['routing_accuracy']:.2%}")

    print()
    print("Reliability")
    print(f"  Success rate:         {metrics['success_rate']:.2%}")
    print(f"  Fallback rate:        {metrics['fallback_rate']:.2%}")

    print()
    print("Latency")
    print(f"  P95:                  {latency['p95']:.4f}s")
    print(f"  P99:                  {latency['p99']:.4f}s")

    print()
    print("Cost")
    print(f"  Actual:               ${cost['actual']:.8f}")
    print(f"  GPT-4o baseline:      ${cost['gpt4o_baseline']:.8f}")
    print(f"  Savings rate:         {cost['estimated_savings_rate']:.2%}")

    print()
    print("Production Gates")

    for gate in result["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"

        print(
            f"  {status:<5} "
            f"{gate['name']:<20} "
            f"actual={gate['actual']:.4f} "
            f"{gate['comparison']} "
            f"{gate['threshold']:.4f}"
        )

    print()
    print("-" * 64)
    print(
        f"FINAL GATE: {summary['overall_status']} "
        f"({summary['passed_gates']}/{summary['total_gates']} passed)"
    )
    print("-" * 64)

    print()
    print(f"Evaluation saved to: {RESULTS_PATH}")
    print("=" * 64)

    if summary["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
