"""
Unified deployment gate for LLM Cost Autopilot.

Combines:
    - application readiness checks
    - production evaluation gates
    - regression-test validation

The gate is deterministic and does not make external LLM calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.benchmark import run_benchmark
from evals.evaluator import evaluate_benchmark


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "evals" / "results"
RESULTS_PATH = RESULTS_DIR / "deployment_gate.json"


def _check_application_readiness() -> list[dict]:
    """Run the existing deployment-readiness test suite."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_deployment_readiness.py",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )

    return [
        {
            "name": "application_readiness",
            "passed": result.returncode == 0,
            "details": (
                "Deployment readiness tests passed."
                if result.returncode == 0
                else "Deployment readiness tests failed."
            ),
        }
    ]


def _check_regression_suite() -> dict:
    """Run the complete regression suite."""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )

    return {
        "name": "regression_suite",
        "passed": result.returncode == 0,
        "details": (
            "Full regression suite passed."
            if result.returncode == 0
            else "Full regression suite failed."
        ),
    }


def run_deployment_gate() -> dict:
    """Run all deployment gates and return a machine-readable report."""

    benchmark_result = run_benchmark()

    evaluation = evaluate_benchmark(
        benchmark_result,
    )

    application_gates = _check_application_readiness()

    regression_gate = _check_regression_suite()

    all_gates = [
        *application_gates,
        *evaluation["gates"],
        regression_gate,
    ]

    passed = all(gate["passed"] for gate in all_gates)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "name": "LLM Cost Autopilot Deployment Gate",
            "version": "1.0",
            "provider_calls": "mocked",
            "deterministic": True,
        },
        "application": {
            "gates": application_gates,
        },
        "evaluation": {
            "gates": evaluation["gates"],
            "summary": evaluation["summary"],
        },
        "regression": regression_gate,
        "summary": {
            "total_gates": len(all_gates),
            "passed_gates": sum(gate["passed"] for gate in all_gates),
            "failed_gates": sum(not gate["passed"] for gate in all_gates),
            "overall_status": "PASS" if passed else "FAIL",
        },
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
    """Run and display the unified deployment gate."""

    result = run_deployment_gate()

    summary = result["summary"]
    evaluation = result["evaluation"]

    print()
    print("=" * 64)
    print("LLM COST AUTOPILOT — DEPLOYMENT GATE")
    print("=" * 64)

    print()
    print("Application Readiness")

    for gate in result["application"]["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  {status:<5} {gate['name']}")

    print()
    print("Production Evaluation")

    for gate in evaluation["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"

        print(
            f"  {status:<5} "
            f"{gate['name']:<20} "
            f"actual={gate['actual']:.4f} "
            f"{gate['comparison']} "
            f"{gate['threshold']:.4f}"
        )

    print()
    print("Regression")

    regression_status = "PASS" if result["regression"]["passed"] else "FAIL"

    print(f"  {regression_status:<5} {result['regression']['name']}")

    print()
    print("-" * 64)
    print(
        f"FINAL DEPLOYMENT GATE: {summary['overall_status']} "
        f"({summary['passed_gates']}/{summary['total_gates']} passed)"
    )
    print("-" * 64)

    print()
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 64)

    if summary["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
