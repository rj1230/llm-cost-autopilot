"""Deterministic concurrency benchmark for LLM Cost Autopilot."""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.models.response import Response
from src.routing import route_request


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark" / "results"
RESULTS_PATH = RESULTS_DIR / "concurrency_latest.json"


REQUEST_COUNT = 100
CONCURRENCY = 10


def _successful_response() -> Response:
    """Return a deterministic provider response."""

    return Response(
        output_text="Deterministic concurrency benchmark response.",
        input_tokens=50,
        output_tokens=30,
        latency_s=0.01,
        cost_usd=0.00001,
        model_name="groq-gpt-oss-20b",
        provider="groq",
    )


def _run_request(index: int):
    """Execute one benchmark request."""

    started = time.perf_counter()

    result = route_request(f"Concurrency benchmark request {index}")

    elapsed = time.perf_counter() - started

    return result, elapsed


def _percentile(values: list[float], percentile: float) -> float:
    """Calculate a percentile using linear interpolation."""

    if not values:
        return 0.0

    ordered = sorted(values)

    position = (len(ordered) - 1) * percentile

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    """Run the concurrency benchmark and persist results."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = RESULTS_DIR / "concurrency_requests.db"

    if db_path.exists():
        db_path.unlink()

    with (
        patch(
            "src.logging_db.DB_PATH",
            db_path,
        ),
        patch(
            "src.routing.classify_complexity",
            lambda prompt: 3,
        ),
        patch(
            "src.routing._call_model",
            side_effect=lambda prompt, model: _successful_response(),
        ),
    ):
        started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            results = list(
                executor.map(
                    _run_request,
                    range(REQUEST_COUNT),
                )
            )

        total_duration = time.perf_counter() - started

    routing_results = [result for result, _ in results]

    latencies = [elapsed for _, elapsed in results]

    successful = sum(result.response.error is None for result in routing_results)

    request_ids = {result.request_id for result in routing_results}

    with sqlite3.connect(db_path) as conn:
        persisted_rows = conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]

    throughput = REQUEST_COUNT / total_duration if total_duration > 0 else 0.0

    metrics = {
        "requests": REQUEST_COUNT,
        "concurrency": CONCURRENCY,
        "total_duration_s": round(total_duration, 6),
        "throughput_requests_per_second": round(
            throughput,
            4,
        ),
        "successful_requests": successful,
        "success_rate_pct": round(
            successful / REQUEST_COUNT * 100,
            2,
        ),
        "unique_request_ids": len(request_ids),
        "persisted_audit_rows": persisted_rows,
        "latency_avg_s": round(
            statistics.mean(latencies),
            6,
        ),
        "latency_p50_s": round(
            _percentile(latencies, 0.50),
            6,
        ),
        "latency_p95_s": round(
            _percentile(latencies, 0.95),
            6,
        ),
        "latency_p99_s": round(
            _percentile(latencies, 0.99),
            6,
        ),
    }

    RESULTS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
