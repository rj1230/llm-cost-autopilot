"""Concurrency tests for LLM Cost Autopilot."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from src.models.response import Response
from src.routing import route_request


def _successful_response() -> Response:
    """Build a deterministic successful provider response."""

    return Response(
        output_text="Concurrent request succeeded.",
        input_tokens=10,
        output_tokens=8,
        latency_s=0.05,
        cost_usd=0.000003,
        model_name="groq-gpt-oss-20b",
        provider="groq",
    )


def test_concurrent_requests_return_unique_request_ids_and_successes(
    tmp_path,
    monkeypatch,
):
    """Validate concurrent routing and SQLite audit persistence."""

    db_path = tmp_path / "requests.db"

    monkeypatch.setattr(
        "src.logging_db.DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        "src.routing.classify_complexity",
        lambda prompt: 3,
    )

    def run_request(index: int):
        return route_request(f"Concurrent test request {index}")

    with patch(
        "src.routing._call_model",
        side_effect=lambda prompt, model: _successful_response(),
    ) as mock_call:
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(
                executor.map(
                    run_request,
                    range(20),
                )
            )

    assert len(results) == 20

    assert all(result.response.error is None for result in results)

    assert all(result.routed_model == "groq-gpt-oss-20b" for result in results)

    request_ids = {result.request_id for result in results}

    assert len(request_ids) == 20

    assert mock_call.call_count == 20

    with sqlite3.connect(db_path) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]

        persisted_ids = {
            row[0]
            for row in conn.execute("SELECT request_id FROM request_log").fetchall()
        }

    assert row_count == 20
    assert persisted_ids == request_ids
