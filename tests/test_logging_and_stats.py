"""
Tests for SQLite logging and dashboard statistics.

Run with:
    uv run pytest tests/test_logging_and_stats.py -v

Uses a temporary SQLite database so tests never modify the real
data/requests.db audit trail.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import stats as stats_module
from src.logging_db import log_request, update_verification
from src.models.response import Response


def _fake_response(
    text: str,
    cost: float,
    in_tok: int = 100,
    out_tok: int = 50,
    model: str = "mistral-small",
) -> Response:
    """Create a deterministic fake provider response."""

    return Response(
        output_text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_s=0.1,
        cost_usd=cost,
        model_name=model,
        provider="test",
    )


class TestLoggingAndStats(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

        self.db_path = Path(self.tmp_dir.name) / "test_requests.db"

        self.patchers = [
            patch(
                "src.logging_db.DB_PATH",
                self.db_path,
            ),
            patch(
                "src.logging_db.DATA_DIR",
                Path(self.tmp_dir.name),
            ),
            patch(
                "src.stats.DB_PATH",
                self.db_path,
            ),
        ]

        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

        self.tmp_dir.cleanup()

    def test_summary_math(self):
        from src.models.registry import get_model

        mistral = get_model("mistral-small")
        groq = get_model("groq-gpt-oss-20b")
        gpt4o = get_model("gpt-4o")

        r1_cost = mistral.cost_for(50, 20)

        log_request(
            "r1",
            "extract x",
            1,
            "mistral-small",
            False,
            _fake_response(
                "out1",
                cost=r1_cost,
                in_tok=50,
                out_tok=20,
                model="mistral-small",
            ),
            primary_model="mistral-small",
        )

        r2_cost = groq.cost_for(100, 50)

        log_request(
            "r2",
            "summarize y",
            2,
            "groq-gpt-oss-20b",
            False,
            _fake_response(
                "out2",
                cost=r2_cost,
                in_tok=100,
                out_tok=50,
                model="groq-gpt-oss-20b",
            ),
            primary_model="groq-gpt-oss-20b",
        )

        r3_cost = groq.cost_for(80, 40)

        log_request(
            "r3",
            "fallback request",
            1,
            "groq-gpt-oss-20b",
            True,
            _fake_response(
                "out3",
                cost=r3_cost,
                in_tok=80,
                out_tok=40,
                model="groq-gpt-oss-20b",
            ),
            primary_model="mistral-small",
        )

        r4_cost = gpt4o.cost_for(500, 300)

        log_request(
            "r4",
            "design a system",
            3,
            "gpt-4o",
            False,
            _fake_response(
                "out4",
                cost=r4_cost,
                in_tok=500,
                out_tok=300,
                model="gpt-4o",
            ),
            primary_model="gpt-4o",
        )

        update_verification(
            "r1",
            0.95,
            False,
        )

        update_verification(
            "r2",
            0.40,
            True,
        )

        summary = stats_module.get_summary()

        self.assertEqual(
            summary.total_requests,
            4,
        )

        self.assertEqual(
            summary.verified_requests,
            2,
        )

        self.assertEqual(
            summary.escalation_count,
            1,
        )

        self.assertAlmostEqual(
            summary.escalation_rate_of_verified,
            0.5,
        )

        self.assertEqual(
            summary.routing_distribution,
            {
                "mistral-small": 1,
                "groq-gpt-oss-20b": 2,
                "gpt-4o": 1,
            },
        )

        self.assertEqual(
            summary.primary_model_distribution,
            {
                "mistral-small": 2,
                "groq-gpt-oss-20b": 1,
                "gpt-4o": 1,
            },
        )

        self.assertEqual(
            summary.fallback_count,
            1,
        )

        self.assertAlmostEqual(
            summary.fallback_rate,
            0.25,
        )

        self.assertGreater(
            summary.savings_usd,
            0,
        )

        self.assertGreater(
            summary.savings_pct,
            0,
        )

        self.assertAlmostEqual(
            summary.avg_quality_score,
            (0.95 + 0.40) / 2,
        )

    def test_fallback_attribution(self):
        from src.logging_db import DB_PATH

        log_request(
            request_id="fallback-1",
            prompt="test fallback",
            tier=1,
            primary_model="mistral-small",
            routed_model="groq-gpt-oss-20b",
            used_fallback=True,
            response=_fake_response(
                "fallback response",
                cost=0.0001,
                in_tok=100,
                out_tok=50,
                model="groq-gpt-oss-20b",
            ),
        )

        conn = sqlite3.connect(DB_PATH)

        try:
            row = conn.execute(
                """
                SELECT
                    primary_model,
                    routed_model,
                    used_fallback
                FROM request_log
                WHERE request_id = ?
                """,
                ("fallback-1",),
            ).fetchone()
        finally:
            conn.close()

        self.assertEqual(
            row,
            (
                "mistral-small",
                "groq-gpt-oss-20b",
                1,
            ),
        )

    def test_existing_database_is_migrated(self):
        conn = sqlite3.connect(self.db_path)

        try:
            conn.execute(
                """
                CREATE TABLE request_log (
                    request_id      TEXT PRIMARY KEY,
                    timestamp       TEXT NOT NULL,
                    prompt_hash     TEXT NOT NULL,
                    tier            INTEGER NOT NULL,
                    routed_model    TEXT NOT NULL,
                    used_fallback   INTEGER NOT NULL,
                    input_tokens    INTEGER NOT NULL,
                    output_tokens   INTEGER NOT NULL,
                    cost_usd        REAL NOT NULL,
                    latency_s       REAL NOT NULL,
                    quality_score   REAL,
                    escalated       INTEGER,
                    verified        INTEGER NOT NULL DEFAULT 0
                )
                """
            )

            conn.execute(
                """
                INSERT INTO request_log
                (
                    request_id,
                    timestamp,
                    prompt_hash,
                    tier,
                    routed_model,
                    used_fallback,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_s,
                    verified
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "old-row",
                    "2026-09-23T07:00:00+00:00",
                    "abc123",
                    1,
                    "mistral-small",
                    0,
                    50,
                    20,
                    0.00002,
                    0.2,
                    0,
                ),
            )

            conn.commit()
        finally:
            conn.close()

        summary = stats_module.get_summary()

        self.assertEqual(
            summary.total_requests,
            1,
        )

        self.assertEqual(
            summary.primary_model_distribution,
            {
                "mistral-small": 1,
            },
        )

        conn = sqlite3.connect(self.db_path)

        try:
            columns = [
                row[1]
                for row in conn.execute("PRAGMA table_info(request_log)").fetchall()
            ]

            row = conn.execute(
                """
                SELECT
                    primary_model,
                    routed_model
                FROM request_log
                WHERE request_id = ?
                """,
                ("old-row",),
            ).fetchone()
        finally:
            conn.close()

        self.assertIn(
            "primary_model",
            columns,
        )

        self.assertEqual(
            row,
            (
                "mistral-small",
                "mistral-small",
            ),
        )

    def test_empty_db_does_not_crash(self):
        summary = stats_module.get_summary()

        self.assertEqual(
            summary.total_requests,
            0,
        )

        self.assertEqual(
            summary.verified_requests,
            0,
        )

        self.assertEqual(
            summary.escalation_count,
            0,
        )

        self.assertEqual(
            summary.fallback_count,
            0,
        )

        self.assertEqual(
            summary.fallback_rate,
            0.0,
        )

        self.assertEqual(
            summary.primary_model_distribution,
            {},
        )

        self.assertEqual(
            summary.savings_usd,
            0.0,
        )

        self.assertEqual(
            summary.savings_pct,
            0.0,
        )

        self.assertIsNone(
            summary.avg_quality_score,
        )


if __name__ == "__main__":
    unittest.main()
