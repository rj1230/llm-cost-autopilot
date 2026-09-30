"""
Tests for provider reliability metrics.

Run with:
    uv run pytest tests/test_provider_health.py -v

Uses a temporary SQLite database.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.logging_db import log_request
from src.models.response import Response
from src.provider_health import (
    get_provider_health,
    get_provider_health_for,
)


def _response(
    cost: float,
    input_tokens: int,
    output_tokens: int,
    latency: float,
    model: str,
) -> Response:
    """Create a deterministic provider response."""

    return Response(
        output_text="test response" if cost > 0 else "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency,
        cost_usd=cost,
        model_name=model,
        provider=("groq" if "groq" in model else "mistral"),
        error=None if cost > 0 else "provider failure",
    )


class TestProviderHealth(unittest.TestCase):
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
                "src.provider_health.DB_PATH",
                self.db_path,
            ),
        ]

        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

        self.tmp_dir.cleanup()

    def test_provider_health_success_and_failure(self):
        log_request(
            request_id="groq-success-1",
            prompt="test groq success",
            tier=2,
            primary_model="groq-gpt-oss-20b",
            routed_model="groq-gpt-oss-20b",
            used_fallback=False,
            response=_response(
                cost=0.001,
                input_tokens=100,
                output_tokens=50,
                latency=1.0,
                model="groq-gpt-oss-20b",
            ),
        )

        log_request(
            request_id="groq-success-2",
            prompt="test groq success 2",
            tier=2,
            primary_model="groq-gpt-oss-20b",
            routed_model="groq-gpt-oss-20b",
            used_fallback=False,
            response=_response(
                cost=0.002,
                input_tokens=200,
                output_tokens=100,
                latency=2.0,
                model="groq-gpt-oss-20b",
            ),
        )

        log_request(
            request_id="mistral-failure",
            prompt="test mistral failure",
            tier=1,
            primary_model="mistral-small",
            routed_model="mistral-small",
            used_fallback=False,
            response=_response(
                cost=0.0,
                input_tokens=0,
                output_tokens=0,
                latency=3.0,
                model="mistral-small",
            ),
        )

        health = get_provider_health()

        self.assertIn("groq", health)
        self.assertIn("mistral", health)

        groq = health["groq"]

        self.assertEqual(
            groq.total_requests,
            2,
        )

        self.assertEqual(
            groq.successful_requests,
            2,
        )

        self.assertEqual(
            groq.failed_requests,
            0,
        )

        self.assertAlmostEqual(
            groq.failure_rate,
            0.0,
        )

        self.assertAlmostEqual(
            groq.average_latency_s,
            1.5,
        )

        self.assertAlmostEqual(
            groq.total_cost_usd,
            0.003,
        )

        mistral = health["mistral"]

        self.assertEqual(
            mistral.total_requests,
            1,
        )

        self.assertEqual(
            mistral.successful_requests,
            0,
        )

        self.assertEqual(
            mistral.failed_requests,
            1,
        )

        self.assertAlmostEqual(
            mistral.failure_rate,
            1.0,
        )

    def test_fallback_requests_are_attributed_to_routed_provider(self):
        log_request(
            request_id="fallback-1",
            prompt="fallback request",
            tier=1,
            primary_model="mistral-small",
            routed_model="groq-gpt-oss-20b",
            used_fallback=True,
            response=_response(
                cost=0.001,
                input_tokens=100,
                output_tokens=50,
                latency=1.2,
                model="groq-gpt-oss-20b",
            ),
        )

        health = get_provider_health()

        groq = health["groq"]

        self.assertEqual(
            groq.total_requests,
            1,
        )

        self.assertEqual(
            groq.successful_requests,
            1,
        )

        self.assertEqual(
            groq.fallback_requests,
            1,
        )

        self.assertNotIn(
            "mistral",
            health,
        )

    def test_unknown_provider_returns_empty_metrics(self):
        health = get_provider_health_for("unknown-provider")

        self.assertEqual(
            health.provider,
            "unknown-provider",
        )

        self.assertEqual(
            health.total_requests,
            0,
        )

        self.assertEqual(
            health.successful_requests,
            0,
        )

        self.assertEqual(
            health.failed_requests,
            0,
        )

        self.assertEqual(
            health.failure_rate,
            0.0,
        )

    def test_empty_database(self):
        health = get_provider_health()

        self.assertEqual(
            health,
            {},
        )


if __name__ == "__main__":
    unittest.main()
