"""End-to-end API tests for the LLM Cost Autopilot service."""

import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.classifier.predict import ComplexityPrediction
from src.models.response import Response
from src.routing import RoutingResult


client = TestClient(app)

AUTH_HEADERS = {
    "Authorization": "Bearer test-api-key",
}


def _mock_successful_routing_result(
    classifier_tier: int = 2,
) -> RoutingResult:
    """Build a deterministic successful routing result."""

    response = Response(
        output_text=(
            "A circuit breaker prevents repeated calls to an unhealthy provider."
        ),
        input_tokens=12,
        output_tokens=14,
        latency_s=0.42,
        cost_usd=0.0000051,
        model_name="groq-gpt-oss-20b",
        provider="groq",
    )

    return RoutingResult(
        request_id="e2e-test-request-id",
        response=response,
        tier=classifier_tier,
        primary_model="groq-gpt-oss-20b",
        routed_model="groq-gpt-oss-20b",
        used_fallback=False,
        classifier_tier=classifier_tier,
        classification_confidence=0.99,
        low_confidence=False,
    )


def _mock_verification():
    """Build a deterministic verification result."""

    return SimpleNamespace(
        escalated=False,
        final_response=Response(
            output_text="Verified circuit breaker explanation.",
            input_tokens=12,
            output_tokens=8,
            latency_s=0.55,
            cost_usd=0.000004,
            model_name="groq-gpt-oss-20b",
            provider="groq",
        ),
    )


def test_e2e_authenticated_completion_returns_expected_response(
    monkeypatch,
):
    """Validate the complete FastAPI completion response path."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    result = _mock_successful_routing_result()
    verification = _mock_verification()

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=(result, verification),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain circuit breakers in distributed systems.",
                    }
                ]
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == "e2e-test-request-id"

    assert len(data["choices"]) == 1
    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "circuit breaker" in data["choices"][0]["message"]["content"]

    routing = data["routing"]

    assert routing["request_id"] == "e2e-test-request-id"
    assert routing["tier"] == 2
    assert routing["selected_model"] == "groq-gpt-oss-20b"
    assert routing["used_fallback"] is False
    assert routing["cost_usd"] == 0.0000051
    assert routing["latency_s"] == 0.42
    assert routing["verification"] == "queued"


def test_e2e_wait_for_verification_returns_verification_status(
    monkeypatch,
):
    """Validate the synchronous verification response path."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    result = _mock_successful_routing_result()
    verification = _mock_verification()

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=(result, verification),
    ) as mock_route:
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain circuit breakers.",
                    }
                ],
                "wait_for_verification": True,
            },
        )

    assert response.status_code == 200

    mock_route.assert_called_once_with(
        "Explain circuit breakers.",
        synchronous=True,
    )

    data = response.json()

    assert data["choices"][0]["message"]["content"] == (
        "Verified circuit breaker explanation."
    )
    assert data["routing"]["verification"] == "passed"


def test_e2e_tier_three_skips_verification(monkeypatch):
    """Validate that classifier tier 3 requests bypass verification."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    result = _mock_successful_routing_result(classifier_tier=3)

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=(result, None),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Solve this complex reasoning problem.",
                    }
                ]
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["routing"]["tier"] == 3
    assert data["routing"]["verification"] == "skipped (tier 3)"


def test_e2e_completion_does_not_expose_prompt(
    monkeypatch,
):
    """Ensure the raw user prompt is not returned in the response."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    secret_prompt = "PRIVATE-E2E-PROMPT-DO-NOT-RETURN"

    result = _mock_successful_routing_result()
    verification = _mock_verification()

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=(result, verification),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": secret_prompt,
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert secret_prompt not in response.text


def test_e2e_successful_request_is_persisted_to_audit_db(
    monkeypatch,
    tmp_path,
):
    """Validate successful request persistence and Phase 3C metadata."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    db_path = tmp_path / "requests.db"

    monkeypatch.setattr(
        "src.logging_db.DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        "src.routing.predict_complexity",
        lambda prompt: ComplexityPrediction(
            tier=3,
            confidence=0.99,
            probabilities={1: 0.0, 2: 0.01, 3: 0.99},
        ),
    )

    provider_response = Response(
        output_text="Circuit breakers isolate unhealthy providers.",
        input_tokens=10,
        output_tokens=9,
        latency_s=0.31,
        cost_usd=0.0000042,
        model_name="groq-gpt-oss-20b",
        provider="groq",
    )

    with patch(
        "src.routing.send_request",
        return_value=provider_response,
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain circuit breakers.",
                    }
                ]
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["routing"]["tier"] == 3
    assert data["routing"]["selected_model"] == "groq-gpt-oss-20b"
    assert data["routing"]["verification"] == "skipped (tier 3)"

    request_id = data["id"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT
                request_id,
                prompt_hash,
                tier,
                classifier_tier,
                classification_confidence,
                low_confidence,
                primary_model,
                routed_model,
                used_fallback,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_s,
                error_type,
                primary_error_type,
                circuit_state,
                verified
            FROM request_log
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None

    assert row["request_id"] == request_id
    assert row["tier"] == 3

    assert row["classifier_tier"] == 3
    assert row["classification_confidence"] == 0.99
    assert row["low_confidence"] == 0

    assert row["primary_model"] == "groq-gpt-oss-20b"
    assert row["routed_model"] == "groq-gpt-oss-20b"

    assert row["used_fallback"] == 0

    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 9

    assert row["cost_usd"] == 0.0000042
    assert row["latency_s"] == 0.31

    assert row["error_type"] is None
    assert row["primary_error_type"] is None

    assert row["circuit_state"] == "closed"
    assert row["verified"] == 0

    assert row["prompt_hash"] is not None
    assert len(row["prompt_hash"]) == 16

    assert "Explain circuit breakers." not in row["prompt_hash"]
