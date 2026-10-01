"""Failure-injection tests for the LLM Cost Autopilot API."""

import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.classifier.predict import ComplexityPrediction
from src.models.registry import get_model
from src.models.response import Response
from src.resilience import ProviderTimeoutError


client = TestClient(app)

AUTH_HEADERS = {
    "Authorization": "Bearer test-api-key",
}


def _failure_response() -> Response:
    """Build a deterministic primary-provider failure."""

    model = get_model("mistral-small")

    return Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=0.1,
        cost_usd=0.0,
        model_name=model.name,
        provider=model.provider.value,
        error="simulated rate limit",
        error_type="rate_limit",
    )


def _fallback_success_response() -> Response:
    """Build a deterministic fallback-provider success."""

    model = get_model("groq-gpt-oss-20b")

    return Response(
        output_text="Fallback response succeeded.",
        input_tokens=11,
        output_tokens=8,
        latency_s=0.25,
        cost_usd=0.0000032,
        model_name=model.name,
        provider=model.provider.value,
    )


def _tier_one_prediction() -> ComplexityPrediction:
    """Build a deterministic high-confidence T1 prediction."""

    return ComplexityPrediction(
        tier=1,
        confidence=0.99,
        probabilities={
            1: 0.99,
            2: 0.01,
            3: 0.0,
        },
    )


def test_api_primary_failure_triggers_fallback_and_persists_audit(
    monkeypatch,
    tmp_path,
):
    """Validate API fallback behavior and persisted failure metadata."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    db_path = tmp_path / "requests.db"

    monkeypatch.setattr(
        "src.logging_db.DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        "src.routing.predict_complexity",
        lambda prompt: _tier_one_prediction(),
    )

    responses = [
        _failure_response(),
        _fallback_success_response(),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ) as mock_call:
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain what an API is.",
                    }
                ]
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == ("Fallback response succeeded.")

    routing = data["routing"]

    assert routing["tier"] == 1
    assert routing["selected_model"] == "groq-gpt-oss-20b"
    assert routing["used_fallback"] is True
    assert routing["cost_usd"] == 0.0000032
    assert routing["latency_s"] == 0.25

    assert mock_call.call_count == 2

    request_id = data["id"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT
                request_id,
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
    assert row["tier"] == 1

    assert row["classifier_tier"] == 1
    assert row["classification_confidence"] == 0.99
    assert row["low_confidence"] == 0

    assert row["primary_model"] == "mistral-small"
    assert row["routed_model"] == "groq-gpt-oss-20b"

    assert row["used_fallback"] == 1

    assert row["input_tokens"] == 11
    assert row["output_tokens"] == 8
    assert row["cost_usd"] == 0.0000032
    assert row["latency_s"] == 0.25

    assert row["primary_error_type"] == "rate_limit"

    assert row["error_type"] is None

    assert row["verified"] == 0


def test_api_timeout_triggers_fallback_and_persists_audit(
    monkeypatch,
    tmp_path,
):
    """Validate timeout handling, fallback, and persisted timeout metadata."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    db_path = tmp_path / "requests.db"

    monkeypatch.setattr(
        "src.logging_db.DB_PATH",
        db_path,
    )

    monkeypatch.setattr(
        "src.routing.predict_complexity",
        lambda prompt: _tier_one_prediction(),
    )

    responses = [
        ProviderTimeoutError("simulated timeout"),
        _fallback_success_response(),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ) as mock_call:
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain API timeouts.",
                    }
                ]
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == ("Fallback response succeeded.")

    routing = data["routing"]

    assert routing["tier"] == 1
    assert routing["selected_model"] == "groq-gpt-oss-20b"
    assert routing["used_fallback"] is True
    assert routing["cost_usd"] == 0.0000032
    assert routing["latency_s"] == 0.25

    assert mock_call.call_count == 2

    request_id = data["id"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT
                request_id,
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
    assert row["tier"] == 1

    assert row["classifier_tier"] == 1
    assert row["classification_confidence"] == 0.99
    assert row["low_confidence"] == 0

    assert row["primary_model"] == "mistral-small"
    assert row["routed_model"] == "groq-gpt-oss-20b"

    assert row["used_fallback"] == 1

    assert row["input_tokens"] == 11
    assert row["output_tokens"] == 8
    assert row["cost_usd"] == 0.0000032
    assert row["latency_s"] == 0.25

    assert row["primary_error_type"] == "timeout"

    assert row["error_type"] is None

    assert row["verified"] == 0


def test_fallback_is_called_with_correct_model_after_primary_failure(
    monkeypatch,
):
    """Verify fallback receives the request with the configured fallback model."""

    monkeypatch.setenv("API_KEY", "test-api-key")

    monkeypatch.setattr(
        "src.routing.predict_complexity",
        lambda prompt: _tier_one_prediction(),
    )

    primary_failure = _failure_response()
    fallback_success = _fallback_success_response()

    responses = [
        primary_failure,
        fallback_success,
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ) as mock_call:
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain why fallback routing is useful.",
                    }
                ]
            },
        )

    assert response.status_code == 200

    assert mock_call.call_count == 2

    first_call = mock_call.call_args_list[0]
    second_call = mock_call.call_args_list[1]

    assert first_call.args[0] == ("Explain why fallback routing is useful.")
    assert first_call.args[1].name == "mistral-small"

    assert second_call.args[0] == ("Explain why fallback routing is useful.")
    assert second_call.args[1].name == "groq-gpt-oss-20b"

    data = response.json()

    assert data["routing"]["tier"] == 1
    assert data["routing"]["selected_model"] == "groq-gpt-oss-20b"
    assert data["routing"]["used_fallback"] is True

    assert data["choices"][0]["message"]["content"] == ("Fallback response succeeded.")
