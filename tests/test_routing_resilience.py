from unittest.mock import patch

from src.models.registry import get_model
from src.models.response import Response
from src.resilience import CircuitRegistry, CircuitState
from src.routing import route_request


def _success_response(model_name: str, provider: str) -> Response:
    return Response(
        output_text="successful response",
        input_tokens=10,
        output_tokens=20,
        latency_s=0.1,
        cost_usd=0.001,
        model_name=model_name,
        provider=provider,
    )


def _failure_response(
    model_name: str,
    provider: str,
    error_type: str,
) -> Response:
    return Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=0.1,
        cost_usd=0.0,
        model_name=model_name,
        provider=provider,
        error=f"simulated {error_type}",
        error_type=error_type,
    )


def test_success_closes_primary_circuit():
    with patch(
        "src.routing._call_model",
        return_value=_success_response(
            "groq-gpt-oss-20b",
            "groq",
        ),
    ):
        result = route_request("Explain what an API is.")

    assert result.response.error is None
    assert result.used_fallback is False


def test_primary_failure_triggers_fallback():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    responses = [
        _failure_response(
            primary.name,
            "mistral",
            "rate_limit",
        ),
        _success_response(
            fallback.name,
            "groq",
        ),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ):
        result = route_request("Write a short explanation of Python.")

    assert result.used_fallback is True
    assert result.routed_model == "groq-gpt-oss-20b"
    assert result.response.error is None


def test_timeout_triggers_fallback():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    from src.resilience import ProviderTimeoutError

    responses = [
        ProviderTimeoutError("simulated timeout"),
        _success_response(
            fallback.name,
            "groq",
        ),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ):
        result = route_request("Explain recursion.")

    assert result.used_fallback is True
    assert result.routed_model == "groq-gpt-oss-20b"
    assert result.response.error is None


def test_circuit_open_triggers_fallback():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    from src.resilience import CircuitOpenError

    responses = [
        CircuitOpenError("simulated open circuit"),
        _success_response(
            fallback.name,
            "groq",
        ),
    ]

    with patch(
        "src.routing._call_model",
        side_effect=responses,
    ):
        result = route_request("What is machine learning?")

    assert result.used_fallback is True
    assert result.routed_model == "groq-gpt-oss-20b"
    assert result.response.error is None
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.classifier.predict import ComplexityPrediction


def test_fallback_creates_one_audit_row_with_final_response_accounting():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    primary_failure = _failure_response(
        primary.name,
        "mistral",
        "rate_limit",
    )

    fallback_success = Response(
        output_text="fallback response",
        input_tokens=37,
        output_tokens=19,
        latency_s=0.25,
        cost_usd=0.000123,
        model_name=fallback.name,
        provider="groq",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "requests.db"

        with (
            patch(
                "src.logging_db.DB_PATH",
                db_path,
            ),
            patch(
                "src.logging_db.DATA_DIR",
                Path(tmp_dir),
            ),
            patch(
                "src.routing.predict_complexity",
                return_value=ComplexityPrediction(
                    tier=1,
                    confidence=0.99,
                    probabilities={
                        1: 0.99,
                        2: 0.01,
                        3: 0.0,
                    },
                ),
            ),
            patch(
                "src.routing._call_model",
                side_effect=[
                    primary_failure,
                    fallback_success,
                ],
            ),
        ):
            result = route_request(
                "Explain why fallback routing is useful."
            )

        assert result.used_fallback is True
        assert result.primary_model == primary.name
        assert result.routed_model == fallback.name
        assert result.response is fallback_success

        conn = sqlite3.connect(db_path)

        try:
            rows = conn.execute(
                """
                SELECT
                    primary_model,
                    routed_model,
                    used_fallback,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    error_type,
                    primary_error_type
                FROM request_log
                """
            ).fetchall()
        finally:
            conn.close()

    # One logical request must produce exactly one audit row.
    assert len(rows) == 1

    (
        primary_model,
        routed_model,
        used_fallback,
        input_tokens,
        output_tokens,
        cost_usd,
        error_type,
        primary_error_type,
    ) = rows[0]

    assert primary_model == primary.name
    assert routed_model == fallback.name
    assert used_fallback == 1

    # Accounting belongs to the final successful response.
    assert input_tokens == fallback_success.input_tokens
    assert output_tokens == fallback_success.output_tokens
    assert cost_usd == fallback_success.cost_usd

    # The primary failure is retained separately.
    assert error_type is None
    assert primary_error_type == "rate_limit"


def test_low_confidence_t3_remains_t3_and_uses_t3_route():
    t3_model = get_model("groq-gpt-oss-20b")

    prediction = ComplexityPrediction(
        tier=3,
        confidence=0.40,
        probabilities={
            1: 0.10,
            2: 0.50,
            3: 0.40,
        },
    )

    with (
        patch(
            "src.routing.predict_complexity",
            return_value=prediction,
        ),
        patch(
            "src.routing._call_model",
            return_value=_success_response(
                t3_model.name,
                "groq",
            ),
        ),
        patch(
            "src.routing.log_request",
        ) as mock_log,
    ):
        result = route_request(
            "Design a production LLM routing architecture."
        )

    assert result.classifier_tier == 3
    assert result.classification_confidence == 0.40
    assert result.low_confidence is True
    assert result.tier == 3
    assert result.primary_model == "groq-gpt-oss-20b"

    logged = mock_log.call_args.kwargs

    assert logged["classifier_tier"] == 3
    assert logged["classification_confidence"] == 0.40
    assert logged["low_confidence"] is True
    assert logged["tier"] == 3


def test_low_confidence_t1_promotes_to_t2_before_model_selection():
    prediction = ComplexityPrediction(
        tier=1,
        confidence=0.70,
        probabilities={
            1: 0.70,
            2: 0.25,
            3: 0.05,
        },
    )

    with (
        patch(
            "src.routing.predict_complexity",
            return_value=prediction,
        ),
        patch(
            "src.routing._call_model",
            return_value=_success_response(
                "groq-gpt-oss-20b",
                "groq",
            ),
        ),
        patch(
            "src.routing.log_request",
        ) as mock_log,
    ):
        result = route_request(
            "Explain an API gateway."
        )

    assert result.classifier_tier == 1
    assert result.low_confidence is True
    assert result.tier == 2

    # routing.yaml currently maps T2 to Groq.
    assert result.primary_model == "groq-gpt-oss-20b"

    logged = mock_log.call_args.kwargs

    assert logged["classifier_tier"] == 1
    assert logged["tier"] == 2
    assert logged["low_confidence"] is True
def test_primary_and_fallback_failure_preserve_terminal_error_and_single_audit_row():
    primary = get_model("mistral-small")
    fallback = get_model("groq-gpt-oss-20b")

    primary_failure = _failure_response(
        primary.name,
        "mistral",
        "rate_limit",
    )

    fallback_failure = _failure_response(
        fallback.name,
        "groq",
        "provider_unavailable",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "requests.db"

        with (
            patch(
                "src.logging_db.DB_PATH",
                db_path,
            ),
            patch(
                "src.logging_db.DATA_DIR",
                Path(tmp_dir),
            ),
            patch(
                "src.routing.predict_complexity",
                return_value=ComplexityPrediction(
                    tier=1,
                    confidence=0.99,
                    probabilities={
                        1: 0.99,
                        2: 0.01,
                        3: 0.0,
                    },
                ),
            ),
            patch(
                "src.routing._call_model",
                side_effect=[
                    primary_failure,
                    fallback_failure,
                ],
            ) as mock_call,
        ):
            result = route_request(
                "Explain what happens when both providers fail."
            )

        assert mock_call.call_count == 2

        assert result.used_fallback is True
        assert result.primary_model == primary.name
        assert result.routed_model == fallback.name

        assert result.response is fallback_failure
        assert result.response.error_type == "provider_unavailable"

        conn = sqlite3.connect(db_path)

        try:
            rows = conn.execute(
                """
                SELECT
                    primary_model,
                    routed_model,
                    used_fallback,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    error_type,
                    primary_error_type
                FROM request_log
                """
            ).fetchall()
        finally:
            conn.close()

    assert len(rows) == 1

    (
        primary_model,
        routed_model,
        used_fallback,
        input_tokens,
        output_tokens,
        cost_usd,
        error_type,
        primary_error_type,
    ) = rows[0]

    assert primary_model == primary.name
    assert routed_model == fallback.name
    assert used_fallback == 1

    assert input_tokens == fallback_failure.input_tokens
    assert output_tokens == fallback_failure.output_tokens
    assert cost_usd == fallback_failure.cost_usd

    assert error_type == "provider_unavailable"
    assert primary_error_type == "rate_limit"
