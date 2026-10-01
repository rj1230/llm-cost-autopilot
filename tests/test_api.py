from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)

AUTH_HEADERS = {
    "Authorization": "Bearer test-api-key",
}


def test_healthz_is_public():
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_requires_authentication():
    response = client.get("/v1/models")

    assert response.status_code == 401


def test_models_accepts_valid_authentication(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    response = client.get(
        "/v1/models",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) >= 2


def test_models_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    response = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer wrong-key",
        },
    )

    assert response.status_code == 401


def test_stats_requires_authentication():
    response = client.get("/v1/stats")

    assert response.status_code == 401


def test_routing_config_requires_authentication():
    response = client.put(
        "/v1/routing-config",
        json={
            "routing": {
                1: "mistral-small",
                2: "groq-gpt-oss-20b",
                3: "groq-gpt-oss-20b",
            },
            "fallback": {
                1: "groq-gpt-oss-20b",
                2: "mistral-small",
                3: "mistral-small",
            },
        },
    )

    assert response.status_code == 401


def test_routing_config_rejects_same_primary_and_fallback(
    monkeypatch,
):
    monkeypatch.setenv("API_KEY", "test-api-key")

    response = client.put(
        "/v1/routing-config",
        headers=AUTH_HEADERS,
        json={
            "routing": {
                1: "mistral-small",
                2: "groq-gpt-oss-20b",
                3: "groq-gpt-oss-20b",
            },
            "fallback": {
                1: "mistral-small",
            },
        },
    )

    assert response.status_code == 400


def test_completion_requires_authentication():
    response = client.post(
        "/v1/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]
        },
    )

    assert response.status_code == 401


def test_completion_validation_still_works_with_authentication(
    monkeypatch,
):
    monkeypatch.setenv("API_KEY", "test-api-key")

    response = client.post(
        "/v1/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [],
        },
    )

    assert response.status_code == 422


def test_readyz_returns_200_when_dependencies_are_ready():
    response = client.get("/readyz")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["classifier"] == "ok"
    assert data["checks"]["routing_config"] == "ok"
    assert data["checks"]["mistral_provider"] == "ok"
    assert data["checks"]["groq_provider"] == "ok"


def test_readyz_returns_503_when_provider_is_not_configured(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.api.main.config.GROQ_API_KEY",
        "",
    )

    response = client.get("/readyz")

    assert response.status_code == 503

    data = response.json()

    assert data["status"] == "not_ready"
    assert data["checks"]["groq_provider"] == "error"


def _mock_failed_routing_result(error_type: str):
    from src.models.response import Response
    from src.routing import RoutingResult

    response = Response(
        output_text="",
        input_tokens=0,
        output_tokens=0,
        latency_s=0.0,
        cost_usd=0.0,
        model_name="groq-gpt-oss-20b",
        provider="groq",
        error="PRIVATE_INTERNAL_PROVIDER_ERROR: secret details",
        error_type=error_type,
    )

    return RoutingResult(
        request_id="test-request-id",
        response=response,
        tier=2,
        primary_model="groq-gpt-oss-20b",
        routed_model="groq-gpt-oss-20b",
        used_fallback=False,
        classifier_tier=2,
        classification_confidence=0.95,
        low_confidence=False,
    ), None


def test_completion_maps_timeout_to_504(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=_mock_failed_routing_result("timeout"),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Test timeout",
                    }
                ]
            },
        )

    assert response.status_code == 504

    data = response.json()

    assert data["error_type"] == "timeout"
    assert data["detail"] == "The selected provider timed out."
    assert "PRIVATE_INTERNAL_PROVIDER_ERROR" not in str(data)


def test_completion_maps_circuit_open_to_503(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=_mock_failed_routing_result("circuit_open"),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Test circuit",
                    }
                ]
            },
        )

    assert response.status_code == 503

    data = response.json()

    assert data["error_type"] == "circuit_open"
    assert "PRIVATE_INTERNAL_PROVIDER_ERROR" not in str(data)


def test_completion_maps_rate_limit_to_503(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=_mock_failed_routing_result("rate_limit"),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Test rate limit",
                    }
                ]
            },
        )

    assert response.status_code == 503

    data = response.json()

    assert data["error_type"] == "rate_limit"
    assert "PRIVATE_INTERNAL_PROVIDER_ERROR" not in str(data)


def test_completion_maps_unknown_error_to_500(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")

    with patch(
        "src.api.main.route_request_with_verification",
        return_value=_mock_failed_routing_result("unknown"),
    ):
        response = client.post(
            "/v1/completions",
            headers=AUTH_HEADERS,
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Test unknown error",
                    }
                ]
            },
        )

    assert response.status_code == 500

    data = response.json()

    assert data["error_type"] == "unknown"
    assert data["detail"] == "An unexpected internal error occurred."
    assert "PRIVATE_INTERNAL_PROVIDER_ERROR" not in str(data)
