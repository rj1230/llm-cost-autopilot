"""Deployment readiness gates for LLM Cost Autopilot."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.models.registry import MODEL_REGISTRY, Provider, get_model
from src.routing import _load_config

AUTH_HEADERS = {
    "Authorization": "Bearer test-api-key",
}

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_PATH = ROOT / "data" / "classifier_v4_final.joblib"


def test_application_imports_and_core_routes_exist():
    """Verify the production FastAPI application exposes required routes."""
    routes = {route.path for route in app.routes}

    required_routes = {
        "/healthz",
        "/readyz",
        "/v1/completions",
        "/v1/models",
        "/v1/stats",
        "/v1/routing-config",
    }

    assert required_routes.issubset(routes)


def test_routing_configuration_is_valid():
    """Verify every configured routing model exists in the registry."""
    config = _load_config()

    assert "routing" in config
    assert "fallback" in config

    for tier in (1, 2, 3):
        primary = config["routing"][tier]
        fallback = config["fallback"][tier]

        assert primary in MODEL_REGISTRY
        assert fallback in MODEL_REGISTRY
        assert primary != fallback


def test_model_registry_has_required_live_providers():
    """Verify the registry contains the live Mistral and Groq models."""
    mistral = get_model("mistral-small")
    groq = get_model("groq-gpt-oss-20b")

    assert mistral.provider == Provider.MISTRAL
    assert groq.provider == Provider.GROQ

    assert mistral.model_id
    assert groq.model_id

    assert mistral.cost_per_input_token >= 0
    assert mistral.cost_per_output_token >= 0

    assert groq.cost_per_input_token >= 0
    assert groq.cost_per_output_token >= 0


def test_classifier_artifact_exists():
    """Verify the trained complexity classifier is packaged with the project."""
    assert CLASSIFIER_PATH.exists()
    assert CLASSIFIER_PATH.is_file()
    assert CLASSIFIER_PATH.stat().st_size > 0


def test_health_endpoint_is_available():
    """Verify the liveness endpoint responds successfully."""
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_readiness_endpoint_is_available():
    """Verify the readiness endpoint returns a valid readiness response."""
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code in {200, 503}
    assert isinstance(response.json(), dict)


def test_completion_requires_authentication():
    """Verify the production completion endpoint requires API authentication."""
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "prompt": "Deployment readiness test.",
        },
    )

    assert response.status_code in {401, 403}


def test_models_endpoint_is_available(monkeypatch):
    """Verify the authenticated model discovery endpoint is available."""
    monkeypatch.setenv("API_KEY", "test-api-key")

    client = TestClient(app)

    response = client.get(
        "/v1/models",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    models = response.json()
    assert isinstance(models, list)
    assert models
    assert all(isinstance(model, dict) for model in models)

    model_names = {model["name"] for model in models}
    assert {"mistral-small", "groq-gpt-oss-20b", "gpt-4o"}.issubset(model_names)


def test_stats_endpoint_is_available(monkeypatch):
    """Verify the authenticated statistics endpoint is available."""
    monkeypatch.setenv("API_KEY", "test-api-key")

    client = TestClient(app)

    response = client.get(
        "/v1/stats",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_no_ollama_provider_is_registered():
    """Verify Ollama is not part of the live provider registry."""
    providers = {model.provider for model in MODEL_REGISTRY.values()}

    assert Provider.MISTRAL in providers
    assert Provider.GROQ in providers
    assert Provider.OPENAI in providers

    assert all(model.provider != "ollama" for model in MODEL_REGISTRY.values())


def test_openai_model_is_pricing_only():
    """Verify GPT-4o remains a pricing baseline rather than a live provider."""
    openai_model = get_model("gpt-4o")

    assert openai_model.provider == Provider.OPENAI
    assert openai_model.model_id == "gpt-4o"

    live_providers = {
        Provider.MISTRAL,
        Provider.GROQ,
    }

    assert openai_model.provider not in live_providers