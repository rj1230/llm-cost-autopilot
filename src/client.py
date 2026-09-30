"""
Provider-agnostic LLM client.

The rest of the system calls:

    send_request(prompt, model_config) -> Response

The selected ModelConfig determines whether the request is sent to
Mistral or Groq. Ollama is intentionally not supported.
"""

from src.models.registry import ModelConfig, Provider
from src.models.response import Response
from src.providers.base import BaseProvider
from src.providers.groq_provider import GroqProvider
from src.providers.mistral_provider import MistralProvider

_PROVIDER_INSTANCES: dict[Provider, BaseProvider] = {
    Provider.MISTRAL: MistralProvider(),
    Provider.GROQ: GroqProvider(),
}


def send_request(prompt: str, model_config: ModelConfig) -> Response:
    provider = _PROVIDER_INSTANCES.get(model_config.provider)

    if provider is None:
        raise ValueError(
            f"Unsupported provider: {model_config.provider}. "
            "Supported providers are MISTRAL and GROQ."
        )

    return provider.call(prompt, model_config)
