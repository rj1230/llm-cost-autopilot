"""Groq provider implementation."""

import time

from groq import Groq

from src import config
from src.models.registry import ModelConfig
from src.models.response import Response
from src.providers.base import BaseProvider
from src.providers.error_utils import classify_provider_error


class GroqProvider(BaseProvider):
    """Groq provider with standardized responses and SDK timeout."""

    def __init__(self) -> None:
        self._client: Groq | None = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            self._client = Groq(
                api_key=config.GROQ_API_KEY,
                timeout=config.PROVIDER_TIMEOUT_S,
            )

        return self._client

    def call(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> Response:
        start = time.perf_counter()

        try:
            completion = self.client.chat.completions.create(
                model=model_config.model_id or config.GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

        except Exception as exc:  # noqa: BLE001
            return Response(
                output_text="",
                input_tokens=0,
                output_tokens=0,
                latency_s=time.perf_counter() - start,
                cost_usd=0.0,
                model_name=model_config.name,
                provider="groq",
                error=str(exc),
                error_type=classify_provider_error(exc),
            )

        latency = time.perf_counter() - start

        usage = completion.usage

        input_tokens = usage.prompt_tokens if usage else 0

        output_tokens = usage.completion_tokens if usage else 0

        return Response(
            output_text=completion.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost_usd=model_config.cost_for(
                input_tokens,
                output_tokens,
            ),
            model_name=model_config.name,
            provider="groq",
            error=None,
            error_type=None,
        )
