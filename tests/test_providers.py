"""Mistral provider with bounded retry handling for transient API failures."""

import time

from mistralai.client import Mistral

from src import config
from src.models.registry import ModelConfig
from src.models.response import Response
from src.providers.base import BaseProvider


class MistralProvider(BaseProvider):
    """Mistral API provider with retry handling for transient failures."""

    MAX_RETRIES = 3
    INITIAL_BACKOFF_S = 1.0
    MAX_BACKOFF_S = 8.0

    def __init__(self) -> None:
        self._client: Mistral | None = None

    @property
    def client(self) -> Mistral:
        if self._client is None:
            self._client = Mistral(api_key=config.MISTRAL_API_KEY)

        return self._client

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        """Extract an HTTP status code from an SDK exception when available."""

        status_code = getattr(exc, "status_code", None)

        if isinstance(status_code, int):
            return status_code

        response = getattr(exc, "response", None)

        if response is not None:
            response_status = getattr(response, "status_code", None)

            if isinstance(response_status, int):
                return response_status

        return None

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        """Extract Retry-After from an SDK exception when available."""

        response = getattr(exc, "response", None)

        if response is None:
            return None

        headers = getattr(response, "headers", None)

        if not headers:
            return None

        value = headers.get("Retry-After")

        if value is None:
            return None

        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        """Return True for transient HTTP failures."""

        status_code = cls._status_code(exc)

        if status_code is not None:
            return status_code == 429 or 500 <= status_code <= 599

        # The Mistral SDK may expose the status only in the exception text.
        message = str(exc).lower()

        return (
            "429" in message
            or "rate limit" in message
            or "rate_limited" in message
            or "500" in message
            or "502" in message
            or "503" in message
            or "504" in message
        )

    @classmethod
    def _backoff_seconds(
        cls,
        attempt: int,
        exc: Exception,
    ) -> float:
        """Return Retry-After when available, otherwise exponential backoff."""

        retry_after = cls._retry_after(exc)

        if retry_after is not None:
            return min(retry_after, cls.MAX_BACKOFF_S)

        delay = cls.INITIAL_BACKOFF_S * (2**attempt)

        return min(delay, cls.MAX_BACKOFF_S)

    def call(self, prompt: str, model_config: ModelConfig) -> Response:
        """Call Mistral and retry transient failures with bounded backoff."""

        start = time.perf_counter()
        last_exception: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                message = self.client.chat.complete(
                    model=model_config.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )

                latency = time.perf_counter() - start
                usage = message.usage

                input_tokens = usage.prompt_tokens if usage else 0

                output_tokens = usage.completion_tokens if usage else 0

                output_text = message.choices[0].message.content or ""

                return Response(
                    output_text=output_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_s=latency,
                    cost_usd=model_config.cost_for(
                        input_tokens,
                        output_tokens,
                    ),
                    model_name=model_config.name,
                    provider="mistral",
                )

            except Exception as exc:  # noqa: BLE001
                last_exception = exc

                if attempt >= self.MAX_RETRIES or not self._is_retryable(exc):
                    break

                delay = self._backoff_seconds(attempt, exc)

                time.sleep(delay)

        return Response(
            output_text="",
            input_tokens=0,
            output_tokens=0,
            latency_s=time.perf_counter() - start,
            cost_usd=0.0,
            model_name=model_config.name,
            provider="mistral",
            error=str(last_exception),
        )
