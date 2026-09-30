"""The standardized object every provider call returns, regardless of provider."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Response:
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    model_name: str
    provider: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str | None = None
    error_type: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "output_text": self.output_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 4),
            "cost_usd": round(self.cost_usd, 8),
            "model_name": self.model_name,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "error": self.error,
            "error_type": self.error_type,
        }
