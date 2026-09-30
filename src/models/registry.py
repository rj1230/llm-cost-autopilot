"""
Model registry for the LLM Cost Autopilot.

The router selects live models from this registry. Pricing is stored per
token so downstream cost calculations remain simple multiplications.

Live providers:
    - Mistral
    - Groq

OpenAI is retained only as a pricing baseline for cost comparison and is
never called by the live request dispatcher.
"""

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class Provider(str, Enum):
    MISTRAL = "mistral"
    GROQ = "groq"
    OPENAI = "openai"  # Pricing-only baseline; never used for live requests.


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: Provider
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_s: float
    quality_tier: QualityTier

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated USD cost for a request."""
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


def _per_million(usd_per_million: float) -> float:
    """Convert USD per million tokens into USD per token."""
    return usd_per_million / 1_000_000


GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "mistral-small": ModelConfig(
        name="mistral-small",
        provider=Provider.MISTRAL,
        model_id="mistral-small-latest",
        cost_per_input_token=_per_million(0.15),
        cost_per_output_token=_per_million(0.60),
        avg_latency_s=1.5,
        quality_tier=QualityTier.MEDIUM,
    ),
    "groq-gpt-oss-20b": ModelConfig(
        name="groq-gpt-oss-20b",
        provider=Provider.GROQ,
        model_id=GROQ_MODEL,
        cost_per_input_token=_per_million(0.075),
        cost_per_output_token=_per_million(0.30),
        avg_latency_s=1.0,
        quality_tier=QualityTier.HIGH,
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        cost_per_input_token=_per_million(2.50),
        cost_per_output_token=_per_million(10.00),
        avg_latency_s=2.0,
        quality_tier=QualityTier.HIGH,
    ),
}


def get_model(name: str) -> ModelConfig:
    """Return a registered model by its router name."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc
