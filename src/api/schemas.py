"""Request and response schemas for the LLM Cost Autopilot API."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGES = 50
MAX_MESSAGE_CONTENT_LENGTH = 16_000


class MessageRole(str, Enum):
    """Supported chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """One message in a completion request."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_CONTENT_LENGTH,
        description="Message content.",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject whitespace-only message content."""

        value = value.strip()

        if not value:
            raise ValueError("message content must not be empty")

        return value


class CompletionRequest(BaseModel):
    """Request body for the provider-agnostic completion endpoint."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGES,
        description="Conversation messages. At least one message is required.",
    )
    wait_for_verification: bool = Field(
        default=False,
        description=(
            "If true, wait for quality verification before responding. "
            "If false, verification runs asynchronously for eligible tiers."
        ),
    )


class RoutingMetadata(BaseModel):
    """Routing and operational metadata returned with a completion."""

    model_config = ConfigDict(extra="forbid")

    request_id: str

    # Final routing decision after the confidence-aware safety policy.
    tier: int = Field(..., ge=1, le=3)

    # Raw ML classifier decision before safety promotion.
    classifier_tier: int = Field(..., ge=1, le=3)

    # Probability assigned to the raw classifier prediction.
    classification_confidence: float = Field(..., ge=0, le=1)

    # Whether the classifier prediction was below the confidence threshold.
    low_confidence: bool

    selected_model: str
    reasoning: str
    used_fallback: bool
    cost_usd: float = Field(..., ge=0)
    latency_s: float = Field(..., ge=0)
    verification: str


class ChatChoice(BaseModel):
    """Assistant completion choice."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(default=0, ge=0)
    message: ChatMessage
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    """Standardized completion response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    choices: list[ChatChoice] = Field(
        ...,
        min_length=1,
    )
    routing: RoutingMetadata


class ModelInfo(BaseModel):
    """Public model registry information."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    cost_per_input_token: float = Field(..., ge=0)
    cost_per_output_token: float = Field(..., ge=0)
    quality_tier: str


class RoutingConfigUpdate(BaseModel):
    """Runtime routing policy update."""

    model_config = ConfigDict(extra="forbid")

    routing: dict[int, str]
    fallback: dict[int, str] = Field(default_factory=dict)

    @field_validator("routing")
    @classmethod
    def validate_routing_tiers(cls, value: dict[int, str]) -> dict[int, str]:
        """Require exactly the supported routing tiers."""

        expected = {1, 2, 3}

        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)

            parts = []

            if missing:
                parts.append(f"missing tiers: {missing}")

            if extra:
                parts.append(f"unsupported tiers: {extra}")

            raise ValueError(
                "routing must contain exactly tiers 1, 2, and 3"
                + (f" ({'; '.join(parts)})" if parts else "")
            )

        return value

    @field_validator("fallback")
    @classmethod
    def validate_fallback_tiers(
        cls,
        value: dict[int, str],
    ) -> dict[int, str]:
        """Allow fallback configuration only for supported tiers."""

        unsupported = sorted(set(value) - {1, 2, 3})

        if unsupported:
            raise ValueError(f"fallback contains unsupported tiers: {unsupported}")

        return value
