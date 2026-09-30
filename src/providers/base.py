"""Every provider client implements this same interface, so client.py never
needs to know which provider it's talking to."""

from abc import ABC, abstractmethod

from src.models.registry import ModelConfig
from src.models.response import Response


class BaseProvider(ABC):
    @abstractmethod
    def call(self, prompt: str, model_config: ModelConfig) -> Response:
        """Send prompt to the provider and return a standardized Response.

        Implementations should never raise on an API/network error - catch it
        and return a Response with error set instead, so a bad call from one
        model doesn't crash a batch run across the whole registry.
        """
        raise NotImplementedError
