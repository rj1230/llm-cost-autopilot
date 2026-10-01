"""Classifier inference helpers for LLM Cost Autopilot."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from src.classifier.features import features_to_vector

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODEL_PATH = DATA_DIR / "classifier_v4_final.joblib"

CONFIDENCE_THRESHOLD = 0.85


@dataclass(frozen=True)
class ComplexityPrediction:
    """
    Raw classifier prediction plus confidence information.

    `tier` is the ML classifier's predicted complexity tier.
    `confidence` is the probability assigned to that predicted tier.
    `probabilities` contains the full class probability distribution.
    """

    tier: int
    confidence: float
    probabilities: dict[int, float]

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < CONFIDENCE_THRESHOLD

    @property
    def routing_tier(self) -> int:
        """
        Apply the confidence-aware routing safety policy.

        Low-confidence T1/T2 predictions are promoted by one tier.
        T3 remains T3 because there is no safer higher tier.
        """
        if not self.is_low_confidence:
            return self.tier

        return min(self.tier + 1, 3)


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found - the frozen V4 classifier artifact is required."
        )

    bundle = joblib.load(MODEL_PATH)
    return bundle["model"]


def predict_complexity(prompt: str) -> ComplexityPrediction:
    """
    Return the classifier prediction, confidence, and class probabilities.
    """
    model = _load_model()
    vector = features_to_vector(prompt)

    probabilities = model.predict_proba([vector])[0]
    predicted_index = int(probabilities.argmax())

    tier = int(model.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])

    probability_map = {
        int(class_id): float(probability)
        for class_id, probability in zip(model.classes_, probabilities)
    }

    return ComplexityPrediction(
        tier=tier,
        confidence=confidence,
        probabilities=probability_map,
    )


def classify_complexity(prompt: str) -> int:
    """
    Preserve the raw classifier API.

    This intentionally returns the ML prediction without applying
    confidence-aware routing. The routing layer decides whether the
    prediction should be promoted to a safer routing tier.
    """
    return predict_complexity(prompt).tier
