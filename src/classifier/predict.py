"""Loads the model train.py saved and exposes a single classify_complexity()
call for routing.py (and anything else) to use."""

from functools import lru_cache
from pathlib import Path

import joblib

from src.classifier.features import features_to_vector

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "classifier.joblib"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found - run `python -m src.classifier.train` first "
            "(and `python -m data.generate_dataset` before that if you haven't)."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"]


def classify_complexity(prompt: str) -> int:
    """Returns 1 (simple), 2 (moderate), or 3 (complex)."""
    model = _load_model()
    vector = features_to_vector(prompt)
    return int(model.predict([vector])[0])
