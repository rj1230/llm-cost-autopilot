"""
Phase 3, step 4: "Every routing failure becomes a new training example for
the complexity classifier."

A failed verification means the tier the classifier picked was too cheap for
this prompt - it needed the reference (highest-tier) model's quality. So the
corrected label is always the reference tier: 3 if tier 3 is what caught the
gap, per config/routing.yaml.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FEEDBACK_PATH = DATA_DIR / "classifier_feedback.csv"


def append_feedback(prompt: str, corrected_tier: int) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    is_new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["prompt", "tier"])
        writer.writerow([prompt, corrected_tier])
