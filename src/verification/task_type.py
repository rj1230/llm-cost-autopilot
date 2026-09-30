"""
Phase 3, step 1 needs a "request type" (extraction / summarization /
classification / other) to pick a quality threshold for. The classifier in
src/classifier/ predicts *complexity tier*, which is a different axis (how
hard the request is) from *task type* (what kind of task it is) - a
classification prompt can be simple or gnarly. So this is a second, separate
heuristic, deliberately lightweight since it only has to pick a threshold,
not drive routing.
"""

EXTRACTION_MARKERS = ["extract", "pull out", "find the"]
CLASSIFICATION_MARKERS = ["classify", "sentiment", "categorize", "category"]
SUMMARIZATION_MARKERS = ["summarize", "summary", "summarise", "tl;dr"]

TaskType = str  # "extraction" | "classification" | "summarization" | "other"


def infer_task_type(prompt: str) -> TaskType:
    text = prompt.lower()
    if any(marker in text for marker in EXTRACTION_MARKERS):
        return "extraction"
    if any(marker in text for marker in CLASSIFICATION_MARKERS):
        return "classification"
    if any(marker in text for marker in SUMMARIZATION_MARKERS):
        return "summarization"
    return "other"
