"""
Phase 3, step 1: "Define quality thresholds per use case."

    extraction:     did it get all the key fields? -> word-overlap score,
                     high bar (partial credit for close-but-not-exact)
    classification:  does the label match what the top-tier model would have
                      said? -> exact match, no partial credit
    summarization:   LLM-as-judge score above 4/5 -> 0.80 on our 0-1 scale
    other:           same LLM-as-judge mechanism, slightly looser bar

All scores are normalized to 0.0-1.0 so one threshold table works regardless
of which scoring method produced the number - see scoring.py.
"""

QUALITY_THRESHOLDS: dict[str, float] = {
    "extraction": 0.90,
    "classification": 1.00,
    "summarization": 0.80,  # LLM-as-judge >= 4/5
    "other": 0.75,
}


def passes_threshold(task_type: str, score: float) -> bool:
    threshold = QUALITY_THRESHOLDS.get(task_type, QUALITY_THRESHOLDS["other"])
    return score >= threshold


def threshold_for(task_type: str) -> float:
    return QUALITY_THRESHOLDS.get(task_type, QUALITY_THRESHOLDS["other"])
