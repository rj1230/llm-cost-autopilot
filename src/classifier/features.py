"""
Turns a raw prompt string into the numeric features the classifier trains on.

Per Phase 2 step 2: "features you'll extract: token count, presence of
instructions like 'analyze' or 'compare', number of constraints, whether
context is provided, and output format complexity."

Kept dependency-free (no tokenizer) so this runs instantly on both training
data and a live request with no extra setup - word count is a fine proxy for
token count at this stage.
"""

import re

REASONING_KEYWORDS = [
    "analyze", "analyse", "compare", "contrast", "design", "recommend",
    "evaluate", "justify", "explain the difference", "trade-off", "weigh",
    "debug", "solve", "prove", "ethical", "nuanced", "architecture",
]
CONSTRAINT_MARKERS = [
    "must", "should", "at least", "no more than", "do not", "don't",
    "only", "exactly", "between", "considering", "in about", "words",
    "sentences", "lines", "step by step",
]
FORMAT_MARKERS = [
    "json", "table", "markdown", "csv", "yaml", "schema", "format",
    "bullet", "list", "object",
]
CONTEXT_MARKERS = ["'", '"', ":", "following", "this text", "this paragraph"]


def _count_hits(text_lower: str, markers: list[str]) -> int:
    return sum(1 for m in markers if m in text_lower)


def extract_features(prompt: str) -> dict[str, float]:
    text_lower = prompt.lower()
    word_count = len(prompt.split())

    return {
        "token_count": word_count,  # word count as a token-count proxy
        "char_count": len(prompt),
        "has_reasoning_keyword": int(_count_hits(text_lower, REASONING_KEYWORDS) > 0),
        "reasoning_keyword_count": _count_hits(text_lower, REASONING_KEYWORDS),
        "num_constraints": _count_hits(text_lower, CONSTRAINT_MARKERS),
        "has_context": int(_count_hits(text_lower, CONTEXT_MARKERS) > 0),
        "output_format_complexity": _count_hits(text_lower, FORMAT_MARKERS),
        "num_sentences": max(1, len(re.findall(r"[.!?]+", prompt))),
        "has_numbers": int(bool(re.search(r"\d", prompt))),
    }


FEATURE_NAMES = list(
    extract_features("placeholder text for feature name ordering.").keys()
)


def features_to_vector(prompt: str) -> list[float]:
    feats = extract_features(prompt)
    return [feats[name] for name in FEATURE_NAMES]
