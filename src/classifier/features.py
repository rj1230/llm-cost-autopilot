"""
Deterministic feature extraction for request-complexity classification.

V3 focuses on generalizable structural complexity rather than a large
collection of binary task-specific keyword features.

The classifier remains responsible for deciding the final tier.
This module only converts a prompt into numeric features.
"""

import re

# ---------------------------------------------------------------------------
# Core lexical / reasoning signals
# ---------------------------------------------------------------------------

REASONING_KEYWORDS = [
    "analyze",
    "analyse",
    "compare",
    "contrast",
    "design",
    "recommend",
    "evaluate",
    "justify",
    "explain the difference",
    "trade-off",
    "tradeoff",
    "weigh",
    "debug",
    "solve",
    "prove",
    "ethical",
    "nuanced",
    "architecture",
]


CONSTRAINT_MARKERS = [
    "must",
    "should",
    "at least",
    "no more than",
    "do not",
    "don't",
    "only",
    "exactly",
    "between",
    "considering",
    "in about",
    "words",
    "sentences",
    "lines",
    "step by step",
]


FORMAT_MARKERS = [
    "json",
    "table",
    "markdown",
    "csv",
    "yaml",
    "schema",
    "format",
    "bullet",
    "list",
    "object",
]


CONTEXT_MARKERS = [
    "following",
    "this text",
    "this paragraph",
]


# ---------------------------------------------------------------------------
# Task-intent markers
# ---------------------------------------------------------------------------

TASK_INTENT_GROUPS = {
    "explanation": [
        "explain",
        "describe",
        "how does",
        "how do",
        "how to",
        "what is",
        "what are",
        "why does",
        "why is",
        "why are",
    ],
    "comparison": [
        "compare",
        "comparison",
        "contrast",
        "difference between",
        "versus",
        "vs",
    ],
    "analysis": [
        "analyze",
        "analyse",
        "analysis",
        "implications",
        "trade-off",
        "tradeoff",
        "weigh",
        "assess",
    ],
    "design": [
        "design",
        "architecture",
        "architect",
        "build",
        "develop a strategy",
        "develop a framework",
        "create a strategy",
        "create an architecture",
    ],
    "recommendation": [
        "recommend",
        "recommendation",
        "which should",
        "should i",
        "advise",
        "advising",
        "choose",
        "select",
    ],
    "evaluation": [
        "evaluate",
        "evaluation",
        "benchmark",
        "measure",
        "assess",
        "compare performance",
    ],
    "generation": [
        "write",
        "create",
        "generate",
        "compose",
        "draft",
        "produce",
    ],
    "rewrite": [
        "rewrite",
        "rephrase",
        "paraphrase",
        "make this",
        "change the tone",
        "formal tone",
    ],
    "extraction": [
        "extract",
        "find the",
        "identify the",
        "return the",
        "get the",
    ],
    "categorization": [
        "categorize",
        "categorise",
        "classify",
        "group these",
        "group the",
    ],
    "calculation": [
        "calculate",
        "compute",
        "solve",
        "how much",
        "how many",
    ],
}


JUSTIFICATION_MARKERS = [
    "justify",
    "justification",
    "with reasoning",
    "because",
    "explain why",
    "support your answer",
]


TRADEOFF_MARKERS = [
    "trade-off",
    "tradeoff",
    "pros and cons",
    "advantages and disadvantages",
    "weigh the",
]


STEP_MARKERS = [
    "step by step",
    "step-by-step",
    "steps",
    "procedure",
    "process",
]


NUANCED_MARKERS = [
    "nuanced",
    "implications",
    "consider both",
    "limitations",
    "edge cases",
    "in depth",
    "deep dive",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_hits(text_lower: str, markers: list[str]) -> int:
    """Count distinct marker types that occur in the prompt."""
    count = 0

    for marker in markers:
        escaped = re.escape(marker.lower())
        pattern = rf"(?<!\w){escaped}(?!\w)"

        if re.search(pattern, text_lower):
            count += 1

    return count


def _count_phrase_occurrences(text_lower: str, markers: list[str]) -> int:
    """Count all occurrences of markers in a marker group."""
    count = 0

    for marker in markers:
        escaped = re.escape(marker.lower())
        pattern = rf"(?<!\w){escaped}(?!\w)"
        count += len(re.findall(pattern, text_lower))

    return count


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(prompt: str) -> dict[str, float]:
    text_lower = prompt.lower()
    words = prompt.split()
    word_count = len(words)

    reasoning_keyword_count = _count_hits(
        text_lower,
        REASONING_KEYWORDS,
    )

    num_constraints = _count_hits(
        text_lower,
        CONSTRAINT_MARKERS,
    )

    output_format_complexity = _count_hits(
        text_lower,
        FORMAT_MARKERS,
    )

    # -----------------------------------------------------------------------
    # Task-intent counts
    # -----------------------------------------------------------------------

    intent_counts = {
        name: _count_phrase_occurrences(text_lower, markers)
        for name, markers in TASK_INTENT_GROUPS.items()
    }

    task_intent_count = sum(
        int(count > 0)
        for count in intent_counts.values()
    )

    action_count = sum(
        intent_counts[name]
        for name in (
            "design",
            "recommendation",
            "evaluation",
            "generation",
            "rewrite",
            "extraction",
            "categorization",
            "calculation",
            "analysis",
        )
    )

    # -----------------------------------------------------------------------
    # Reasoning structure
    # -----------------------------------------------------------------------

    justification_count = _count_phrase_occurrences(
        text_lower,
        JUSTIFICATION_MARKERS,
    )

    tradeoff_count = _count_phrase_occurrences(
        text_lower,
        TRADEOFF_MARKERS,
    )

    step_count = _count_phrase_occurrences(
        text_lower,
        STEP_MARKERS,
    )

    nuanced_count = _count_phrase_occurrences(
        text_lower,
        NUANCED_MARKERS,
    )

    # -----------------------------------------------------------------------
    # Structural signals
    # -----------------------------------------------------------------------

    comma_count = text_lower.count(",")

    conjunction_count = len(
        re.findall(
            r"\b(?:and|or|but|while|whereas|then)\b",
            text_lower,
        )
    )

    # Clauses are approximated using sentence boundaries plus coordinating
    # conjunctions. This is deliberately deterministic and dependency-free.
    sentence_count = max(
        1,
        len(re.findall(r"[.!?]+", prompt)),
    )

    clause_count = max(
        1,
        sentence_count + conjunction_count,
    )

    # A requirement should represent an actual requested operation or
    # explicit constraint, not merely the presence of "and".
    requirement_count = (
        num_constraints
        + justification_count
        + tradeoff_count
        + step_count
        + max(0, task_intent_count - 1)
    )

    normalized_length = word_count / max(sentence_count, 1)

    requirement_density = (
        requirement_count / max(word_count, 1)
    )

    # Reasoning depth captures multiple reasoning dimensions without making
    # any single keyword decisive.
    reasoning_depth = (
        reasoning_keyword_count
        + justification_count
        + tradeoff_count
        + step_count
        + nuanced_count
    )

    # Structural complexity combines independent structural signals.
    # It is a continuous feature; the classifier still decides the tier.
    structural_complexity = (
        0.30 * min(word_count / 10.0, 5.0)
        + 0.20 * min(comma_count, 5)
        + 0.20 * min(conjunction_count, 5)
        + 0.15 * min(task_intent_count, 5)
        + 0.15 * min(requirement_count, 5)
    )

    return {
        # V1 foundation
        "token_count": word_count,
        "char_count": len(prompt),
        "has_reasoning_keyword": int(reasoning_keyword_count > 0),
        "reasoning_keyword_count": reasoning_keyword_count,
        "num_constraints": num_constraints,
        "has_context": int(
            _count_hits(text_lower, CONTEXT_MARKERS) > 0
            or bool(re.search(r"""['"]""", prompt))
        ),
        "output_format_complexity": output_format_complexity,
        "num_sentences": sentence_count,
        "has_numbers": int(bool(re.search(r"\d", prompt))),

        # V3 structural/task features
        "comma_count": comma_count,
        "conjunction_count": conjunction_count,
        "clause_count": clause_count,
        "action_count": action_count,
        "task_intent_count": task_intent_count,
        "requirement_count": requirement_count,
        "normalized_length": normalized_length,
        "requirement_density": requirement_density,
        "structural_complexity": structural_complexity,
        "reasoning_depth": reasoning_depth,
    }


FEATURE_NAMES = list(
    extract_features(
        "placeholder text for feature name ordering."
    ).keys()
)


def features_to_vector(prompt: str) -> list[float]:
    feats = extract_features(prompt)
    return [feats[name] for name in FEATURE_NAMES]
