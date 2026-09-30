"""
Deterministic feature extraction for request-complexity classification.

V2 adds task-intent and structural-complexity signals while preserving the
original lexical features. The classifier remains responsible for deciding
the final tier; this module only converts a prompt into numeric features.
"""

import re


# ---------------------------------------------------------------------------
# Existing lexical signals
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
# V2 task-intent signals
# ---------------------------------------------------------------------------

EXPLANATION_MARKERS = [
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
]

COMPARISON_MARKERS = [
    "compare",
    "comparison",
    "contrast",
    "difference between",
    "versus",
    "vs",
]

ANALYSIS_MARKERS = [
    "analyze",
    "analyse",
    "analysis",
    "implications",
    "trade-off",
    "tradeoff",
    "weigh",
    "assess",
]

DESIGN_MARKERS = [
    "design",
    "architecture",
    "architect",
    "build",
    "develop a strategy",
    "develop a framework",
    "create a strategy",
    "create an architecture",
]

RECOMMENDATION_MARKERS = [
    "recommend",
    "recommendation",
    "which should",
    "should i",
    "advise",
    "advising",
    "choose",
    "select",
]

EVALUATION_MARKERS = [
    "evaluate",
    "evaluation",
    "benchmark",
    "measure",
    "assess",
    "compare performance",
]

GENERATION_MARKERS = [
    "write",
    "create",
    "generate",
    "compose",
    "draft",
    "produce",
]

REWRITE_MARKERS = [
    "rewrite",
    "rephrase",
    "paraphrase",
    "make this",
    "change the tone",
    "formal tone",
]

EXTRACTION_MARKERS = [
    "extract",
    "find the",
    "identify the",
    "return the",
    "get the",
]

CATEGORIZATION_MARKERS = [
    "categorize",
    "categorise",
    "classify",
    "group these",
    "group the",
]

CALCULATION_MARKERS = [
    "calculate",
    "compute",
    "solve",
    "how much",
    "how many",
]


# ---------------------------------------------------------------------------
# V2 reasoning / structure signals
# ---------------------------------------------------------------------------

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

MULTI_REQUIREMENT_MARKERS = [
    "including",
    "along with",
    "as well as",
    "while",
    "and",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_hits(text_lower: str, markers: list[str]) -> int:
    """Count distinct marker occurrences using word/phrase-aware matching."""
    count = 0

    for marker in markers:
        escaped = re.escape(marker.lower())

        # Match standalone words/phrases instead of arbitrary substrings.
        pattern = rf"(?<!\w){escaped}(?!\w)"

        if re.search(pattern, text_lower):
            count += 1

    return count


def _count_phrase_occurrences(text_lower: str, markers: list[str]) -> int:
    """Count all matching marker occurrences across a marker group."""
    count = 0

    for marker in markers:
        escaped = re.escape(marker.lower())
        pattern = rf"(?<!\w){escaped}(?!\w)"
        count += len(re.findall(pattern, text_lower))

    return count


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

    # Task intent
    explanation_count = _count_phrase_occurrences(
        text_lower,
        EXPLANATION_MARKERS,
    )

    comparison_count = _count_phrase_occurrences(
        text_lower,
        COMPARISON_MARKERS,
    )

    analysis_count = _count_phrase_occurrences(
        text_lower,
        ANALYSIS_MARKERS,
    )

    design_count = _count_phrase_occurrences(
        text_lower,
        DESIGN_MARKERS,
    )

    recommendation_count = _count_phrase_occurrences(
        text_lower,
        RECOMMENDATION_MARKERS,
    )

    evaluation_count = _count_phrase_occurrences(
        text_lower,
        EVALUATION_MARKERS,
    )

    generation_count = _count_phrase_occurrences(
        text_lower,
        GENERATION_MARKERS,
    )

    rewrite_count = _count_phrase_occurrences(
        text_lower,
        REWRITE_MARKERS,
    )

    extraction_count = _count_phrase_occurrences(
        text_lower,
        EXTRACTION_MARKERS,
    )

    categorization_count = _count_phrase_occurrences(
        text_lower,
        CATEGORIZATION_MARKERS,
    )

    calculation_count = _count_phrase_occurrences(
        text_lower,
        CALCULATION_MARKERS,
    )

    # Reasoning structure
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

    multiple_requirement_count = _count_phrase_occurrences(
        text_lower,
        MULTI_REQUIREMENT_MARKERS,
    )

    # Structural signals
    conjunction_count = len(
        re.findall(r"\b(?:and|or|but|while|whereas|then)\b", text_lower)
    )

    action_count = sum(
        [
            design_count,
            recommendation_count,
            evaluation_count,
            generation_count,
            rewrite_count,
            extraction_count,
            categorization_count,
            calculation_count,
            analysis_count,
        ]
    )

    requirement_count = (
        num_constraints
        + multiple_requirement_count
        + justification_count
        + tradeoff_count
        + step_count
    )

    multi_part_request = int(
        (conjunction_count >= 2 and word_count >= 12)
        or requirement_count >= 3
        or action_count >= 2
    )

    return {
        # Existing V1 features
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
        "num_sentences": max(1, len(re.findall(r"[.!?]+", prompt))),
        "has_numbers": int(bool(re.search(r"\d", prompt))),
        # V2 task intent
        "has_explanation": int(explanation_count > 0),
        "has_comparison": int(comparison_count > 0),
        "has_analysis": int(analysis_count > 0),
        "has_design": int(design_count > 0),
        "has_recommendation": int(recommendation_count > 0),
        "has_evaluation": int(evaluation_count > 0),
        "has_generation": int(generation_count > 0),
        "has_rewrite": int(rewrite_count > 0),
        "has_extraction": int(extraction_count > 0),
        "has_categorization": int(categorization_count > 0),
        "has_calculation": int(calculation_count > 0),
        # V2 reasoning structure
        "has_justification": int(justification_count > 0),
        "has_tradeoff": int(tradeoff_count > 0),
        "has_step_by_step": int(step_count > 0),
        "has_nuanced_reasoning": int(nuanced_count > 0),
        "has_multiple_requirements": int(multiple_requirement_count > 0),
        # V2 structural complexity
        "requirement_count": requirement_count,
        "action_count": action_count,
        "conjunction_count": conjunction_count,
        "multi_part_request": multi_part_request,
    }


FEATURE_NAMES = list(
    extract_features("placeholder text for feature name ordering.").keys()
)


def features_to_vector(prompt: str) -> list[float]:
    feats = extract_features(prompt)
    return [feats[name] for name in FEATURE_NAMES]
