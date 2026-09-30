"""
The actual comparison methods behind each task type's quality check.

extraction/classification are compared directly against the reference
model's own answer with cheap, deterministic heuristics - no need to spend a
second API call asking a model to judge something a set-overlap or string
match already answers. summarization/other get a real LLM-as-judge call,
since "did this summary capture the same substance" isn't something a word
overlap score can safely answer (two good summaries can share almost no
words).
"""

import re
import string

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_words(text: str) -> set[str]:
    text = text.lower().translate(_PUNCT_TABLE)
    return {w for w in text.split() if w}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-overlap score in [0, 1]. Used for extraction: two field values
    that share most of their words score high even if formatting differs."""
    words_a, words_b = _normalize_words(text_a), _normalize_words(text_b)
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def label_match(text_a: str, text_b: str) -> float:
    """1.0 if the two responses' first line/word agree, else 0.0. Used for
    classification, where there's no partial credit for the wrong label."""
    a = text_a.strip().splitlines()[0].strip().lower().rstrip(string.punctuation) if text_a.strip() else ""
    b = text_b.strip().splitlines()[0].strip().lower().rstrip(string.punctuation) if text_b.strip() else ""
    return 1.0 if a and a == b else 0.0


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether two AI responses to the same request agree in substance, not wording.

Original request:
{prompt}

Response A:
{response_a}

Response B (yours):
{response_b}

Rate how well Response A agrees with Response B in substance, on a scale of 1-5,
where 5 means they convey the same information/conclusion and 1 means they
meaningfully disagree or A is missing/wrong. Reply with ONLY the digit."""


def build_judge_prompt(prompt: str, cheap_output: str, reference_output: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, response_a=cheap_output, response_b=reference_output
    )


def parse_judge_score(judge_output: str) -> float:
    """Extracts the 1-5 digit and normalizes to 0-1. Defaults to the middle
    of the range (0.6) if the judge model didn't reply with a clean digit,
    so a parsing hiccup doesn't silently look like a confident pass or fail."""
    match = re.search(r"[1-5]", judge_output)
    if not match:
        return 0.6
    return int(match.group()) / 5.0
