"""
Phase 3, step 2: verification of routed LLM responses.

verify_response() performs synchronous quality verification for one
successful routed response. queue.py is responsible for asynchronous
execution.

Verification is deliberately separated into:
- provider/reference failures
- judge failures
- judge parsing failures
- genuine quality pass/fail decisions

Infrastructure failures must never be represented as a synthetic
quality score.
"""

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.client import send_request
from src.models.registry import get_model
from src.models.response import Response
from src.verification.scoring import (
    build_judge_prompt,
    jaccard_similarity,
    label_match,
    parse_judge_score,
)
from src.verification.task_type import infer_task_type
from src.verification.thresholds import passes_threshold, threshold_for

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "routing.yaml"


def _reference_model_name() -> str:
    """Return the highest-quality/reference model configured for tier 3."""

    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)

    return config["routing"][3]


@dataclass
class VerificationOutcome:
    """Result of one verification attempt."""

    request_id: str
    prompt: str
    task_type: str
    score: float | None
    threshold: float
    passed: bool
    original_model: str
    reference_model: str
    reference_response: Response
    escalated: bool
    final_response: Response
    cost_delta_usd: float
    quality_gap: float
    latency_s: float
    status: str = "verified_pass"
    error_type: str | None = None
    judge_cost_usd: float = 0.0

    def to_log_dict(self) -> dict:
        """Return a JSON-serializable verification audit record."""

        return {
            "request_id": self.request_id,
            "prompt_hash": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16],
            "task_type": self.task_type,
            "score": (round(self.score, 3) if self.score is not None else None),
            "threshold": self.threshold,
            "passed": self.passed,
            "original_model": self.original_model,
            "reference_model": self.reference_model,
            "escalated": self.escalated,
            "cost_delta_usd": round(self.cost_delta_usd, 8),
            "judge_cost_usd": round(self.judge_cost_usd, 8),
            "quality_gap": round(self.quality_gap, 3),
            "latency_s": round(self.latency_s, 4),
            "status": self.status,
            "error_type": self.error_type,
        }


def _parse_judge_score_strict(judge_output: str) -> float | None:
    """
    Parse a judge response only when it contains exactly one rating.

    Valid:
        "1"
        "2"
        "3"
        "4"
        "5"

    Invalid:
        "I rate this 4 out of 5"
        "The answer is a 3"
        "4 because the response is good"

    Returning None lets the caller distinguish malformed judge output
    from a genuine quality score.
    """

    if not re.fullmatch(r"\s*[1-5]\s*", judge_output):
        return None

    return parse_judge_score(judge_output)


def _score(
    prompt: str,
    cheap_output: str,
    reference_output: str,
    task_type: str,
) -> tuple[float | None, str, float]:
    """
    Score the cheap response against the reference response.

    Returns:
        score:
            Quality score, or None when verification infrastructure
            failed.

        status:
            verified_pass-compatible scoring status, judge_error, or
            judge_parse_error.

        judge_cost_usd:
            Cost incurred by the optional LLM judge call.
    """

    if task_type == "extraction":
        return (
            jaccard_similarity(cheap_output, reference_output),
            "scored",
            0.0,
        )

    if task_type == "classification":
        return (
            label_match(cheap_output, reference_output),
            "scored",
            0.0,
        )

    # Summarization / other -> LLM-as-judge.
    judge_prompt = build_judge_prompt(
        prompt,
        cheap_output,
        reference_output,
    )

    reference_config = get_model(_reference_model_name())

    try:
        judge_response = send_request(
            judge_prompt,
            reference_config,
        )
    except Exception:  # noqa: BLE001  # noqa: BLE001 - provider failures become structured verification errors
        return None, "judge_error", 0.0

    if judge_response.error:
        return None, "judge_error", judge_response.cost_usd

    score = _parse_judge_score_strict(
        judge_response.output_text,
    )

    if score is None:
        return None, "judge_parse_error", judge_response.cost_usd

    return score, "scored", judge_response.cost_usd


def verify_response(
    prompt: str,
    cheap_response: Response,
    original_model: str,
    request_id: str,
) -> VerificationOutcome:
    """
    Verify one successful cheap-model response.

    Reference/judge infrastructure failures are not treated as quality
    failures. In those cases the original response is retained and the
    outcome carries an explicit verification error status.
    """

    start = time.perf_counter()

    task_type = infer_task_type(prompt)
    reference_name = _reference_model_name()
    reference_config = get_model(reference_name)

    # ---------------------------------------------------------------
    # Reference model call.
    # ---------------------------------------------------------------
    try:
        reference_response = send_request(
            prompt,
            reference_config,
        )
    except Exception:  # noqa: BLE001  # noqa: BLE001 - provider failures become structured verification errors
        return VerificationOutcome(
            request_id=request_id,
            prompt=prompt,
            task_type=task_type,
            score=None,
            threshold=threshold_for(task_type),
            passed=False,
            original_model=original_model,
            reference_model=reference_name,
            reference_response=cheap_response,
            escalated=False,
            final_response=cheap_response,
            cost_delta_usd=0.0,
            quality_gap=0.0,
            latency_s=time.perf_counter() - start,
            status="reference_error",
            error_type="reference_error",
            judge_cost_usd=0.0,
        )

    threshold = threshold_for(task_type)

    # Never compare the cheap response against an error response.
    if reference_response.error:
        return VerificationOutcome(
            request_id=request_id,
            prompt=prompt,
            task_type=task_type,
            score=None,
            threshold=threshold,
            passed=False,
            original_model=original_model,
            reference_model=reference_name,
            reference_response=reference_response,
            escalated=False,
            final_response=cheap_response,
            cost_delta_usd=0.0,
            quality_gap=0.0,
            latency_s=time.perf_counter() - start,
            status="reference_error",
            error_type=reference_response.error_type or "reference_error",
            judge_cost_usd=0.0,
        )

    # ---------------------------------------------------------------
    # Score cheap response against successful reference response.
    # ---------------------------------------------------------------
    score, score_status, judge_cost_usd = _score(
        prompt,
        cheap_response.output_text,
        reference_response.output_text,
        task_type,
    )

    # ---------------------------------------------------------------
    # Judge infrastructure failure.
    # Keep the original response. Do not escalate merely because the
    # verification mechanism itself failed.
    # ---------------------------------------------------------------
    if score is None:
        return VerificationOutcome(
            request_id=request_id,
            prompt=prompt,
            task_type=task_type,
            score=None,
            threshold=threshold,
            passed=False,
            original_model=original_model,
            reference_model=reference_name,
            reference_response=reference_response,
            escalated=False,
            final_response=cheap_response,
            cost_delta_usd=0.0,
            quality_gap=0.0,
            latency_s=time.perf_counter() - start,
            status=score_status,
            error_type=score_status,
            judge_cost_usd=judge_cost_usd,
        )

    # ---------------------------------------------------------------
    # Genuine quality decision.
    # ---------------------------------------------------------------
    passed = passes_threshold(
        task_type,
        score,
    )

    escalated = not passed

    final_response = reference_response if escalated else cheap_response

    cost_delta = (
        reference_response.cost_usd + judge_cost_usd - cheap_response.cost_usd
        if escalated
        else 0.0
    )

    quality_gap = max(0.0, threshold - score) if not passed else 0.0

    status = "verified_pass" if passed else "verified_fail"

    return VerificationOutcome(
        request_id=request_id,
        prompt=prompt,
        task_type=task_type,
        score=score,
        threshold=threshold,
        passed=passed,
        original_model=original_model,
        reference_model=reference_name,
        reference_response=reference_response,
        escalated=escalated,
        final_response=final_response,
        cost_delta_usd=cost_delta,
        quality_gap=quality_gap,
        latency_s=time.perf_counter() - start,
        status=status,
        error_type=None,
        judge_cost_usd=judge_cost_usd,
    )
