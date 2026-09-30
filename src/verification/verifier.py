"""
Phase 3, step 2: "queue an async job that sends the same prompt to the
highest-tier model and compares outputs. Score the agreement. If the cheap
model's output diverges significantly, log it as a routing failure."

verify_response() does the synchronous work for one request; queue.py is
what makes calling it asynchronous. Kept separate so the scoring logic can be
unit-tested without touching threads or the network (see tests/test_verification.py).
"""

import time
import yaml
from dataclasses import dataclass
from pathlib import Path

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
    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)
    return config["routing"][3]  # tier 3 is always the highest-quality model


@dataclass
class VerificationOutcome:
    request_id: str
    prompt: str
    task_type: str
    score: float
    threshold: float
    passed: bool
    original_model: str
    reference_model: str
    reference_response: Response
    escalated: bool
    final_response: Response  # original response if passed, reference response if escalated
    cost_delta_usd: float
    quality_gap: float
    latency_s: float

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "task_type": self.task_type,
            "score": round(self.score, 3),
            "threshold": self.threshold,
            "passed": self.passed,
            "original_model": self.original_model,
            "reference_model": self.reference_model,
            "escalated": self.escalated,
            "cost_delta_usd": round(self.cost_delta_usd, 8),
            "quality_gap": round(self.quality_gap, 3),
            "latency_s": round(self.latency_s, 4),
        }


def _score(prompt: str, cheap_output: str, reference_output: str, task_type: str) -> float:
    if task_type == "extraction":
        return jaccard_similarity(cheap_output, reference_output)
    if task_type == "classification":
        return label_match(cheap_output, reference_output)
    # summarization / other -> LLM-as-judge, scored by the reference model itself
    judge_prompt = build_judge_prompt(prompt, cheap_output, reference_output)
    reference_config = get_model(_reference_model_name())
    judge_response = send_request(judge_prompt, reference_config)
    if judge_response.error:
        return 0.6  # can't judge - treat as a soft, not-confident pass (see parse_judge_score)
    return parse_judge_score(judge_response.output_text)


def verify_response(
    prompt: str, cheap_response: Response, original_model: str, request_id: str
) -> VerificationOutcome:
    start = time.perf_counter()
    task_type = infer_task_type(prompt)
    reference_name = _reference_model_name()
    reference_config = get_model(reference_name)
    reference_response = send_request(prompt, reference_config)

    score = _score(prompt, cheap_response.output_text, reference_response.output_text, task_type)
    threshold = threshold_for(task_type)
    passed = passes_threshold(task_type, score)

    escalated = not passed
    final_response = reference_response if escalated else cheap_response
    cost_delta = reference_response.cost_usd - cheap_response.cost_usd if escalated else 0.0
    quality_gap = max(0.0, threshold - score) if not passed else 0.0

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
    )
