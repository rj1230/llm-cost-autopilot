"""
Phase 3, step 2: "After the response is returned to the user, queue an async
job..."

A background thread pool is enough to demonstrate the pattern without adding
infrastructure dependencies (Celery/RQ + a broker) a solo portfolio project
doesn't need yet - swap this for a real task queue if this ever needs to run
across multiple processes/machines. Jobs are fire-and-forget: submit()
returns immediately, and on_complete (if given) runs in the worker thread
once the verification call finishes.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from src.logging_db import update_verification
from src.models.response import Response
from src.verification.feedback import append_feedback
from src.verification.logging_store import log_escalation, log_verification
from src.verification.verifier import VerificationOutcome, verify_response

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="verifier")


def handle_outcome(outcome: VerificationOutcome) -> None:
    """Logs the outcome, fills in the audit-trail row (Phase 4), and on
    escalation appends the classifier feedback row. Public because
    routing.py's synchronous verification path reuses it directly, rather
    than always going through the background executor."""
    log_verification(outcome.to_log_dict())
    update_verification(outcome.request_id, outcome.score, outcome.escalated)
    if outcome.escalated:
        log_escalation(outcome.to_log_dict())
        # tier 3 is hardcoded here because verify_response always verifies
        # against config/routing.yaml's tier-3 model - see _reference_model_name()
        append_feedback(outcome.prompt, corrected_tier=3)


def submit_verification(
    prompt: str,
    cheap_response: Response,
    original_model: str,
    request_id: str,
    on_complete: Callable[[VerificationOutcome], None] | None = None,
) -> Future:
    """Non-blocking: schedules verify_response() on a background thread and
    returns immediately. Logging + the audit-row update + the feedback-loop
    write always happen; on_complete is an extra hook (e.g. for a caller that
    wants the outcome for its own bookkeeping, like the demo script)."""

    def _job() -> VerificationOutcome:
        outcome = verify_response(prompt, cheap_response, original_model, request_id)
        handle_outcome(outcome)
        if on_complete:
            on_complete(outcome)
        return outcome

    return _executor.submit(_job)


def shutdown(wait: bool = True) -> None:
    """Call at process exit so no in-flight verification jobs get dropped."""
    _executor.shutdown(wait=wait)
