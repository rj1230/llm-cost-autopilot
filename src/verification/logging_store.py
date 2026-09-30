"""
Lightweight append-only logging for Phase 3. Phase 4 owns the real SQLite
audit trail + dashboard and will read these JSONL files in as its seed data -
Phase 3 only needs *something* durable to log escalation events to per step
3 ("Log the escalation event with: original model, escalated model, cost
delta, and the quality gap"), so a plain JSON-lines file keeps this phase
self-contained without building Phase 4's database early.
"""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
VERIFICATION_LOG_PATH = DATA_DIR / "verification_log.jsonl"
ESCALATION_LOG_PATH = DATA_DIR / "escalation_log.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_verification(record: dict[str, Any]) -> None:
    """Every verification job's outcome - pass or fail."""
    _append_jsonl(VERIFICATION_LOG_PATH, record)


def log_escalation(record: dict[str, Any]) -> None:
    """Only the ones that failed and got escalated - the routing-failure log
    step 3 and step 4 both build on."""
    _append_jsonl(ESCALATION_LOG_PATH, record)
