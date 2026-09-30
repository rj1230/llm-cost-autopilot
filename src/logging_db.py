"""
SQLite audit logging for LLM Cost Autopilot.

Each request gets an audit row containing routing, token, cost, latency,
verification, escalation, provider failure, and resilience metadata.

Only a hash of the original prompt is stored.
"""

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.models.response import Response


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "requests.db"

_lock = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    request_id          TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    prompt_hash         TEXT NOT NULL,
    tier                INTEGER NOT NULL,
    primary_model       TEXT NOT NULL,
    routed_model        TEXT NOT NULL,
    used_fallback       INTEGER NOT NULL,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    latency_s           REAL NOT NULL,
    quality_score       REAL,
    escalated           INTEGER,
    verified            INTEGER NOT NULL DEFAULT 0,
    error_type          TEXT,
    primary_error_type  TEXT,
    circuit_state       TEXT
);
"""


def _prepare_connection(conn: sqlite3.Connection) -> None:
    """Create the schema and apply lightweight migrations."""

    conn.execute(SCHEMA)

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(request_log)").fetchall()
    }

    if "primary_model" not in columns:
        conn.execute("ALTER TABLE request_log ADD COLUMN primary_model TEXT")

        conn.execute(
            "UPDATE request_log "
            "SET primary_model = routed_model "
            "WHERE primary_model IS NULL"
        )

    if "error_type" not in columns:
        conn.execute("ALTER TABLE request_log ADD COLUMN error_type TEXT")

    if "primary_error_type" not in columns:
        conn.execute("ALTER TABLE request_log ADD COLUMN primary_error_type TEXT")

    if "circuit_state" not in columns:
        conn.execute("ALTER TABLE request_log ADD COLUMN circuit_state TEXT")

    conn.commit()


def ensure_schema(db_path: Path | None = None) -> None:
    """
    Ensure the audit database exists and has the current schema.

    Safe to call repeatedly. Existing databases created by older
    versions are migrated automatically.
    """

    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        conn = sqlite3.connect(
            path,
            timeout=10,
        )

        try:
            _prepare_connection(conn)
        finally:
            conn.close()


def _connect() -> sqlite3.Connection:
    """Open the audit database with the current schema applied."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    try:
        _prepare_connection(conn)
        return conn
    except Exception:
        conn.close()
        raise


def hash_prompt(prompt: str) -> str:
    """Return a short deterministic hash of the original prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def log_request(
    request_id: str,
    prompt: str,
    tier: int,
    routed_model: str,
    used_fallback: bool,
    response: Response,
    primary_model: str | None = None,
    primary_error_type: str | None = None,
    circuit_state: str | None = None,
) -> None:
    """
    Persist one completed routing request.

    primary_model:
        Model selected by the routing policy before fallback.

    routed_model:
        Model that ultimately generated the response.

    used_fallback:
        Whether the primary model failed and the fallback handled
        the request.

    primary_error_type:
        Explicit failure category of the primary model when it failed.

    circuit_state:
        Circuit state associated with the primary provider when the
        request was recorded.

    error_type:
        Explicit failure category of the final routed response.
        None means the final response succeeded.
    """

    if primary_model is None:
        primary_model = routed_model

    conn = None

    with _lock:
        try:
            conn = _connect()

            conn.execute(
                """
                INSERT OR REPLACE INTO request_log
                (
                    request_id,
                    timestamp,
                    prompt_hash,
                    tier,
                    primary_model,
                    routed_model,
                    used_fallback,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_s,
                    verified,
                    error_type,
                    primary_error_type,
                    circuit_state
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    hash_prompt(prompt),
                    tier,
                    primary_model,
                    routed_model,
                    int(used_fallback),
                    response.input_tokens,
                    response.output_tokens,
                    response.cost_usd,
                    response.latency_s,
                    response.error_type,
                    primary_error_type,
                    circuit_state,
                ),
            )

            conn.commit()

        finally:
            if conn is not None:
                conn.close()


def update_verification(
    request_id: str,
    quality_score: float,
    escalated: bool,
) -> None:
    """Update verification metadata for an existing audit row."""

    conn = None

    with _lock:
        try:
            conn = _connect()

            conn.execute(
                """
                UPDATE request_log
                SET
                    quality_score = ?,
                    escalated = ?,
                    verified = 1
                WHERE request_id = ?
                """,
                (
                    quality_score,
                    int(escalated),
                    request_id,
                ),
            )

            conn.commit()

        finally:
            if conn is not None:
                conn.close()
