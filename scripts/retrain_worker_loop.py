"""
The docker-compose "worker" service's entrypoint. Per-request verification
(Phase 3) runs in-process inside the API service via a thread pool - it's
already async and doesn't block responses, and a single Python process
handling both request routing and its own background thread pool is a
perfectly normal architecture at this scale, so it doesn't need a second
container. See README's Phase 5 section for the fuller version of this
trade-off.

What *does* benefit from its own container is the genuinely independent,
periodic job: retraining the classifier from accumulated feedback (Phase 3
step 4). This script is that worker - it sleeps, wakes up on an interval,
retrains, and goes back to sleep, sharing the data/ volume with the API
service so both read/write the same SQLite DB, classifier file, and
feedback CSV.

RETRAIN_INTERVAL_SECONDS defaults to a week; override it (e.g. to 60 for a
demo) via the environment.
"""

import os
import time
from datetime import datetime, timezone

from src.verification.retrain_weekly import main as retrain_main

DEFAULT_INTERVAL = 7 * 24 * 60 * 60  # one week, in seconds


def run_forever() -> None:
    interval = int(os.getenv("RETRAIN_INTERVAL_SECONDS", DEFAULT_INTERVAL))
    print(f"retrain worker started - interval {interval}s ({interval / 3600:.1f}h)")

    while True:
        now = datetime.now(timezone.utc).isoformat()
        print(f"[{now}] running weekly retrain...")
        try:
            retrain_main()
        except FileNotFoundError as exc:
            # base dataset not generated yet - nothing to train on
            print(f"[{now}] skipped: {exc}")
        except Exception as exc:  # noqa: BLE001 - a bad retrain shouldn't kill the loop
            print(f"[{now}] retrain failed: {exc}")

        print(f"sleeping {interval}s until next retrain...")
        time.sleep(interval)


if __name__ == "__main__":
    run_forever()
