"""
Phase 6, step 1: "Run a realistic load test: Send 500-1,000 diverse prompts
through the system. Generate the final cost savings report. Screenshot the
dashboard. These artifacts are what you put in your portfolio."

Requires real API keys in .env - this calls actual providers, so it costs
real (small) money. Generates a *fresh* set of prompts (different random
seed than data/labeled_dataset.csv) by reusing the Phase 2 template
generators, so this is closer to unseen production traffic than to the
training set.

Usage:
    python -m scripts.load_test                       # 750 prompts, async verification
    python -m scripts.load_test --count 500 --sync     # wait for each verification inline
    python -m scripts.load_test --fresh                # back up and start data/requests.db empty first

After this finishes, run:
    python -m scripts.generate_report
to turn the results into data/cost_savings_report.md + chart PNGs.
"""

import argparse
import random
import shutil
import time
from pathlib import Path

from data.generate_dataset import gen_tier1, gen_tier2, gen_tier3
from src.routing import route_request_with_verification
from src.verification.queue import shutdown

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "requests.db"

TIER_SPLIT = {1: 0.55, 2: 0.30, 3: 0.15}  # mirrors scripts/seed_demo_data.py's realistic mix


def build_prompt_pool(count: int, seed: int) -> list[str]:
    random.seed(seed)
    n1, n2 = int(count * TIER_SPLIT[1]), int(count * TIER_SPLIT[2])
    n3 = count - n1 - n2
    # over-generate a bit since the tier generators dedupe internally
    prompts = gen_tier1(int(n1 * 1.5)) + gen_tier2(int(n2 * 1.5)) + gen_tier3(int(n3 * 1.5))
    seen, unique = set(), []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    random.shuffle(unique)
    return unique[:count]


def backup_and_clear_db() -> None:
    if DB_PATH.exists():
        backup_path = DB_PATH.with_suffix(f".db.bak-{int(time.time())}")
        shutil.move(str(DB_PATH), str(backup_path))
        print(f"backed up existing DB to {backup_path.name}")


def run_load_test(count: int, synchronous: bool, seed: int) -> None:
    prompts = build_prompt_pool(count, seed)
    print(f"sending {len(prompts)} prompts through route_request_with_verification "
          f"(synchronous={synchronous})...")

    errors = 0
    started = time.perf_counter()
    for i, prompt in enumerate(prompts, start=1):
        try:
            result, _ = route_request_with_verification(prompt, synchronous=synchronous)
            if result.response.error:
                errors += 1
        except Exception as exc:  # noqa: BLE001 - one bad prompt shouldn't kill a 750-request run
            errors += 1
            print(f"  [{i}] request failed: {exc}")

        if i % 50 == 0 or i == len(prompts):
            elapsed = time.perf_counter() - started
            print(f"  {i}/{len(prompts)} sent ({elapsed:.0f}s elapsed, {errors} errors so far)")

    print("draining background verification queue...")
    shutdown(wait=True)
    print(f"done. {len(prompts)} requests, {errors} errors. Run `python -m scripts.generate_report` next.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=750)
    parser.add_argument("--sync", action="store_true", help="wait for verification on every request")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--fresh", action="store_true", help="back up and start data/requests.db empty")
    args = parser.parse_args()

    if args.fresh:
        backup_and_clear_db()

    run_load_test(args.count, args.sync, args.seed)


if __name__ == "__main__":
    main()
