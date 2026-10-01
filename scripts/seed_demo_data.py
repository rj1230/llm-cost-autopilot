"""
Populates data/requests.db with SYNTHETIC historical request data spanning
the last 14 days, so `streamlit run dashboard/app.py` has something to show
right away instead of an empty dashboard. This is clearly fake data for
demo/screenshot purposes - it does not call any provider and costs nothing.

For your real portfolio numbers, delete data/requests.db and let it fill up
from actual `route_request_with_verification()` calls instead (or run
scripts/run_single_request.py a bunch of times against your real keys).

Run with:  python -m scripts.seed_demo_data [--requests 400] [--days 14]
"""

import argparse
import random
import uuid
from datetime import UTC, datetime, timedelta

from src.logging_db import _connect, hash_prompt
from src.models.registry import get_model

random.seed(7)

# roughly mirrors what a real prompt mix looks like: mostly simple/moderate
# traffic, a smaller slice of genuinely hard requests
TIER_WEIGHTS = {1: 0.55, 2: 0.30, 3: 0.15}
TIER_TO_MODEL = {1: "local-llama", 2: "gpt-4o-mini", 3: "gpt-4o"}
ESCALATION_RATE = 0.08  # fraction of verified (tier 1/2) requests that fail and escalate


def _random_tokens(tier: int) -> tuple[int, int]:
    # complex requests tend to have longer prompts and longer answers
    if tier == 1:
        return random.randint(15, 80), random.randint(5, 60)
    if tier == 2:
        return random.randint(40, 200), random.randint(40, 250)
    return random.randint(80, 400), random.randint(100, 500)


def seed(num_requests: int, num_days: int) -> None:
    now = datetime.now(UTC)
    conn = _connect()
    try:
        for i in range(num_requests):
            tier = random.choices(list(TIER_WEIGHTS), weights=list(TIER_WEIGHTS.values()))[0]
            model_name = TIER_TO_MODEL[tier]
            model_config = get_model(model_name)
            input_tokens, output_tokens = _random_tokens(tier)
            cost = model_config.cost_for(input_tokens, output_tokens)
            latency = model_config.avg_latency_s * random.uniform(0.7, 1.4)

            # spread requests evenly-ish across the last num_days, more on
            # recent days to look like a growing, realistic usage curve
            day_offset = random.triangular(0, num_days, num_days * 0.2)
            timestamp = now - timedelta(days=day_offset, seconds=random.randint(0, 86400))

            request_id = str(uuid.uuid4())
            verified = 0
            escalated = None
            quality_score = None

            if tier != 3:
                verified = 1
                is_escalation = random.random() < ESCALATION_RATE
                escalated = int(is_escalation)
                quality_score = round(random.uniform(0.3, 0.6) if is_escalation else random.uniform(0.82, 1.0), 3)
                if is_escalation:
                    # escalated requests actually got served by gpt-4o after the fact
                    ref = get_model("gpt-4o")
                    extra_cost = ref.cost_for(input_tokens, output_tokens)
                    cost += extra_cost

            conn.execute(
                """INSERT OR REPLACE INTO request_log
                   (request_id, timestamp, prompt_hash, tier, routed_model, used_fallback,
                    input_tokens, output_tokens, cost_usd, latency_s, quality_score,
                    escalated, verified)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id, timestamp.isoformat(), hash_prompt(f"demo-prompt-{i}"),
                    tier, model_name, input_tokens, output_tokens, cost, latency,
                    quality_score, escalated, verified,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    print(f"seeded {num_requests} synthetic requests across the last {num_days} days into data/requests.db")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    seed(args.requests, args.days)


if __name__ == "__main__":
    main()
