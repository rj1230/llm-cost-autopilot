"""
Generate and send realistic production-style traffic through LLM Cost Autopilot.

Usage:
    python -m scripts.load_test
    python -m scripts.load_test --count 500 --sync
    python -m scripts.load_test --fresh
"""

import argparse
import random
import shutil
import time
from pathlib import Path

from src.routing import route_request_with_verification
from src.verification.queue import shutdown

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "requests.db"

TIER_SPLIT = {1: 0.55, 2: 0.30, 3: 0.15}

TIER1_PROMPTS = [
    "What is the capital of France?",
    "Convert 25 Celsius to Fahrenheit.",
    "What is 15 percent of 240?",
    "Define an API in one sentence.",
    "What does HTTP 404 mean?",
    "List three common Python data types.",
    "What is the difference between RAM and storage?",
    "Give a one-line definition of machine learning.",
    "What is JSON used for?",
    "What does SQL stand for?",
    "Name three common Linux commands.",
    "What is a primary key in a database?",
    "What is the purpose of a Python virtual environment?",
    "What is the difference between HTTP and HTTPS?",
    "Give two examples of supervised learning.",
]

TIER2_PROMPTS = [
    "Explain how vector similarity search works.",
    "Explain precision, recall, and F1 score with a simple example.",
    "Compare REST APIs and GraphQL for a typical backend service.",
    "Explain how a random forest classifier makes predictions.",
    "Describe the main stages of a production machine learning pipeline.",
    "Explain how database indexing improves query performance.",
    "Compare synchronous and asynchronous Python execution.",
    "Explain the difference between classification and regression.",
    "Describe how caching can reduce API latency.",
    "Explain the purpose of Docker in a machine learning deployment.",
    "Compare batch inference with online inference.",
    "Explain how a circuit breaker protects a distributed service.",
    "Describe common causes of data leakage in machine learning.",
    "Explain how model evaluation differs from model training.",
    "Describe how tokenization works in modern language models.",
]

TIER3_PROMPTS = [
    "Design a scalable distributed system for millions of users and explain the major architectural tradeoffs.",
    "Design a production-grade RAG system with retrieval, reranking, evaluation, observability, and failure handling.",
    "Develop an architecture for an LLM routing platform that minimizes cost while preserving response quality.",
    "Design a multi-region inference service with failover, rate limiting, circuit breakers, and detailed observability.",
    "Propose an end-to-end MLOps architecture covering training, model registry, deployment gates, monitoring, and automated retraining.",
    "Design a reliable event-driven system for processing millions of messages while preserving ordering where required.",
    "Develop a security and reliability strategy for a public AI API serving multiple enterprise customers.",
    "Design an evaluation framework for comparing multiple LLM providers on quality, latency, cost, and reliability.",
    "Architect a fault-tolerant agentic AI system with tool execution, state management, guardrails, and evaluation.",
    "Design a production data platform that supports real-time features, batch analytics, governance, and disaster recovery.",
]


def _sample_pool(pool: list[str], count: int, rng: random.Random) -> list[str]:
    """Return enough prompts from a tier pool without requiring training artifacts."""
    if count <= 0:
        return []

    return [pool[i % len(pool)] for i in range(count)]


def build_prompt_pool(count: int, seed: int) -> list[str]:
    """Build a deterministic but shuffled production-style traffic mix."""
    if count <= 0:
        return []

    rng = random.Random(seed)

    n1 = int(count * TIER_SPLIT[1])
    n2 = int(count * TIER_SPLIT[2])
    n3 = count - n1 - n2

    prompts = (
        _sample_pool(TIER1_PROMPTS, n1, rng)
        + _sample_pool(TIER2_PROMPTS, n2, rng)
        + _sample_pool(TIER3_PROMPTS, n3, rng)
    )

    rng.shuffle(prompts)
    return prompts[:count]


def backup_and_clear_db() -> None:
    if DB_PATH.exists():
        backup_path = DB_PATH.with_suffix(f".db.bak-{int(time.time())}")
        shutil.move(str(DB_PATH), str(backup_path))
        print(f"backed up existing DB to {backup_path.name}")


def run_load_test(count: int, synchronous: bool, seed: int) -> None:
    prompts = build_prompt_pool(count, seed)

    print(
        f"sending {len(prompts)} prompts through "
        f"route_request_with_verification (synchronous={synchronous})..."
    )

    errors = 0
    started = time.perf_counter()

    for i, prompt in enumerate(prompts, start=1):
        try:
            result, _ = route_request_with_verification(
                prompt,
                synchronous=synchronous,
            )
            if result.response.error:
                errors += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"  [{i}] request failed: {exc}")

        if i % 50 == 0 or i == len(prompts):
            elapsed = time.perf_counter() - started
            print(
                f"  {i}/{len(prompts)} sent "
                f"({elapsed:.0f}s elapsed, {errors} errors so far)"
            )

    print("draining background verification queue...")
    shutdown(wait=True)
    print(
        f"done. {len(prompts)} requests, {errors} errors. "
        "Run `python -m scripts.generate_report` next."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=750)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="wait for verification on every request",
    )
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="back up and start data/requests.db empty",
    )
    args = parser.parse_args()

    if args.fresh:
        backup_and_clear_db()

    run_load_test(args.count, args.sync, args.seed)


if __name__ == "__main__":
    main()
