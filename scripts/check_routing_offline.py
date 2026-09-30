"""
Quick sanity check with no API keys required: classifies the Phase 1 baseline
prompts and shows which model each would be routed to, without actually
calling any provider. Useful right after training to eyeball whether the
routing "feels" right before you spend real API budget on it.

Run with:  python -m scripts.check_routing_offline
"""

from src.classifier.predict import classify_complexity
from src.routing import _load_config
from tests.prompts import BASELINE_PROMPTS

TIER_LABELS = {1: "simple", 2: "moderate", 3: "complex"}


def main() -> None:
    config = _load_config()
    for prompt in BASELINE_PROMPTS:
        tier = classify_complexity(prompt)
        model = config["routing"][tier]
        preview = prompt if len(prompt) <= 70 else prompt[:67] + "..."
        print(f"[tier {tier} - {TIER_LABELS[tier]:<8}] -> {model:<15} {preview}")


if __name__ == "__main__":
    main()
