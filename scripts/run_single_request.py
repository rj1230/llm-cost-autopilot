"""
Sends one real request through the full Phase 1-3 pipeline: classify ->
route -> call -> (if not already tier 3) verify -> maybe escalate. Requires
real API keys in .env - this is not a mocked/offline script (see
tests/test_verification.py and scripts/check_routing_offline.py for those).

Usage:
    python -m scripts.run_single_request "Summarize the following: ..."
    python -m scripts.run_single_request "..." --sync   # wait for verification
"""

import sys

from src.routing import route_request_with_verification


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    synchronous = "--sync" in sys.argv

    if not args:
        print('Usage: python -m scripts.run_single_request "your prompt here" [--sync]')
        raise SystemExit(1)

    prompt = args[0]
    result, verification = route_request_with_verification(prompt, synchronous=synchronous)

    print(f"tier:          {result.tier}")
    print(f"routed model:  {result.routed_model}{' (fallback)' if result.used_fallback else ''}")
    print(f"cost:          ${result.response.cost_usd:.6f}")
    print(f"latency:       {result.response.latency_s:.2f}s")
    print(f"output:        {result.response.output_text[:200]}")

    if verification is None:
        print("\n(tier 3 - already the top model, nothing to verify against)")
    elif synchronous:
        print(f"\nverification:  passed={verification.passed} score={verification.score:.2f} "
              f"escalated={verification.escalated}")
        if verification.escalated:
            print(f"escalated to:  {verification.reference_model}  "
                  f"(cost delta ${verification.cost_delta_usd:.6f})")
            print(f"better output: {verification.final_response.output_text[:200]}")
    else:
        print("\nverification queued in the background - check data/verification_log.jsonl "
              "shortly, or rerun with --sync to wait for it.")


if __name__ == "__main__":
    main()
