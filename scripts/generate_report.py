"""
Phase 6, steps 2-3: "Generate the final cost savings report... Calculate and
prominently display the cost reduction percentage... that number is the
headline of your portfolio piece."

Run with:  python -m scripts.generate_report

Reads whatever's currently in data/requests.db (ideally the output of
scripts/load_test.py) and writes:
    data/cost_savings_report.md    - the headline number + full breakdown
    data/report_charts/*.png       - the same four charts the Streamlit
                                      dashboard shows, as static images you
                                      can drop straight into a portfolio
                                      write-up or README without needing a
                                      live dashboard screenshot
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available - just write files
import matplotlib.pyplot as plt

from src.stats import get_daily_escalation_rate, get_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORT_PATH = DATA_DIR / "cost_savings_report.md"
CHARTS_DIR = DATA_DIR / "report_charts"


def _chart_cost_per_day(summary) -> None:
    if not summary.daily_cost:
        return
    dates = [d["date"] for d in summary.daily_cost]
    costs = [d["cost_usd"] for d in summary.daily_cost]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dates, costs, color="#4C72B0")
    ax.set_title("Cost per day")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "cost_per_day.png", dpi=150)
    plt.close(fig)


def _chart_routing_distribution(summary) -> None:
    if not summary.routing_distribution:
        return
    labels = list(summary.routing_distribution.keys())
    counts = list(summary.routing_distribution.values())
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(counts, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Routing distribution")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "routing_distribution.png", dpi=150)
    plt.close(fig)


def _chart_quality_distribution(summary) -> None:
    if not summary.quality_scores:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary.quality_scores, bins=20, color="#55A868")
    ax.axvline(0.8, color="red", linestyle="--", label="typical threshold")
    ax.set_title("Quality score distribution")
    ax.set_xlabel("Agreement score vs. reference model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "quality_score_distribution.png", dpi=150)
    plt.close(fig)


def _chart_escalation_rate(daily_escalation: list[dict]) -> None:
    if not daily_escalation:
        return
    dates = [d["date"] for d in daily_escalation]
    rates = [d["escalation_rate"] * 100 for d in daily_escalation]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, rates, marker="o", color="#C44E52")
    ax.set_title("Escalation rate over time")
    ax.set_ylabel("% of verified requests escalated")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "escalation_rate.png", dpi=150)
    plt.close(fig)


def _write_report(summary, daily_escalation: list[dict]) -> None:
    top_model = max(summary.routing_distribution, key=summary.routing_distribution.get) \
        if summary.routing_distribution else "n/a"
    quality_line = (
        f"{summary.avg_quality_score:.1%} average agreement with the reference model"
        if summary.avg_quality_score is not None else "n/a (no verified requests yet)"
    )

    report = f"""# LLM Cost Autopilot - Cost Savings Report

## Headline

> Routing to cheaper models where appropriate **cut LLM API costs by
> {summary.savings_pct:.1f}%** (${summary.savings_usd:.4f} saved) compared to
> sending every request to GPT-4o, across {summary.total_requests:,} requests
> — while maintaining {quality_line}.

| Metric | Value |
|---|---|
| Total requests routed | {summary.total_requests:,} |
| Actual spend | ${summary.total_cost_usd:.6f} |
| Hypothetical spend (GPT-4o for everything) | ${summary.hypothetical_cost_usd:.6f} |
| **Savings** | **${summary.savings_usd:.6f} ({summary.savings_pct:.1f}%)** |
| Requests verified (tier 1/2 only) | {summary.verified_requests:,} |
| Escalations | {summary.escalation_count} ({summary.escalation_rate_of_verified:.1%} of verified) |
| Average quality score | {f'{summary.avg_quality_score:.3f}' if summary.avg_quality_score is not None else 'n/a'} |
| Most-used model | {top_model} |

## Routing distribution

| Model | Requests | Share |
|---|---|---|
"""
    total = summary.total_requests or 1
    for model, count in sorted(summary.routing_distribution.items(), key=lambda kv: -kv[1]):
        report += f"| {model} | {count} | {count / total:.1%} |\n"

    report += """
## Charts

See `report_charts/` for these as standalone PNGs (drop straight into a
portfolio write-up or slide):

- `cost_per_day.png`
- `routing_distribution.png`
- `quality_score_distribution.png`
- `escalation_rate.png`

## How this was produced

1. `python -m scripts.load_test --count 750` sent 750 fresh, diverse prompts
   (different random seed than the Phase 2 training set) through the full
   router + verification pipeline.
2. Every request logged a row to `data/requests.db` (Phase 4); every tier-1/2
   request was verified against the tier-3 model in the background (Phase 3).
3. This script (`python -m scripts.generate_report`) read that data back and
   computed the numbers above the same way the live dashboard does
   (`src/stats.py`), so they'll always match what `streamlit run
   dashboard/app.py` shows.
"""
    REPORT_PATH.write_text(report)


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = get_summary()

    if summary.total_requests == 0:
        print("data/requests.db is empty - run `python -m scripts.load_test` "
              "(or `python -m scripts.seed_demo_data` for a synthetic demo) first.")
        return

    daily_escalation = get_daily_escalation_rate()

    _chart_cost_per_day(summary)
    _chart_routing_distribution(summary)
    _chart_quality_distribution(summary)
    _chart_escalation_rate(daily_escalation)
    _write_report(summary, daily_escalation)

    print(f"wrote {REPORT_PATH} and charts to {CHARTS_DIR}/")
    print(f"headline: {summary.savings_pct:.1f}% cost reduction (${summary.savings_usd:.4f} saved) "
          f"across {summary.total_requests:,} requests")


if __name__ == "__main__":
    main()
