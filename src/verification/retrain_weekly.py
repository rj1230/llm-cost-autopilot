"""
Phase 3, step 4: "Build a simple feedback loop that retrains the classifier
weekly using accumulated failure data. This is the flywheel that makes the
system get smarter over time."

Run with:  python -m src.verification.retrain_weekly

This merges data/classifier_feedback.csv (written by queue.py every time a
routing failure gets escalated) into the base dataset - a feedback row for a
prompt overrides that prompt's original label, since it reflects what
actually happened, not just an authored guess - then retrains and overwrites
data/classifier.joblib exactly like `python -m src.classifier.train` does.

For real weekly scheduling, wire this into your OS's scheduler rather than
running it by hand:

    Windows: Task Scheduler -> New Task -> weekly trigger -> action:
        program: path\\to\\venv\\Scripts\\python.exe
        arguments: -m src.verification.retrain_weekly
        start in: path\\to\\llm-cost-autopilot

    macOS/Linux: a cron entry such as
        0 3 * * 1 cd /path/to/llm-cost-autopilot && venv/bin/python -m src.verification.retrain_weekly
"""

from pathlib import Path

import pandas as pd

from src.classifier.train import DATASET_PATH, train

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FEEDBACK_PATH = DATA_DIR / "classifier_feedback.csv"
AUGMENTED_PATH = DATA_DIR / "labeled_dataset_augmented.csv"


def build_augmented_dataset() -> Path:
    base_df = pd.read_csv(DATASET_PATH)

    if not FEEDBACK_PATH.exists():
        print(f"no {FEEDBACK_PATH.name} yet - nothing to merge, training on the base dataset only.")
        base_df.to_csv(AUGMENTED_PATH, index=False)
        return AUGMENTED_PATH

    feedback_df = pd.read_csv(FEEDBACK_PATH)
    print(f"merging {len(feedback_df)} feedback rows from {len(feedback_df['prompt'].unique())} unique prompts")

    # feedback rows override the base label for the same prompt; keep the
    # *last* feedback entry per prompt in case a prompt failed more than once
    feedback_df = feedback_df.drop_duplicates(subset="prompt", keep="last")
    merged = pd.concat([base_df, feedback_df]).drop_duplicates(subset="prompt", keep="last")

    merged.to_csv(AUGMENTED_PATH, index=False)
    print(f"wrote {len(merged)} rows ({len(merged) - len(base_df)} net new/relabeled) to {AUGMENTED_PATH}")
    return AUGMENTED_PATH


def main() -> None:
    augmented_path = build_augmented_dataset()
    train(dataset_path=augmented_path)


if __name__ == "__main__":
    main()
