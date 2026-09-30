"""
Phase 2, step 3: "Train the classifier... Track accuracy and confusion
matrix. Anything above 80% accuracy on a held-out set is fine for V1."

Run with:  python -m src.classifier.train

Trains both a logistic regression and a random forest (the guide names both
as reasonable V1 choices) on the heuristic features from features.py, picks
whichever scores higher on the held-out test split, and saves that one to
data/classifier.joblib for predict.py / routing.py to load.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from src.classifier.features import FEATURE_NAMES, features_to_vector

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATASET_PATH = DATA_DIR / "labeled_dataset.csv"
MODEL_PATH = DATA_DIR / "classifier.joblib"


def load_dataset(dataset_path: Path = DATASET_PATH) -> tuple[list[list[float]], list[int]]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} not found - run `python -m data.generate_dataset` first."
        )
    df = pd.read_csv(dataset_path)
    X = [features_to_vector(p) for p in df["prompt"]]
    y = df["tier"].tolist()
    return X, y


def train(dataset_path: Path = DATASET_PATH) -> None:
    X, y = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    best_name, best_model, best_acc = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n--- {name} ---")
        print(f"accuracy: {acc:.3f}")
        print("confusion matrix (rows=actual, cols=predicted, labels=[1,2,3]):")
        print(confusion_matrix(y_test, preds, labels=[1, 2, 3]))
        print(classification_report(y_test, preds, labels=[1, 2, 3], zero_division=0))

        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    print(f"\nselected {best_name} (accuracy {best_acc:.3f}) -> {MODEL_PATH}")
    if best_acc < 0.80:
        print(
            "warning: below the 80% V1 bar the guide sets - consider adding more "
            "templates to data/generate_dataset.py or engineering more features "
            "before moving on to Phase 3."
        )

    joblib.dump({"model": best_model, "feature_names": FEATURE_NAMES}, MODEL_PATH)


if __name__ == "__main__":
    train()
