from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocess import clean_text
from predict import analyze_message

TEST_DATA = ROOT / "data" / "unseen_test_set.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TARGETS = ["intent", "sentiment", "resolution", "escalation"]


def model_input(df, target):
    customer = df["customer_message"].fillna("").astype(str).map(clean_text)
    if target == "resolution":
        agent = df["agent_response"].fillna("").astype(str).map(clean_text)
        return customer + " [agent] " + agent
    return customer


def main():
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            "Unseen test set not found. Run: python data/create_test_set.py"
        )

    df = pd.read_csv(TEST_DATA)

    models = {
        target: joblib.load(ROOT / "models" / f"{target}_model.pkl")
        for target in TARGETS
    }

    model_summary = {}
    final_summary = {}

    for target in TARGETS:
        X = model_input(df, target)
        y_true = df[target].astype(str)
        y_pred = models[target].predict(X)

        model_summary[target] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }

        cm = confusion_matrix(y_true, y_pred, labels=list(models[target].classes_))
        pd.DataFrame(
            cm,
            index=list(models[target].classes_),
            columns=list(models[target].classes_),
        ).to_csv(REPORTS / f"unseen_{target}_confusion_matrix.csv")

    # Evaluate the actual production decision path, including rules and
    # optional agent response.
    final_predictions = {target: [] for target in TARGETS}
    for _, row in df.iterrows():
        result = analyze_message(
            row["customer_message"],
            row.get("agent_response", "")
        )
        for target in TARGETS:
            final_predictions[target].append(str(result[target]))

    for target in TARGETS:
        y_true = df[target].astype(str)
        y_pred = final_predictions[target]
        final_summary[target] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }

    summary = {
        "test_rows": int(len(df)),
        "model_only": model_summary,
        "final_pipeline": final_summary,
        "note": "This unseen set is manually curated and written differently from the synthetic training templates.",
    }

    path = REPORTS / "unseen_test_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("UNSEEN TEST RESULTS - FINAL PIPELINE")
    print("=" * 72)
    for target, values in final_summary.items():
        print(f"{target:12s} accuracy={values['accuracy']:.4f}  weighted_f1={values['weighted_f1']:.4f}")
        print(values["report"])

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
