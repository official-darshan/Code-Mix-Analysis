from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocess import add_text_features, clean_text

DATA = ROOT / "data" / "support_conversations.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

TARGETS = ["intent", "sentiment", "resolution", "escalation"]


def make_pipeline() -> Pipeline:
    word_features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=20000,
    )

    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=10000,
    )

    features = FeatureUnion(
        [
            ("word", word_features),
            ("char", char_features),
        ]
    )

    classifier = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=100,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight="balanced",
        average=True,
        random_state=42,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1": round(float(f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
    }


def train_one(X, y, name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = evaluate(y_test, pred)
    report_text = classification_report(y_test, pred, zero_division=0)
    labels = list(model.named_steps["classifier"].classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)

    model_path = MODELS / f"{name}_model.pkl"
    joblib.dump(model, model_path, compress=3)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        REPORTS / f"{name}_confusion_matrix.csv"
    )
    (REPORTS / f"{name}_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"{name.upper()} MODEL")
    print("=" * 72)
    print(f"Train rows : {len(X_train)}")
    print(f"Test rows  : {len(X_test)}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print(report_text)

    return {"target": name, "classes": labels, **metrics}


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            "Dataset not found. Run: python data/generate_dataset.py"
        )

    df = pd.read_csv(DATA)
    required = {"customer_message", "agent_response", *TARGETS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    if len(df) < 10000:
        raise ValueError(
            "Run the improved dataset generator first: python data/generate_dataset.py"
        )

    df = add_text_features(df)
    df["clean_agent_response"] = df["agent_response"].fillna("").astype(str).apply(clean_text)
    df["conversation_text"] = (
        df["clean_message"] + " [agent] " + df["clean_agent_response"]
    )

    df = df.drop_duplicates(subset=["customer_message", "agent_response"]).reset_index(drop=True)

    print(f"Dataset rows after duplicate removal: {len(df)}")

    results = []

    # Intent, sentiment and escalation are customer-message tasks.
    customer_text_targets = ["intent", "sentiment", "escalation"]
    for target in customer_text_targets:
        results.append(train_one(df["clean_message"], df[target], target))

    # Resolution is a conversation-level task, so it learns from both sides.
    results.append(train_one(df["conversation_text"], df["resolution"], "resolution"))

    summary = {
        "dataset_rows": int(len(df)),
        "architecture": {
            "intent": "customer_message",
            "sentiment": "customer_message",
            "escalation": "customer_message",
            "resolution": "customer_message + agent_response",
        },
        "training_note": (
            "The dataset is synthetic educational data. SGDClassifier with log-loss is used for scalable text classification on 50,000 rows. The separate unseen test set is the preferred generalization check."
        ),
        "models": results,
    }

    summary_path = REPORTS / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 72)
    print(f"Models : {MODELS}")
    print(f"Report : {summary_path}")


if __name__ == "__main__":
    main()
