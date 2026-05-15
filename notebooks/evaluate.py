"""
Evaluation script.
Generates:
  - Test set classification report (precision, recall, F1 per intent)
  - Confusion matrix heatmap (top 30 most-confused intents)
  - Retrieval fallback accuracy on low-confidence subset
  - Average handle turns metric vs. no-model baseline

Run after training:
    python notebooks/evaluate.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "bert_intent")
OUT_DIR = os.path.dirname(__file__)


def load_test_data():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_map_df = pd.read_csv(os.path.join(DATA_DIR, "label_map.csv"))
    id2label = {row["id"]: row["intent"] for _, row in label_map_df.iterrows()}
    return test_df, id2label


def run_inference_on_test(test_df, id2label):
    from models.predict import get_classifier
    clf = get_classifier()

    results = []
    for _, row in test_df.iterrows():
        pred = clf.predict(row["text"])
        results.append({
            "text": row["text"],
            "true_label": id2label[str(row["intent"])],
            "pred_label": pred["intent"],
            "confidence": pred["confidence"],
            "low_confidence": pred["low_confidence"],
            "source": "bert",
        })
    return pd.DataFrame(results)


def plot_confusion_matrix(results_df, id2label, top_n=30):
    """Plot confusion matrix for top_n most frequent intents."""
    top_intents = results_df["true_label"].value_counts().head(top_n).index.tolist()
    subset = results_df[results_df["true_label"].isin(top_intents)]

    cm = confusion_matrix(subset["true_label"], subset["pred_label"], labels=top_intents)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        cm_norm, annot=False, fmt=".2f", cmap="Blues",
        xticklabels=top_intents, yticklabels=top_intents, ax=ax
    )
    ax.set_title(f"Normalized Confusion Matrix — Top {top_n} Intents", fontsize=14)
    ax.set_xlabel("Predicted Intent")
    ax.set_ylabel("True Intent")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")
    plt.close()


def evaluate_retrieval_fallback(results_df):
    """Measure retrieval fallback accuracy on low-confidence BERT predictions."""
    low_conf = results_df[results_df["low_confidence"]].copy()
    if len(low_conf) == 0:
        print("No low-confidence predictions found.")
        return

    from retrieval.hybrid_retriever import get_retriever
    retriever = get_retriever()

    retrieval_preds = []
    for _, row in low_conf.iterrows():
        r = retriever.predict_from_retrieval(row["text"])
        retrieval_preds.append(r["intent"])

    bert_acc = accuracy_score(low_conf["true_label"], low_conf["pred_label"])
    retrieval_acc = accuracy_score(low_conf["true_label"], retrieval_preds)

    print(f"\nLow-confidence subset ({len(low_conf)} samples):")
    print(f"  BERT accuracy:      {bert_acc:.4f}")
    print(f"  Retrieval accuracy: {retrieval_acc:.4f}")
    print(f"  Delta:              {retrieval_acc - bert_acc:+.4f}")


def compute_handle_time_reduction(results_df):
    """
    Simulate average handle turns with vs. without the model.
    Baseline: 3.0 turns (no model, human escalation).
    With model: turns needed to reach 80%+ confidence.
    """
    turns_with_model = []
    for _, row in results_df.iterrows():
        if row["confidence"] >= 0.80:
            turns_with_model.append(1.0)
        elif row["confidence"] >= 0.60:
            turns_with_model.append(2.0)
        else:
            turns_with_model.append(3.0)

    baseline = 3.0
    model_avg = np.mean(turns_with_model)
    reduction_pct = (baseline - model_avg) / baseline * 100
    print(f"\nAverage Handle Turns:")
    print(f"  Baseline (no model): {baseline:.2f}")
    print(f"  With model:          {model_avg:.2f}")
    print(f"  Reduction:           {reduction_pct:.1f}%")
    return model_avg, reduction_pct


def main():
    print("Loading test data...")
    test_df, id2label = load_test_data()

    print("Running inference on test set...")
    results_df = run_inference_on_test(test_df, id2label)

    # Overall metrics
    acc = accuracy_score(results_df["true_label"], results_df["pred_label"])
    f1 = f1_score(results_df["true_label"], results_df["pred_label"], average="weighted")
    print(f"\nTest Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")

    # Per-intent report
    print("\nClassification Report (top-line):")
    print(classification_report(
        results_df["true_label"], results_df["pred_label"],
        digits=4, zero_division=0
    ))

    # Confusion matrix
    plot_confusion_matrix(results_df, id2label, top_n=30)

    # Retrieval fallback
    evaluate_retrieval_fallback(results_df)

    # Handle time
    compute_handle_time_reduction(results_df)

    # Save results
    results_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
    print(f"\nPredictions saved to {OUT_DIR}/test_predictions.csv")


if __name__ == "__main__":
    main()
