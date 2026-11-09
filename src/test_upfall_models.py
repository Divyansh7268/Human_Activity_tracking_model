"""
src/test_upfall_models.py

Initial end-to-end model test using UP-Fall pipeline-validation data.

IMPORTANT: This dataset was constructed from personal-dataset frames reformatted
into the UP-Fall schema to verify the adapter pipeline. It is NOT a real
multi-subject UP-Fall benchmark. These results represent PIPELINE VALIDATION ONLY.

Dataset: data/processed/upfall_temporal_features.csv
  - 792 rows
  - 3 activity classes: standing, sitting, walking (no running in UP-Fall ADL subset)
  - 247 spatial features + 8 temporal features = 255 features

Split strategy: Hold out all frames from two complete trial sequences as test set.
  Prevents any data leakage across sequences from the same trial.

Outputs:
  models/upfall_temporal_classifier_test.pkl
  results/upfall_initial_model_test.txt
  results/upfall_initial_confusion_matrix.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

matplotlib.use("Agg")

TEMPORAL_FEATURES_PATH = "data/processed/upfall_temporal_features.csv"
POSE_DATASET_PATH = "data/processed/upfall_pose_dataset.csv"
MODEL_OUTPUT_PATH = "models/upfall_temporal_classifier_test.pkl"
RESULTS_TEXT_PATH = "results/upfall_initial_model_test.txt"
CONFUSION_MATRIX_PATH = "results/upfall_initial_confusion_matrix.png"

RANDOM_STATE = 42
CLASSES = ["standing", "sitting", "walking"]  # 3 classes — no running in UP-Fall ADL subset


def load_data():
    df = pd.read_csv(TEMPORAL_FEATURES_PATH)
    print(f"Loaded temporal features: {df.shape}")
    print(f"Activity distribution:\n{df['activity'].value_counts()}")
    return df


def sequence_based_split(df: pd.DataFrame, pose_df: pd.DataFrame):
    """
    Split by complete trial sequences to avoid leakage.
    Test set = last sequence for each activity class (held out completely).
    Train set = all remaining sequences.
    """
    # Reconstruct sequence_id mapping from pose_df (same row order, minus dropped first frames)
    # pose_df has 800 rows; temporal_df has 792 rows (each sequence loses its first frame)
    # We need to find which temporal rows belong to which sequences.
    # Strategy: each sequence contributes 99 rows to temporal_df (100 - 1).
    # Sequences in sorted order: C1_S1_A6_T1, C1_S1_A6_T2, C1_S1_A7_T1, C1_S1_A7_T2,
    #                             C1_S1_A8_T1, C1_S1_A8_T2, C1_S2_A7_T1, C1_S2_A8_T1
    sequences_in_pose = sorted(pose_df["sequence_id"].unique())

    # Build sequence ID series aligned to temporal_df rows
    seq_labels = []
    for seq_id in sequences_in_pose:
        n = (pose_df["sequence_id"] == seq_id).sum()
        if n >= 2:
            seq_labels.extend([seq_id] * (n - 1))  # first frame dropped

    seq_series = pd.Series(seq_labels, name="sequence_id")
    assert len(seq_series) == len(df), f"Mismatch: {len(seq_series)} vs {len(df)}"

    df = df.copy()
    df["_seq_id"] = seq_series.values

    # Hold out test sequences:
    #   walking  — C1_S1_A6_T2  (second walking trial)
    #   standing — C1_S2_A7_T1  (second subject standing)
    #   sitting  — C1_S2_A8_T1  (second subject sitting)
    test_sequences = ["C1_S1_A6_T2", "C1_S2_A7_T1", "C1_S2_A8_T1"]

    test_mask = df["_seq_id"].isin(test_sequences)
    train_df = df[~test_mask].drop(columns=["_seq_id"])
    test_df = df[test_mask].drop(columns=["_seq_id"])

    print(f"\nTrain sequences: {df[~test_mask]['_seq_id'].unique().tolist()}")
    print(f"Test  sequences: {test_sequences}")
    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
    print(f"Train activity distribution:\n{train_df['activity'].value_counts()}")
    print(f"Test  activity distribution:\n{test_df['activity'].value_counts()}")

    X_train = train_df.drop(columns=["activity"])
    y_train = train_df["activity"]
    X_test = test_df.drop(columns=["activity"])
    y_test = test_df["activity"]

    return X_train, X_test, y_train, y_test


def build_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=1.0)),
    ])


def build_random_forest():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)

    per_class_f1 = {}
    for cls in CLASSES:
        f1 = f1_score(y_test == cls, y_pred == cls, zero_division=0)
        per_class_f1[cls] = f1

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Weighted F1     : {f1_weighted:.4f}")
    for cls in CLASSES:
        print(f"  F1 [{cls:10s}]: {per_class_f1[cls]:.4f}")
    print(f"\n  Classification Report:\n{report}")

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "per_class_f1": per_class_f1,
        "report": report,
        "cm": cm,
    }


def save_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_yticklabels([c.capitalize() for c in CLASSES])
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def save_results_report(results_list, train_rows, test_rows, best_name):
    os.makedirs("results", exist_ok=True)
    lines = []

    lines.append("=" * 68)
    lines.append("UP-FALL INITIAL END-TO-END MODEL TEST — PIPELINE VALIDATION REPORT")
    lines.append("=" * 68)
    lines.append("")
    lines.append("⚠  IMPORTANT DISCLAIMER")
    lines.append("   This report reflects a pipeline validation test only.")
    lines.append("   The 'UP-Fall' data used here consists of personal-dataset frames")
    lines.append("   reformatted into the UP-Fall CSV schema to verify adapter compatibility.")
    lines.append("   It does NOT represent real UP-Fall subject recordings.")
    lines.append("   These results CANNOT be presented as a multi-subject benchmark.")
    lines.append("")
    lines.append("─" * 68)
    lines.append("1. DATASET SUMMARY")
    lines.append("─" * 68)
    lines.append(f"   Source          : data/processed/upfall_temporal_features.csv")
    lines.append(f"   Total rows      : {train_rows + test_rows}")
    lines.append(f"   Activity classes: 3 (standing, sitting, walking — no running in UP-Fall ADL)")
    lines.append(f"   Features        : 247 spatial + 8 temporal = 255 total")
    lines.append(f"   Train rows      : {train_rows}")
    lines.append(f"   Test  rows      : {test_rows}")
    lines.append(f"   Split strategy  : Sequence-based hold-out (complete trial sequences held out)")
    lines.append("")
    lines.append("─" * 68)
    lines.append("2. MODEL RESULTS")
    lines.append("─" * 68)

    for r in results_list:
        lines.append(f"\n  Model: {r['name']}")
        lines.append(f"    Accuracy        : {r['accuracy']:.4f}")
        lines.append(f"    Weighted F1     : {r['f1_weighted']:.4f}")
        for cls in CLASSES:
            lines.append(f"    F1 [{cls:10s}]: {r['per_class_f1'][cls]:.4f}")
        lines.append(f"\n    Per-class Report:\n")
        for line in r["report"].splitlines():
            lines.append(f"      {line}")

    lines.append("")
    lines.append("─" * 68)
    lines.append(f"3. BEST MODEL: {best_name}")
    lines.append(f"   Saved to: {MODEL_OUTPUT_PATH}")
    lines.append("")
    lines.append("─" * 68)
    lines.append("4. CONFUSION MATRIX")
    lines.append(f"   Saved to: {CONFUSION_MATRIX_PATH}")
    lines.append("")
    lines.append("─" * 68)
    lines.append("5. INTERPRETATION & LIMITATIONS")
    lines.append("─" * 68)
    lines.append("   • The dataset was derived from personal recordings, so these")
    lines.append("     results primarily validate that the full pipeline (feature")
    lines.append("     engineering → temporal features → training → evaluation) works")
    lines.append("     end-to-end with the UP-Fall data schema.")
    lines.append("   • A 3-class problem (no running) is simpler than the 4-class")
    lines.append("     personal-dataset scenario.")
    lines.append("   • To obtain generalizable multi-subject results, the actual")
    lines.append("     UP-Fall ADL videos (A6, A7, A8) need to be processed through")
    lines.append("     MediaPipe Pose to extract real skeleton data.")
    lines.append("   • Do NOT cite these results as a research benchmark.")
    lines.append("")
    lines.append("=" * 68)

    report_text = "\n".join(lines)
    with open(RESULTS_TEXT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nSaved results report → {RESULTS_TEXT_PATH}")
    print(report_text)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load data
    df = load_data()
    pose_df = pd.read_csv(POSE_DATASET_PATH)

    # Sequence-based train/test split
    X_train, X_test, y_train, y_test = sequence_based_split(df, pose_df)

    # Train and evaluate both models
    lr_result = evaluate_model("Logistic Regression", build_logistic_regression(),
                               X_train, X_test, y_train, y_test)
    rf_result = evaluate_model("Random Forest", build_random_forest(),
                               X_train, X_test, y_train, y_test)

    results_list = [lr_result, rf_result]

    # Pick best model by weighted F1
    best = max(results_list, key=lambda r: r["f1_weighted"])
    print(f"\nBest model: {best['name']} (Weighted F1 = {best['f1_weighted']:.4f})")

    # Save best model
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"Saved best model → {MODEL_OUTPUT_PATH}")

    # Confusion matrix for best model
    save_confusion_matrix(
        best["cm"],
        f"UP-Fall Pipeline Validation — {best['name']}\n(3-class: standing, sitting, walking)",
        CONFUSION_MATRIX_PATH,
    )

    # Full results report
    save_results_report(results_list, len(X_train), len(X_test), best["name"])


if __name__ == "__main__":
    main()

