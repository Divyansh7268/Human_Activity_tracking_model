"""
Baseline Pose Classifier Training & Evaluation

Loads the engineered feature dataset, trains two classifiers
(Logistic Regression and Random Forest), evaluates them, compares results,
and saves the best model as a Scikit-learn Pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

FEATURES_PATH = os.path.join("data", "processed", "features.csv")
MODEL_PATH = os.path.join("models", "activity_classifier.pkl")
RESULTS_DIR = "results"
MODELS_DIR = "models"

RANDOM_STATE = 42
ACTIVITY_CLASSES = ["running", "sitting", "standing", "walking"]


# ─── Data Loading & Validation ───────────────────────────────────────────────

def load_dataset(path: str):
    """Load features.csv and return X (features) and y (labels)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Sanity checks
    if "activity" not in df.columns:
        raise ValueError("'activity' column missing from the dataset.")

    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"[WARNING] {null_count} missing values found — dropping those rows.")
        df = df.dropna()

    non_numeric = df.drop(columns=["activity"]).select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Unexpected non-numeric feature columns: {non_numeric}")

    print("Class distribution:")
    print(df["activity"].value_counts().to_string())

    X = df.drop(columns=["activity"])
    y = df["activity"]
    return X, y


# ─── Metrics ─────────────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a fitted model on the test set and print a classification report."""
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*50}")
    print(f"Model: {name}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "model_name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": y_pred,
    }


# ─── Visualization ───────────────────────────────────────────────────────────

def save_confusion_matrix(name: str, y_test, y_pred, filename: str) -> None:
    """Plot and save a confusion matrix for the given model."""
    # Only keep classes that actually appear in the test labels
    present_classes = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {filename}")


def save_f1_comparison_chart(results: list[dict], filename: str) -> None:
    """Bar chart comparing model F1 scores."""
    names = [r["model_name"] for r in results]
    f1_scores = [r["f1_score"] for r in results]
    colors = ["#3c8dbc", "#2b5c8f"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, f1_scores, color=colors[:len(names)], edgecolor="black", width=0.4)
    plt.ylim(0, 1.05)
    plt.ylabel("Weighted F1 Score", fontsize=12)
    plt.title("Model F1 Score Comparison", fontsize=14, pad=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, f1_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved F1 comparison chart: {filename}")


# ─── Main Training Pipeline ──────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    X, y = load_dataset(FEATURES_PATH)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape        : {y.shape}")

    # 2. Train/test split (stratified so each class is represented proportionally)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")

    # ─── Model 1: Logistic Regression ──────────────────────────────────────
    # Wrapped in a Pipeline so the scaler is fitted only on training data
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_results = evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)

    save_confusion_matrix(
        "Logistic Regression",
        y_test,
        lr_results["predictions"],
        os.path.join(RESULTS_DIR, "confusion_matrix_logistic_regression.png"),
    )

    # ─── Model 2: Random Forest ─────────────────────────────────────────────
    # Random Forest is tree-based so scaling isn't necessary,
    # but keeping it in a pipeline makes the interface consistent.
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    ])
    rf_pipeline.fit(X_train, y_train)

    rf_results = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)

    save_confusion_matrix(
        "Random Forest",
        y_test,
        rf_results["predictions"],
        os.path.join(RESULTS_DIR, "confusion_matrix_random_forest.png"),
    )

    # ─── Compare and select best model ──────────────────────────────────────
    all_results = [lr_results, rf_results]
    best = max(all_results, key=lambda r: r["f1_score"])

    print(f"\n{'─'*50}")
    print(f"Best model  : {best['model_name']}")
    print(f"Best F1     : {best['f1_score']:.4f}")

    # ─── Comparison table ───────────────────────────────────────────────────
    comparison_df = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1 Score": round(r["f1_score"], 4),
        }
        for r in all_results
    ])
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel Comparison:\n{comparison_df.to_string(index=False)}")
    print(f"\nSaved comparison table: {comparison_path}")

    # ─── Save best model (Pipeline = scaler + classifier) ───────────────────
    best_pipeline = lr_pipeline if best["model_name"] == "Logistic Regression" else rf_pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved best model ({best['model_name']}) to: {MODEL_PATH}")

    # ─── Visualizations ─────────────────────────────────────────────────────
    save_f1_comparison_chart(
        all_results,
        os.path.join(RESULTS_DIR, "model_f1_comparison.png"),
    )

    # ─── Basic model verification ────────────────────────────────────────────
    print("\n--- Model Verification ---")
    loaded_model = joblib.load(MODEL_PATH)
    sample_X = X_test.iloc[:5]
    sample_y = y_test.iloc[:5].tolist()
    sample_preds = loaded_model.predict(sample_X).tolist()
    print("Actual   :", sample_y)
    print("Predicted:", sample_preds)
    print("Verification complete — model loads and predicts correctly.")


if __name__ == "__main__":
    main()

