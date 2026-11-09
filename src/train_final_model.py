"""
Final Deployment Model Training
================================
Trains a Random Forest classifier on the COMPLETE multisubject dataset
(all 4 subjects, all 4 activities).

IMPORTANT DISCLAIMER:
This model is trained on the full available dataset for deployment/demo purposes.
Its training accuracy must NOT be interpreted as subject-independent generalization
performance. Subject-independent (LOSO) performance is reported separately in:
  results/loso_evaluation_report.txt

Do NOT use this model's training metrics to claim generalization ability.
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "data/processed/multisubject_normalized_temporal_features.csv"
MODEL_PATH   = "models/final_deployment_model.pkl"
SCHEMA_PATH  = "models/final_feature_schema.json"
REPORT_PATH  = "results/final_model_training_report.txt"

METADATA_COLS = [
    "subject_id", "activity", "activity_code",
    "trial_id", "sequence_id", "frame_id", "timestamp"
]

RF_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> str:
    print(msg)
    return msg


def section(title: str) -> str:
    bar = "=" * 60
    block = f"\n{bar}\n{title}\n{bar}"
    print(block)
    return block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    lines = []  # collect report lines

    lines.append(section("FINAL DEPLOYMENT MODEL TRAINING"))
    lines.append(log(f"Timestamp : {datetime.datetime.now().isoformat()}"))
    lines.append(log(f"Dataset   : {DATASET_PATH}"))
    lines.append(log(f"Model out : {MODEL_PATH}"))
    lines.append(log(f"Schema out: {SCHEMA_PATH}"))
    lines.append(log(f"Report out: {REPORT_PATH}"))

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    lines.append(section("1. LOADING DATASET"))

    if not os.path.exists(DATASET_PATH):
        sys.exit(f"ERROR: Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    lines.append(log(f"Rows loaded   : {len(df):,}"))
    lines.append(log(f"Columns loaded: {len(df.columns)}"))
    lines.append(log(f"Subjects found: {sorted(df['subject_id'].unique())}"))
    lines.append(log(f"Activities    : {sorted(df['activity'].unique())}"))

    # ------------------------------------------------------------------
    # 2. Extract features
    # ------------------------------------------------------------------
    lines.append(section("2. FEATURE EXTRACTION"))

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    n_features   = len(feature_cols)
    lines.append(log(f"ML feature columns : {n_features}"))

    if n_features != 158:
        sys.exit(
            f"ERROR: Expected exactly 158 feature columns, found {n_features}. "
            "Aborting to protect research integrity."
        )
    lines.append(log("Feature count verified: 158 ✓"))

    # Sanity check: no metadata column leaked into features
    leaked = [c for c in feature_cols if c in METADATA_COLS]
    if leaked:
        sys.exit(f"ERROR: Metadata columns leaked into features: {leaked}")
    lines.append(log("No metadata leakage ✓"))

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["activity"].to_numpy()

    lines.append(log(f"X shape: {X.shape}"))
    lines.append(log(f"y shape: {y.shape}"))
    lines.append(log(f"Class distribution:"))
    for cls, cnt in zip(*np.unique(y, return_counts=True)):
        lines.append(log(f"  {cls:<12}: {cnt:>6,} frames"))

    # NaN / Inf check
    if np.isnan(X).any() or np.isinf(X).any():
        sys.exit("ERROR: Feature matrix contains NaN or Inf values. Cannot train.")
    lines.append(log("NaN / Inf check passed ✓"))

    # ------------------------------------------------------------------
    # 3. Train Random Forest
    # ------------------------------------------------------------------
    lines.append(section("3. TRAINING RANDOM FOREST"))
    lines.append(log(f"Parameters: {RF_PARAMS}"))
    lines.append(log("Training on ALL subjects and ALL frames ..."))

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)

    lines.append(log("Training complete ✓"))

    # ------------------------------------------------------------------
    # 4. Training-set metrics  (for diagnostics only — NOT generalization)
    # ------------------------------------------------------------------
    lines.append(section("4. TRAINING-SET DIAGNOSTICS (NOT generalization metrics)"))

    y_pred_train = rf.predict(X)
    train_acc    = accuracy_score(y, y_pred_train)
    train_report = classification_report(y, y_pred_train, digits=4)

    lines.append(log(
        "\n*** WARNING ***\n"
        "The metrics below are computed on the TRAINING data.\n"
        "They reflect memorisation, NOT generalisation to new subjects.\n"
        "For subject-independent performance see: results/loso_evaluation_report.txt\n"
    ))
    lines.append(log(f"Training accuracy (in-sample): {train_acc*100:.2f}%"))
    lines.append(log("\nClassification report (training data):"))
    lines.append(log(train_report))

    # Top-10 feature importances
    importances = rf.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    lines.append(log("Top-10 feature importances:"))
    for rank, idx in enumerate(top_idx, 1):
        lines.append(log(f"  {rank:2d}. {feature_cols[idx]:<45} {importances[idx]:.6f}"))

    # ------------------------------------------------------------------
    # 5. Save model
    # ------------------------------------------------------------------
    lines.append(section("5. SAVING MODEL AND SCHEMA"))

    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, MODEL_PATH, compress=3)
    lines.append(log(f"Model saved → {MODEL_PATH}"))

    schema = {
        "feature_names"  : feature_cols,
        "n_features"     : n_features,
        "classes"        : rf.classes_.tolist(),
        "rf_params"      : RF_PARAMS,
        "training_rows"  : int(len(df)),
        "subjects_used"  : sorted(df["subject_id"].unique().tolist()),
        "activities"     : sorted(df["activity"].unique().tolist()),
        "created_at"     : datetime.datetime.now().isoformat(),
        "disclaimer"     : (
            "This model was trained on the complete available multisubject dataset "
            "for deployment/demo purposes. Its performance must not be interpreted as "
            "subject-independent generalization performance. Subject-independent "
            "performance is reported separately using LOSO evaluation."
        )
    }
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2)
    lines.append(log(f"Schema saved → {SCHEMA_PATH}"))

    # ------------------------------------------------------------------
    # 6. Verification
    # ------------------------------------------------------------------
    lines.append(section("6. VERIFICATION"))

    assert os.path.exists(MODEL_PATH),  "FAIL: model file missing"
    lines.append(log("model file exists ✓"))

    assert os.path.exists(SCHEMA_PATH), "FAIL: schema file missing"
    lines.append(log("schema file exists ✓"))

    loaded_model = joblib.load(MODEL_PATH)
    lines.append(log("model loads with joblib ✓"))

    sample_pred = loaded_model.predict(X[:1])
    assert sample_pred[0] in rf.classes_, "FAIL: prediction returned unknown class"
    lines.append(log(f"sample prediction: {sample_pred[0]} ✓"))

    with open(SCHEMA_PATH) as fh:
        loaded_schema = json.load(fh)
    assert loaded_schema["n_features"] == 158, "FAIL: schema n_features != 158"
    assert len(loaded_schema["feature_names"]) == 158, "FAIL: schema feature list length != 158"
    lines.append(log("schema n_features == 158 ✓"))
    lines.append(log(f"schema classes: {loaded_schema['classes']}"))

    # ------------------------------------------------------------------
    # 7. Write report
    # ------------------------------------------------------------------
    lines.append(section("7. WRITING REPORT"))

    report_text = "\n".join(
        line if isinstance(line, str) else str(line)
        for line in lines
    )

    disclaimer_block = """
========================================================
DEPLOYMENT DISCLAIMER
========================================================
This model was trained on the COMPLETE available multisubject dataset
(all 4 subjects, all 4 activities, all 21,211 frames) for deployment
and live-demo purposes.

Its training-set accuracy MUST NOT be interpreted as subject-independent
generalization performance.

Subject-independent (Leave-One-Subject-Out) performance is reported
separately and transparently in:
  results/loso_evaluation_report.txt

Summary of LOSO results:
  Random Forest Mean Accuracy : 66.26% ± 11.34%
  Random Forest Mean Macro F1 : 64.84% ± 13.13%

These LOSO numbers — not the training metrics above — represent the
true expected performance on a new, unseen subject.
========================================================
"""

    os.makedirs("results", exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        fh.write(report_text)
        fh.write(disclaimer_block)

    print(disclaimer_block)
    print(log(f"\nReport saved → {REPORT_PATH}"))

    print("\n" + "=" * 60)
    print("FINAL_MODEL_TRAINING_COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

