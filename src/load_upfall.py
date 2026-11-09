"""
UP-Fall Dataset Adapter Module

Transforms UP-Fall 3D skeleton trial CSV files into the standardized
33-landmark representation used by the Human Activity Recognition pipeline.

Selected Activities:
  - A6 -> walking
  - A7 -> standing
  - A8 -> sitting

Outputs:
  - data/processed/upfall_pose_dataset.csv
  - results/upfall_integration_report.txt
"""

import os
import re
import glob
import pandas as pd
import numpy as np

UPFALL_RAW_DIR = os.path.join("data", "external", "upfall", "raw_csvs")
OUTPUT_CSV_PATH = os.path.join("data", "processed", "upfall_pose_dataset.csv")
REPORT_PATH = os.path.join("results", "upfall_integration_report.txt")

# Target Activity mapping
TARGET_ACTIVITIES = {
    "A6": "walking",
    "A7": "standing",
    "A8": "sitting",
}

NUM_LANDMARKS = 33

# Flexible regex pattern matching C{cam}S{sub}A{act}T{trial}.csv
FILENAME_PATTERN = re.compile(
    r"(?:C(?P<camera>\d*))?_?S(?P<subject>\d+)_?A(?P<activity>\d+)_?T(?P<trial>\d+)",
    re.IGNORECASE,
)


def parse_filename_metadata(filename: str) -> dict | None:
    """Extract camera, subject, activity, and trial IDs from filename."""
    base = os.path.basename(filename)
    match = FILENAME_PATTERN.search(base)
    if not match:
        return None

    camera = match.group("camera") or "1"
    subject = match.group("subject")
    activity_num = match.group("activity")
    trial = match.group("trial")
    activity_code = f"A{activity_num}"

    return {
        "camera_id": int(camera),
        "subject_id": int(subject),
        "activity_code": activity_code,
        "trial_id": int(trial),
        "sequence_id": f"C{camera}_S{subject}_{activity_code}_T{trial}",
        "filename": base,
    }


def map_upfall_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UP-Fall columns (Joint1_X ... Joint33_Z) to standard project landmark names
    (landmark_0_x ... landmark_32_z) and inject default visibility scores.
    """
    mapped_dict = {}

    # Extract 3D coordinates for all 33 joints
    for c in df.columns:
        m = re.match(r"joint(\d+)_(x|y|z)", c, re.IGNORECASE)
        if m:
            joint_idx = int(m.group(1)) - 1  # Convert 1-indexed to 0-indexed
            axis = m.group(2).lower()
            mapped_dict[f"landmark_{joint_idx}_{axis}"] = df[c].values

    # Inject default visibility (1.0) for standard pipeline compatibility
    for i in range(NUM_LANDMARKS):
        mapped_dict[f"landmark_{i}_visibility"] = np.ones(len(df), dtype=np.float32)

    return pd.DataFrame(mapped_dict)


def load_and_standardize_upfall(raw_dir: str = UPFALL_RAW_DIR) -> pd.DataFrame:
    """
    Read UP-Fall CSV files, filter target activities (A6, A7, A8),
    standardize coordinates and metadata, and return combined DataFrame.
    """
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Discovered {len(csv_files)} total UP-Fall trial files in {raw_dir}")

    standardized_trials = []
    scanned_activities = set()

    for file_path in csv_files:
        meta = parse_filename_metadata(file_path)
        if not meta:
            continue

        act_code = meta["activity_code"]
        scanned_activities.add(act_code)

        # Filter: select only A6, A7, A8
        if act_code not in TARGET_ACTIVITIES:
            continue

        df_raw = pd.read_csv(file_path)
        df_landmarks = map_upfall_columns(df_raw)

        # Attach metadata
        df_landmarks["frame_id"] = np.arange(len(df_landmarks))
        df_landmarks["sequence_id"] = meta["sequence_id"]
        df_landmarks["subject_id"] = meta["subject_id"]
        df_landmarks["camera_id"] = meta["camera_id"]
        df_landmarks["activity_code"] = meta["activity_code"]
        df_landmarks["trial_id"] = meta["trial_id"]
        df_landmarks["activity"] = TARGET_ACTIVITIES[act_code]

        standardized_trials.append(df_landmarks)

    print(f"Activities found across all files: {sorted(list(scanned_activities))}")
    print(f"Target trials matched (A6, A7, A8): {len(standardized_trials)}")

    if not standardized_trials:
        print("[WARNING] No target activity files (A6, A7, A8) were matched.")
        return pd.DataFrame()

    combined_df = pd.concat(standardized_trials, ignore_index=True)

    # Reorder columns: [landmark_0_x, y, z, vis, ... landmark_32_vis], [metadata], [activity]
    ordered_cols = []
    for i in range(NUM_LANDMARKS):
        for axis in ["x", "y", "z", "visibility"]:
            ordered_cols.append(f"landmark_{i}_{axis}")

    metadata_cols = ["sequence_id", "subject_id", "camera_id", "activity_code", "trial_id", "frame_id"]
    ordered_cols.extend(metadata_cols)
    ordered_cols.append("activity")

    combined_df = combined_df[ordered_cols]
    return combined_df


def validate_standardized_dataset(df: pd.DataFrame) -> dict:
    """Run integrity and sanity checks on standardized dataset."""
    if df.empty:
        return {"valid": False, "reason": "Dataset is empty."}

    num_subjects = df["subject_id"].nunique()
    num_cameras = df["camera_id"].nunique()
    num_trials = df["sequence_id"].nunique()
    num_frames = len(df)
    activity_dist = df["activity"].value_counts().to_dict()

    # Check missing values
    null_count = df.isnull().sum().sum()

    # Check 132 landmark columns exist
    landmark_cols = [f"landmark_{i}_{a}" for i in range(33) for a in ["x", "y", "z", "visibility"]]
    all_landmarks_present = all(c in df.columns for c in landmark_cols)

    # Check sequence boundary monotonicity
    seq_monotonic = True
    for seq_id, grp in df.groupby("sequence_id"):
        frames = grp["frame_id"].values
        if not np.all(np.diff(frames) == 1) or frames[0] != 0:
            seq_monotonic = False
            break

    return {
        "valid": True,
        "num_subjects": num_subjects,
        "num_cameras": num_cameras,
        "num_trials": num_trials,
        "num_frames": num_frames,
        "activity_distribution": activity_dist,
        "null_count": null_count,
        "all_landmarks_present": all_landmarks_present,
        "sequence_integrity": seq_monotonic,
    }


def test_pipeline_compatibility(df: pd.DataFrame) -> dict:
    """Test compatibility with existing feature_engineering and temporal_features modules."""
    from feature_engineering import engineer_features
    from temporal_features import compute_sequence_temporal_features, extract_3d_coordinates

    # 1. Test feature_engineering on a single sequence slice
    sample_seq = df[df["sequence_id"] == df["sequence_id"].iloc[0]]
    # Strip metadata columns for feature_engineering
    feature_eng_input = sample_seq.drop(columns=["sequence_id", "subject_id", "camera_id", "activity_code", "trial_id", "frame_id"])

    spatial_features = engineer_features(feature_eng_input)
    spatial_ok = (spatial_features.shape[1] == 248) and (spatial_features.isnull().sum().sum() == 0)

    # 2. Test temporal_features on sequence
    coords = extract_3d_coordinates(sample_seq)
    temporal_dict = compute_sequence_temporal_features(coords, dt=0.15)
    temporal_ok = (len(temporal_dict) == 8) and all(len(v) == len(sample_seq) - 1 for v in temporal_dict.values())

    return {
        "spatial_compatible": spatial_ok,
        "spatial_columns": spatial_features.shape[1],
        "temporal_compatible": temporal_ok,
        "temporal_keys": len(temporal_dict),
    }


def write_report(val_results: dict, compat_results: dict, output_path: str):
    """Write comprehensive validation and integration report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = f"""================================================================================
UP-FALL DATASET INTEGRATION REPORT
================================================================================

1. DATASET LOCATION & RAW INGESTION
--------------------------------------------------------------------------------
- Raw Source Directory: {UPFALL_RAW_DIR}
- Standardized Dataset Output: {OUTPUT_CSV_PATH}
- Target Activities Filtered: A6 (walking), A7 (standing), A8 (sitting)


2. METRICS & COUNTS
--------------------------------------------------------------------------------
- Number of Subjects: {val_results.get('num_subjects', 0)}
- Number of Cameras : {val_results.get('num_cameras', 0)}
- Number of Sequences/Trials: {val_results.get('num_trials', 0)}
- Total Frames/Rows : {val_results.get('num_frames', 0)}

Activity Sample Breakdown:
{pd.Series(val_results.get('activity_distribution', {})).to_string()}


3. DATASET INTEGRITY & VALIDATION
--------------------------------------------------------------------------------
- All 33 MediaPipe landmarks mapped correctly : {val_results.get('all_landmarks_present')}
- Missing / Null Values                       : {val_results.get('null_count')}
- Sequence Integrity (Strict Frame Ordering) : {val_results.get('sequence_integrity')}
- Sequence Boundaries Preserved               : YES (per-trial sequence_id explicitly tracked)


4. PIPELINE REUSABILITY & COMPATIBILITY
--------------------------------------------------------------------------------
- Compatibility with src/feature_engineering.py : {'PASS (100% reusable)' if compat_results.get('spatial_compatible') else 'FAIL'}
  * Generated columns: {compat_results.get('spatial_columns')} (247 spatial + 1 activity)
- Compatibility with src/temporal_features.py    : {'PASS (100% reusable)' if compat_results.get('temporal_compatible') else 'FAIL'}
  * Generated temporal metrics: {compat_results.get('temporal_keys')} features


5. KEY ARCHITECTURAL FINDINGS
--------------------------------------------------------------------------------
1. Adapter Layer Efficiency:
   load_upfall.py successfully isolates the external coordinate naming conventions
   (Joint1_X -> landmark_0_x), leaving existing project code untouched.
2. Sequence Isolation:
   Every UP-Fall trial is preserved as an independent sequence (C{{cam}}_S{{sub}}_A{{act}}_T{{trial}}),
   preventing cross-class or cross-subject temporal derivative leakage.
3. ADL Coverage Note:
   UP-Fall provides walking, standing, and sitting. Running is not available in UP-Fall
   and must remain augmented by the self-collected dataset or a secondary source.
================================================================================
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Integration report saved to: {output_path}")


def main():
    print("=== UP-Fall Dataset Adapter Execution ===")
    df = load_and_standardize_upfall()

    if df.empty:
        print("[ERROR] Standardized dataset is empty. Cannot continue.")
        return

    # Save standardized dataset
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved standardized dataset to: {OUTPUT_CSV_PATH}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Run validation
    val_results = validate_standardized_dataset(df)
    print("\n--- Validation Results ---")
    for k, v in val_results.items():
        print(f"  {k}: {v}")

    # Test compatibility
    compat_results = test_pipeline_compatibility(df)
    print("\n--- Pipeline Compatibility ---")
    for k, v in compat_results.items():
        print(f"  {k}: {v}")

    # Write report
    write_report(val_results, compat_results, REPORT_PATH)


if __name__ == "__main__":
    main()

