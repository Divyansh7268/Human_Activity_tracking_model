
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from feature_engineering import engineer_features
from temporal_features import (
    compute_sequence_temporal_features,
    extract_3d_coordinates,
)

# 1. Load the UP-Fall pose dataset 
print("Loading upfall_pose_dataset.csv …")
pose_df = pd.read_csv("data/processed/upfall_pose_dataset.csv")
print(f"  Loaded {len(pose_df)} rows, {pose_df.shape[1]} columns")

# Metadata columns that are not landmarks
META_COLS = ["sequence_id", "subject_id", "camera_id", "activity_code", "trial_id", "frame_id"]

# 2. Generate spatial features 
print("\nGenerating spatial features …")

# Strip metadata before passing to engineer_features (keep 'activity')
landmark_df = pose_df.drop(columns=META_COLS, errors="ignore")

features_df = engineer_features(landmark_df)
print(f"  Output shape: {features_df.shape}")
print(f"  Feature count: {features_df.shape[1]-1} (excluding activity)")
print(f"  Missing values: {features_df.isnull().sum().sum()}")
print(f"  Activity counts:\n{features_df['activity'].value_counts()}")

features_df.to_csv("data/processed/upfall_features.csv", index=False)
print("  Saved → data/processed/upfall_features.csv")

#3. Generate temporal features 
print("\nGenerating temporal features (respecting sequence boundaries) …")

# Re-attach sequence_id and original row indices for grouping
pose_df_indexed = pose_df.reset_index(drop=True)
features_df_indexed = features_df.reset_index(drop=True)

sequences = pose_df_indexed["sequence_id"].unique()
print(f"  Found {len(sequences)} sequences: {sorted(sequences)}")

temporal_chunks = []

for seq_id in sorted(sequences):
    seq_mask = pose_df_indexed["sequence_id"] == seq_id
    seq_pose = pose_df_indexed[seq_mask].reset_index(drop=True)
    seq_features = features_df_indexed[seq_mask].reset_index(drop=True)

    activity_label = seq_pose["activity"].iloc[0]
    n_frames = len(seq_pose)

    if n_frames < 2:
        print(f"  WARNING: sequence {seq_id} has only {n_frames} frame(s) — skipping.")
        continue

    # Extract (N, 33, 3) coordinate array for temporal computation
    coords = extract_3d_coordinates(seq_pose)  # uses landmark_N_x/y/z columns

    temp_dict = compute_sequence_temporal_features(coords)  # returns dict of arrays length N-1

    if not temp_dict:
        print(f"  WARNING: sequence {seq_id} produced no temporal features — skipping.")
        continue

    n_temporal = len(next(iter(temp_dict.values())))  # N-1
    temporal_rows = pd.DataFrame(temp_dict)

    # Spatial features for frames [1:] (skip first frame — no prior frame for displacement)
    spatial_slice = seq_features.iloc[1:].drop(columns=["activity"]).reset_index(drop=True)
    activity_series = pd.Series([activity_label] * n_temporal, name="activity")

    # Combined: [247 spatial features] + [8 temporal features] + [activity]
    combined = pd.concat([spatial_slice, temporal_rows, activity_series], axis=1)
    temporal_chunks.append(combined)

    print(f"  Sequence {seq_id} ({activity_label}): {n_frames} frames → {n_temporal} temporal rows")

temporal_df = pd.concat(temporal_chunks, ignore_index=True)

print(f"\n  Output shape: {temporal_df.shape}")
print(f"  Missing values: {temporal_df.isnull().sum().sum()}")
print(f"  Activity counts:\n{temporal_df['activity'].value_counts()}")

temporal_df.to_csv("data/processed/upfall_temporal_features.csv", index=False)
print("  Saved → data/processed/upfall_temporal_features.csv")

print("\nDone — both feature files generated.")
