
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from normalized_feature_engineering import (
    engineer_normalized_spatial_features,
    compute_normalized_temporal_features,
    calculate_distance,
    get_point,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)
from temporal_features import extract_3d_coordinates

RAW_POSE_PATH = "data/processed/upfall_pose_dataset.csv"
OUTPUT_PATH = "data/processed/upfall_normalized_temporal_features.csv"

META_COLS = ["sequence_id", "subject_id", "camera_id", "activity_code", "trial_id", "frame_id"]


def main():
    print(f"Loading raw UP-Fall pose dataset: {RAW_POSE_PATH} ...")
    pose_df = pd.read_csv(RAW_POSE_PATH)
    print(f"Loaded shape: {pose_df.shape}")

    # 1. Compute normalized spatial features across all frames
    print("\nExtracting normalized spatial features (translation- & scale-invariant) ...")
    clean_pose_df = pose_df.drop(columns=META_COLS, errors="ignore")
    spatial_df = engineer_normalized_spatial_features(clean_pose_df)
    print(f"Normalized spatial features shape: {spatial_df.shape}")

    # Retain sequence_id for trial-segmented temporal computation
    spatial_df["sequence_id"] = pose_df["sequence_id"].values

    # 2. Extract torso lengths per frame for scale normalization of temporal dynamics
    left_sh = get_point(pose_df, LEFT_SHOULDER)
    right_sh = get_point(pose_df, RIGHT_SHOULDER)
    left_hip = get_point(pose_df, LEFT_HIP)
    right_hip = get_point(pose_df, RIGHT_HIP)
    sh_mid = (left_sh + right_sh) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    all_torso_lengths = calculate_distance(sh_mid, hip_mid)

    # 3. Compute temporal dynamics strictly within each sequence
    sequences = sorted(pose_df["sequence_id"].unique())
    print(f"\nProcessing {len(sequences)} sequence(s) for temporal features:")

    chunks = []
    for seq_id in sequences:
        seq_mask = pose_df["sequence_id"] == seq_id
        seq_pose = pose_df[seq_mask].reset_index(drop=True)
        seq_spatial = spatial_df[seq_mask].reset_index(drop=True)
        seq_torso = all_torso_lengths[seq_mask]
        activity_label = seq_pose["activity"].iloc[0]

        if len(seq_pose) < 2:
            print(f"  Sequence {seq_id}: <2 frames, skipping.")
            continue

        coords_raw = extract_3d_coordinates(seq_pose)
        temp_dict = compute_normalized_temporal_features(coords_raw, seq_torso)

        temp_df = pd.DataFrame(temp_dict)
        # Drop frame 0 of spatial features (no previous frame for displacement)
        spatial_slice = seq_spatial.iloc[1:].drop(columns=["activity", "sequence_id"]).reset_index(drop=True)
        act_series = pd.Series([activity_label] * len(temp_df), name="activity")

        combined_seq = pd.concat([spatial_slice, temp_df, act_series], axis=1)
        chunks.append(combined_seq)
        print(f"  Sequence {seq_id} ({activity_label:10s}): {len(seq_pose)} frames -> {len(temp_df)} temporal rows")

    final_df = pd.concat(chunks, ignore_index=True)

    # 4. Verification
    print("\n" + "=" * 60)
    print("VERIFICATION OF GENERATED DATASET:")
    print("=" * 60)
    print(f"Output path         : {OUTPUT_PATH}")
    print(f"Row count           : {len(final_df)}")
    print(f"Column count        : {final_df.shape[1]}")
    print(f"Feature count       : {final_df.shape[1] - 1} (excluding activity)")
    print(f"Missing values      : {final_df.isnull().sum().sum()}")
    print(f"Infinite values     : {np.isinf(final_df.drop(columns=['activity']).to_numpy()).sum()}")
    print("Class distribution:")
    print(final_df["activity"].value_counts().to_string())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSuccessfully saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

