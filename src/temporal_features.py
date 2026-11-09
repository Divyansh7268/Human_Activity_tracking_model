"""
Temporal Feature Engineering

Calculates frame-to-frame movement dynamics from consecutive pose frames
to distinguish static activities (standing, sitting) from dynamic activities (walking, running).

Temporal Features Generated:
  - mean_landmark_displacement : Mean Euclidean displacement across all 33 landmarks
  - max_landmark_displacement  : Maximum displacement observed among landmarks
  - upper_body_movement        : Mean displacement for head, shoulders, arms, hands (landmarks 0-22)
  - lower_body_movement        : Mean displacement for hips, knees, ankles, feet (landmarks 23-32)
  - left_side_movement         : Mean displacement of left-side body landmarks
  - right_side_movement        : Mean displacement of right-side body landmarks
  - movement_speed             : Displacement divided by sampling time interval (dt ~ 0.15s)
  - movement_acceleration      : Magnitude of change in speed divided by dt

Methodology:
  - Segments raw data into continuous activity recording blocks.
  - Computes motion dynamics strictly WITHIN continuous blocks.
  - Avoids artificial jumps across class transitions by omitting the boundary frame.
  - Combines all 247 baseline single-frame features with the new 8 temporal features.

Outputs:
  - data/processed/temporal_features.csv
  - results/temporal_movement_distribution.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

RAW_DATA_PATH = os.path.join("data", "raw", "pose_dataset.csv")
FEATURES_DATA_PATH = os.path.join("data", "processed", "features.csv")
OUTPUT_TEMPORAL_PATH = os.path.join("data", "processed", "temporal_features.csv")
CHART_PATH = os.path.join("results", "temporal_movement_distribution.png")

# Sampling interval in seconds (from collect_pose_data.py SAMPLE_INTERVAL_MS = 150)
SAMPLING_INTERVAL_S = 0.15
NUM_LANDMARKS = 33

# Anatomical landmark groupings (MediaPipe 33-landmark schema)
UPPER_BODY_INDICES = list(range(23))  # 0 to 22: head, chest, arms, hands
LOWER_BODY_INDICES = list(range(23, 33))  # 23 to 32: hips, knees, ankles, feet
LEFT_SIDE_INDICES = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RIGHT_SIDE_INDICES = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]


def extract_3d_coordinates(df_subset: pd.DataFrame) -> np.ndarray:
    """Extract 3D coordinates for all 33 landmarks as array of shape (N, 33, 3)."""
    coords = []
    for i in range(NUM_LANDMARKS):
        cols = [f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"]
        coords.append(df_subset[cols].to_numpy())
    return np.stack(coords, axis=1)


def compute_sequence_temporal_features(coords: np.ndarray, dt: float = SAMPLING_INTERVAL_S) -> dict:
    """
    Compute frame-to-frame movement metrics for a sequence of 3D coordinates.
    Input shape: (N, 33, 3).
    Returns dict of feature arrays of length N-1 (frame t vs frame t-1).
    """
    if len(coords) < 2:
        return {}

    # Frame-to-frame displacement per landmark: shape (N-1, 33)
    disp = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)

    mean_disp = disp.mean(axis=-1)
    max_disp = disp.max(axis=-1)
    upper_disp = disp[:, UPPER_BODY_INDICES].mean(axis=-1)
    lower_disp = disp[:, LOWER_BODY_INDICES].mean(axis=-1)
    left_disp = disp[:, LEFT_SIDE_INDICES].mean(axis=-1)
    right_disp = disp[:, RIGHT_SIDE_INDICES].mean(axis=-1)
    speed = mean_disp / dt

    # Acceleration: rate of change in speed
    accel = np.zeros_like(speed)
    if len(speed) > 1:
        accel[1:] = np.abs(speed[1:] - speed[:-1]) / dt

    return {
        "mean_landmark_displacement": mean_disp,
        "max_landmark_displacement": max_disp,
        "upper_body_movement": upper_disp,
        "lower_body_movement": lower_disp,
        "left_side_movement": left_disp,
        "right_side_movement": right_disp,
        "movement_speed": speed,
        "movement_acceleration": accel,
    }


def compute_live_temporal_features(
    current_coords: np.ndarray,
    previous_coords: np.ndarray,
    previous_speed: float = 0.0,
    dt: float = SAMPLING_INTERVAL_S,
) -> dict:
    """
    Helper for future real-time streaming inference:
    Given two consecutive (33, 3) landmark arrays, calculates instant movement features.
    """
    disp = np.linalg.norm(current_coords - previous_coords, axis=-1)  # (33,)
    mean_disp = float(disp.mean())
    max_disp = float(disp.max())
    upper_disp = float(disp[UPPER_BODY_INDICES].mean())
    lower_disp = float(disp[LOWER_BODY_INDICES].mean())
    left_disp = float(disp[LEFT_SIDE_INDICES].mean())
    right_disp = float(disp[RIGHT_SIDE_INDICES].mean())
    speed = float(mean_disp / dt)
    accel = float(abs(speed - previous_speed) / dt)

    return {
        "mean_landmark_displacement": mean_disp,
        "max_landmark_displacement": max_disp,
        "upper_body_movement": upper_disp,
        "lower_body_movement": lower_disp,
        "left_side_movement": left_disp,
        "right_side_movement": right_disp,
        "movement_speed": speed,
        "movement_acceleration": accel,
    }


def build_temporal_dataset():
    """Build and save temporal_features.csv."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset missing: {RAW_DATA_PATH}")
    if not os.path.exists(FEATURES_DATA_PATH):
        raise FileNotFoundError(f"Processed single-frame features missing: {FEATURES_DATA_PATH}")

    raw_df = pd.read_csv(RAW_DATA_PATH)
    features_df = pd.read_csv(FEATURES_DATA_PATH)

    print(f"Loaded raw dataset      : {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"Loaded single-frame data: {features_df.shape[0]} rows, {features_df.shape[1]} columns")

    # Detect contiguous activity blocks (recording sequences)
    activity_change = (raw_df["activity"] != raw_df["activity"].shift()).cumsum()
    block_groups = raw_df.groupby(activity_change)
    print(f"Detected {len(block_groups)} contiguous sequence block(s) in dataset.")

    processed_blocks = []

    for block_id, group in block_groups:
        act = group["activity"].iloc[0]
        n_rows = len(group)
        indices = group.index

        if n_rows < 2:
            print(f"Skipping block {block_id} ({act}) — fewer than 2 frames.")
            continue

        coords = extract_3d_coordinates(group)
        temp_dict = compute_sequence_temporal_features(coords, dt=SAMPLING_INTERVAL_S)

        # Create temporal dataframe matching indices[1:] (omitting first frame)
        temp_df = pd.DataFrame(temp_dict, index=indices[1:])

        # Single-frame features for indices[1:]
        single_frame_slice = features_df.loc[indices[1:]].drop(columns=["activity"])
        activity_label = features_df.loc[indices[1:], "activity"]

        # Concatenate: [Single-frame 247 features] + [Temporal 8 features] + [activity]
        block_combined = pd.concat([single_frame_slice, temp_df, activity_label], axis=1)
        processed_blocks.append(block_combined)

    final_df = pd.concat(processed_blocks, axis=0)

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_TEMPORAL_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_TEMPORAL_PATH, index=False)
    print(f"\nSuccessfully generated: {OUTPUT_TEMPORAL_PATH}")
    print(f"Final shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")

    return final_df


def generate_visualization(df: pd.DataFrame, output_path: str):
    """Create a publication-quality comparison figure of movement speed and lower body movement."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    activities = ["standing", "sitting", "walking", "running"]
    colors = ["#3498db", "#95a5a6", "#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Movement Speed Distribution (Boxplot)
    speed_data = [df[df["activity"] == act]["movement_speed"].values for act in activities]
    bp1 = axes[0].boxplot(speed_data, tick_labels=[a.capitalize() for a in activities], patch_artist=True, showmeans=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Movement Speed by Activity", fontsize=13, pad=10)
    axes[0].set_ylabel("Speed (Normalized coordinate units / sec)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Lower Body Movement Distribution (Boxplot)
    lower_data = [df[df["activity"] == act]["lower_body_movement"].values for act in activities]
    bp2 = axes[1].boxplot(lower_data, tick_labels=[a.capitalize() for a in activities], patch_artist=True, showmeans=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Lower-Body Movement by Activity", fontsize=13, pad=10)
    axes[1].set_ylabel("Displacement per Frame (3D Euclidean)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Temporal Movement Feature Comparison Across Human Activities", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal distribution chart: {output_path}")


def print_comparison_stats(df: pd.DataFrame):
    """Print detailed comparative statistics, focusing on Standing vs Walking."""
    print("\n" + "=" * 65)
    print("ACTIVITY DISTRIBUTION:")
    print(df["activity"].value_counts().to_string())

    temporal_cols = [
        "mean_landmark_displacement",
        "max_landmark_displacement",
        "upper_body_movement",
        "lower_body_movement",
        "left_side_movement",
        "right_side_movement",
        "movement_speed",
        "movement_acceleration",
    ]

    print("\n" + "=" * 65)
    print("TEMPORAL FEATURE MEANS ACROSS ALL ACTIVITIES:")
    summary = df.groupby("activity")[temporal_cols].mean().round(4)
    print(summary.to_string())

    print("\n" + "=" * 65)
    print("STANDING VS. WALKING STATISTICAL COMPARISON:")
    standing = df[df["activity"] == "standing"]
    walking = df[df["activity"] == "walking"]

    comparison_records = []
    for col in temporal_cols:
        m_s, s_s = standing[col].mean(), standing[col].std()
        m_w, s_w = walking[col].mean(), walking[col].std()
        ratio = m_w / m_s if m_s > 0 else 0
        pooled_std = np.sqrt((s_s**2 + s_w**2) / 2)
        cohen_d = abs(m_s - m_w) / pooled_std if pooled_std > 0 else 0

        comparison_records.append({
            "Feature": col,
            "Standing Mean (Std)": f"{m_s:.4f} (±{s_s:.4f})",
            "Walking Mean (Std)": f"{m_w:.4f} (±{s_w:.4f})",
            "Walking/Standing Ratio": f"{ratio:.2f}x",
            "Cohen's d": f"{cohen_d:.3f}",
        })

    comp_df = pd.DataFrame(comparison_records)
    print(comp_df.to_string(index=False))


def main():
    final_df = build_temporal_dataset()
    generate_visualization(final_df, CHART_PATH)
    print_comparison_stats(final_df)


if __name__ == "__main__":
    main()

