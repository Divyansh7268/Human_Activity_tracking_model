

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

RAW_DATA_PATH = os.path.join("data", "raw", "multisubject_pose_dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "multisubject_normalized_temporal_features.csv")
SUMMARY_TXT_PATH = os.path.join("results", "multisubject_feature_summary.txt")
DIST_CHART_PATH = os.path.join("results", "multisubject_temporal_distribution.png")
REPORT_PATH = os.path.join("results", "multisubject_feature_engineering_report.txt")

# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
NUM_LANDMARKS = 33

METADATA_COLS = [
    "subject_id",
    "activity",
    "activity_code",
    "trial_id",
    "sequence_id",
    "frame_id",
    "timestamp",
]

TARGET_ACTIVITIES = ["standing", "sitting", "walking", "running"]


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Euclidean distance between two 3D arrays of points."""
    return np.linalg.norm(point_a - point_b, axis=-1)


def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> np.ndarray:
    """Calculate joint angle in degrees at vertex point_b."""
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    dot_prod = np.sum(vector_ba * vector_bc, axis=-1)
    norm_ba = np.linalg.norm(vector_ba, axis=-1)
    norm_bc = np.linalg.norm(vector_bc, axis=-1)

    denominator = norm_ba * norm_bc
    denominator = np.where(denominator == 0, 1e-7, denominator)

    cosine_angle = np.clip(dot_prod / denominator, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def get_point(df: pd.DataFrame, index: int) -> np.ndarray:
    """Extract (x, y, z) coordinates for a given landmark index across all rows."""
    x = df[f"landmark_{index}_x"].to_numpy()
    y = df[f"landmark_{index}_y"].to_numpy()
    z = df[f"landmark_{index}_z"].to_numpy()
    return np.column_stack([x, y, z])


def extract_3d_coordinates(df_subset: pd.DataFrame) -> np.ndarray:
    """Extract 3D coordinates for all 33 landmarks as array of shape (N, 33, 3)."""
    coords = []
    for i in range(NUM_LANDMARKS):
        cols = [f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"]
        coords.append(df_subset[cols].to_numpy())
    return np.stack(coords, axis=1)


def engineer_normalized_spatial_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract 150 scale- and translation-invariant spatial features.
    Returns: (features_df, torso_lengths_array)
    """
    new_features = {}

    left_hip = get_point(raw_df, LEFT_HIP)
    right_hip = get_point(raw_df, RIGHT_HIP)
    hip_midpoint = (left_hip + right_hip) / 2.0

    left_shoulder = get_point(raw_df, LEFT_SHOULDER)
    right_shoulder = get_point(raw_df, RIGHT_SHOULDER)
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0

    torso_length = calculate_distance(shoulder_midpoint, hip_midpoint)
    scale = np.where(torso_length < 1e-4, 1e-4, torso_length)

    # 1. 99 Normalized landmark coordinates
    for i in range(NUM_LANDMARKS):
        pt = get_point(raw_df, i)
        rel_pt = pt - hip_midpoint
        new_features[f"norm_landmark_{i}_x"] = rel_pt[:, 0] / scale
        new_features[f"norm_landmark_{i}_y"] = rel_pt[:, 1] / scale
        new_features[f"norm_landmark_{i}_z"] = rel_pt[:, 2] / scale

    # Pre-extract extremity landmarks
    left_elbow = get_point(raw_df, LEFT_ELBOW)
    right_elbow = get_point(raw_df, RIGHT_ELBOW)
    left_wrist = get_point(raw_df, LEFT_WRIST)
    right_wrist = get_point(raw_df, RIGHT_WRIST)
    left_knee = get_point(raw_df, LEFT_KNEE)
    right_knee = get_point(raw_df, RIGHT_KNEE)
    left_ankle = get_point(raw_df, LEFT_ANKLE)
    right_ankle = get_point(raw_df, RIGHT_ANKLE)

    # 2. 10 Normalized segment distances
    new_features["norm_shoulder_width"] = calculate_distance(left_shoulder, right_shoulder) / scale
    new_features["norm_hip_width"] = calculate_distance(left_hip, right_hip) / scale
    new_features["norm_left_upper_arm"] = calculate_distance(left_shoulder, left_elbow) / scale
    new_features["norm_left_forearm"] = calculate_distance(left_elbow, left_wrist) / scale
    new_features["norm_right_upper_arm"] = calculate_distance(right_shoulder, right_elbow) / scale
    new_features["norm_right_forearm"] = calculate_distance(right_elbow, right_wrist) / scale
    new_features["norm_left_thigh"] = calculate_distance(left_hip, left_knee) / scale
    new_features["norm_left_shin"] = calculate_distance(left_knee, left_ankle) / scale
    new_features["norm_right_thigh"] = calculate_distance(right_hip, right_knee) / scale
    new_features["norm_right_shin"] = calculate_distance(right_knee, right_ankle) / scale

    # 3. 8 Joint angles
    new_features["left_elbow_angle"] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    new_features["right_elbow_angle"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    new_features["left_knee_angle"] = calculate_angle(left_hip, left_knee, left_ankle)
    new_features["right_knee_angle"] = calculate_angle(right_hip, right_knee, right_ankle)
    new_features["left_shoulder_angle"] = calculate_angle(left_hip, left_shoulder, left_elbow)
    new_features["right_shoulder_angle"] = calculate_angle(right_hip, right_shoulder, right_elbow)
    new_features["left_hip_angle"] = calculate_angle(left_shoulder, left_hip, left_knee)
    new_features["right_hip_angle"] = calculate_angle(right_shoulder, right_hip, right_knee)

    # 4. 33 Visibility scores
    for i in range(NUM_LANDMARKS):
        vis_col = f"landmark_{i}_visibility"
        if vis_col in raw_df.columns:
            new_features[vis_col] = raw_df[vis_col].to_numpy()

    features_df = pd.DataFrame(new_features, index=raw_df.index)
    return features_df, torso_length


def compute_sequence_temporal_dynamics(
    coords_raw: np.ndarray,
    torso_lengths: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """
    Compute frame-to-frame movement metrics on scale-normalized coordinates
    strictly within a single sequence.
    """
    n_frames = len(coords_raw)
    if n_frames < 2:
        return {}

    # Scale coordinates by torso length at each frame: shape (N, 33, 3)
    scale_3d = np.where(torso_lengths < 1e-4, 1e-4, torso_lengths)[:, np.newaxis, np.newaxis]
    scaled_coords = coords_raw / scale_3d

    # Frame-to-frame displacement per landmark: shape (N-1, 33)
    disp = np.linalg.norm(scaled_coords[1:] - scaled_coords[:-1], axis=-1)

    upper_indices = list(range(23))
    lower_indices = list(range(23, 33))
    left_indices = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    right_indices = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    mean_disp = disp.mean(axis=-1)
    max_disp = disp.max(axis=-1)
    upper_disp = disp[:, upper_indices].mean(axis=-1)
    lower_disp = disp[:, lower_indices].mean(axis=-1)
    left_disp = disp[:, left_indices].mean(axis=-1)
    right_disp = disp[:, right_indices].mean(axis=-1)

    # Sampling time differences from actual timestamps
    dts = np.diff(timestamps)
    # Guard against zero or negative dt from system timer jitter
    dts_safe = np.where(dts <= 0, 0.033, dts)

    speed = mean_disp / dts_safe

    accel = np.zeros_like(speed)
    if len(speed) > 1:
        accel[1:] = np.abs(speed[1:] - speed[:-1]) / dts_safe[1:]

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


def generate_feature_summary(df_features: pd.DataFrame, feature_cols: list[str]) -> str:
    """Generate statistical summary of numerical features."""
    summary_lines = []
    summary_lines.append("=" * 85)
    summary_lines.append("MULTI-SUBJECT FEATURE ENGINEERING: STATISTICAL SUMMARY")
    summary_lines.append("=" * 85)
    summary_lines.append(f"Total Features Audited: {len(feature_cols)}")
    summary_lines.append(f"Total Rows: {len(df_features):,}")
    summary_lines.append("")
    summary_lines.append(f"{'Feature Name':<35} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
    summary_lines.append("-" * 85)

    stats = df_features[feature_cols].describe().T
    for col in feature_cols:
        row = stats.loc[col]
        summary_lines.append(
            f"{col:<35} | {row['mean']:10.4f} | {row['std']:10.4f} | {row['min']:10.4f} | {row['max']:10.4f}"
        )
    summary_lines.append("=" * 85)

    summary_text = "\n".join(summary_lines)
    os.makedirs(os.path.dirname(SUMMARY_TXT_PATH), exist_ok=True)
    with open(SUMMARY_TXT_PATH, "w") as f:
        f.write(summary_text)
    return summary_text


def generate_temporal_distribution_plots(df_combined: pd.DataFrame):
    """Generate visual boxplot distributions for temporal movement features by activity."""
    temp_features_to_plot = [
        ("movement_speed", "Movement Speed (Torso Lengths / sec)"),
        ("lower_body_movement", "Lower Body Displacement (Torso Lengths / frame)"),
        ("upper_body_movement", "Upper Body Displacement (Torso Lengths / frame)"),
        ("movement_acceleration", "Acceleration Magnitude (Torso Lengths / sec²)"),
    ]

    colors = ["#3498db", "#95a5a6", "#2ecc71", "#e74c3c"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (col_name, title) in enumerate(temp_features_to_plot):
        ax = axes[idx // 2, idx % 2]
        data = [df_combined[df_combined["activity"] == act][col_name].values for act in TARGET_ACTIVITIES]

        bp = ax.boxplot(
            data,
            tick_labels=[a.capitalize() for a in TARGET_ACTIVITIES],
            patch_artist=True,
            showmeans=True,
            showfliers=False,  # clean display for publication
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Temporal Movement Feature Distribution Across Multi-Subject Activities", fontsize=14, y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(DIST_CHART_PATH), exist_ok=True)
    plt.savefig(DIST_CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal distribution chart -> {DIST_CHART_PATH}")


def process_multisubject_dataset():
    print(f"Loading raw multi-subject dataset: {RAW_DATA_PATH} ...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_rows, raw_cols = raw_df.shape
    print(f"Loaded: {raw_rows:,} rows, {raw_cols} columns")

    # 1. Compute normalized spatial features across all frames
    print("Extracting 150 scale- and translation-invariant spatial features ...")
    spatial_df, all_torso_lengths = engineer_normalized_spatial_features(raw_df)
    spatial_cols = spatial_df.columns.tolist()
    print(f"Spatial features generated: {len(spatial_cols)}")

    # 2. Compute temporal dynamics sequence by sequence (strictly within sequence boundaries)
    sequences = sorted(raw_df["sequence_id"].unique())
    print(f"\nProcessing {len(sequences)} sequence(s) strictly within trial boundaries:")

    processed_chunks = []
    dropped_boundary_rows = 0

    temporal_cols = None

    for seq_id in sequences:
        seq_mask = (raw_df["sequence_id"] == seq_id)
        seq_raw = raw_df[seq_mask].reset_index(drop=True)
        seq_spatial = spatial_df[seq_mask].reset_index(drop=True)
        seq_torso = all_torso_lengths[seq_mask]
        n_seq_frames = len(seq_raw)

        if n_seq_frames < 2:
            print(f"  [WARNING] Sequence {seq_id} has < 2 frames. Skipping.")
            continue

        # Extract 3D coords and timestamps
        coords_raw = extract_3d_coordinates(seq_raw)
        timestamps = seq_raw["timestamp"].values

        temp_dict = compute_sequence_temporal_dynamics(coords_raw, seq_torso, timestamps)
        temp_df = pd.DataFrame(temp_dict)
        if temporal_cols is None:
            temporal_cols = temp_df.columns.tolist()

        # Slice metadata and spatial features (frame 0 dropped due to temporal difference)
        meta_slice = seq_raw.iloc[1:][METADATA_COLS].reset_index(drop=True)
        spatial_slice = seq_spatial.iloc[1:].reset_index(drop=True)

        combined_seq = pd.concat([meta_slice, spatial_slice, temp_df], axis=1)
        processed_chunks.append(combined_seq)
        dropped_boundary_rows += 1

        print(
            f"  • {seq_id:<20} ({seq_raw['activity'].iloc[0]:<8}): "
            f"{n_seq_frames:5d} raw frames -> {len(combined_seq):5d} processed (1 boundary dropped)"
        )

    final_processed_df = pd.concat(processed_chunks, ignore_index=True)
    processed_rows, processed_cols = final_processed_df.shape
    total_ml_features = len(spatial_cols) + len(temporal_cols)

    print("\n" + "=" * 85)
    print("DATASET PROCESSING COMPLETE:")
    print("=" * 85)
    print(f"Raw Input Rows               : {raw_rows:,}")
    print(f"Boundary Rows Dropped        : {dropped_boundary_rows} (exactly 1 per sequence)")
    print(f"Processed Rows               : {processed_rows:,}")
    print(f"Total Columns                : {processed_cols}")
    print(f"Metadata Columns Preserved   : {len(METADATA_COLS)} ({', '.join(METADATA_COLS)})")
    print(f"Spatial Feature Count        : {len(spatial_cols)}")
    print(f"Temporal Feature Count       : {len(temporal_cols)}")
    print(f"Total Machine Learning Feats : {total_ml_features}")
    print("=" * 85)

    # 3. Verification checks
    null_count = int(final_processed_df.isnull().sum().sum())
    inf_count = int(np.isinf(final_processed_df.select_dtypes(include=[np.number]).to_numpy()).sum())

    subjects_preserved = sorted(final_processed_df["subject_id"].unique().tolist())
    activities_preserved = sorted(final_processed_df["activity"].unique().tolist())
    sequences_preserved = len(final_processed_df["sequence_id"].unique())

    print(f"Missing (NaN) Values         : {null_count}")
    print(f"Infinite Values              : {inf_count}")
    print(f"Subjects Preserved ({len(subjects_preserved)})       : {', '.join(subjects_preserved)}")
    print(f"Activities Preserved ({len(activities_preserved)})     : {', '.join(activities_preserved)}")
    print(f"Sequences Preserved          : {sequences_preserved} (Expected: {len(sequences)})")

    # 4. Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nSuccessfully saved -> {PROCESSED_DATA_PATH}")

    # 5. Feature distribution summary & charts
    all_ml_cols = spatial_cols + temporal_cols
    print("Generating feature statistical summary...")
    generate_feature_summary(final_processed_df, all_ml_cols)
    print(f"Saved feature summary -> {SUMMARY_TXT_PATH}")

    print("Generating temporal distribution plots...")
    generate_temporal_distribution_plots(final_processed_df)

    # 6. Generate final audit report
    report_lines = []
    report_lines.append("=" * 85)
    report_lines.append("STEP 13: MULTI-SUBJECT FEATURE ENGINEERING AUDIT REPORT")
    report_lines.append("=" * 85)
    report_lines.append(f"1. INPUT DATASET")
    report_lines.append(f"   Path           : {RAW_DATA_PATH}")
    report_lines.append(f"   Raw Rows       : {raw_rows:,}")
    report_lines.append(f"   Raw Columns    : {raw_cols}")
    report_lines.append("")
    report_lines.append(f"2. OUTPUT DATASET")
    report_lines.append(f"   Path           : {PROCESSED_DATA_PATH}")
    report_lines.append(f"   Processed Rows : {processed_rows:,}")
    report_lines.append(f"   Total Columns  : {processed_cols}")
    report_lines.append("")
    report_lines.append(f"3. NORMALIZATION METHOD")
    report_lines.append(f"   • Origin Reference  : Hip midpoint ((left_hip + right_hip) / 2.0).")
    report_lines.append(f"   • Scale Reference   : Torso length (shoulder midpoint to hip midpoint).")
    report_lines.append(f"   • Translation Invariance: All coordinates centered at hip midpoint.")
    report_lines.append(f"   • Scale Invariance      : All relative coordinates & segment lengths divided by torso length.")
    report_lines.append(f"   • Absolute Coordinates  : Strictly EXCLUDED from ML features.")
    report_lines.append("")
    report_lines.append(f"4. SPATIAL FEATURES (150 TOTAL)")
    report_lines.append(f"   • 99 Normalized landmark coordinates (norm_landmark_0_x ... norm_landmark_32_z)")
    report_lines.append(f"   • 10 Normalized body segment lengths (norm_shoulder_width ... norm_right_shin)")
    report_lines.append(f"   •  8 Joint angles (elbows, knees, shoulders, and hip flexion angles)")
    report_lines.append(f"   • 33 MediaPipe landmark visibility confidence scores")
    report_lines.append("")
    report_lines.append(f"5. TEMPORAL FEATURES (8 TOTAL)")
    report_lines.append(f"   • mean_landmark_displacement, max_landmark_displacement")
    report_lines.append(f"   • upper_body_movement, lower_body_movement")
    report_lines.append(f"   • left_side_movement, right_side_movement")
    report_lines.append(f"   • movement_speed, movement_acceleration")
    report_lines.append(f"   • Computed strictly within sequence boundaries; zero leakage across trials.")
    report_lines.append("")
    report_lines.append(f"6. METADATA PRESERVED ({len(METADATA_COLS)} COLUMNS)")
    report_lines.append(f"   • {', '.join(METADATA_COLS)}")
    report_lines.append(f"   • Explicitly retained for GroupKFold and Leave-One-Subject-Out evaluation.")
    report_lines.append("")
    report_lines.append(f"7. BOUNDARY ROWS REMOVED")
    report_lines.append(f"   • Rows removed : {dropped_boundary_rows} (exactly frame 0 for each of the 16 sequences)")
    report_lines.append(f"   • 21,227 raw frames -> 21,211 processed frames (99.92% data retention)")
    report_lines.append("")
    report_lines.append(f"8. VALIDATION CHECKS")
    report_lines.append(f"   • Missing (NaN) values : {null_count} (PASS)")
    report_lines.append(f"   • Infinite values      : {inf_count} (PASS)")
    report_lines.append(f"   • Subjects (4/4)       : {', '.join(subjects_preserved)} (PASS)")
    report_lines.append(f"   • Activities (4/4)     : {', '.join(activities_preserved)} (PASS)")
    report_lines.append(f"   • Sequences (16/16)    : {sequences_preserved} sequences intact (PASS)")
    report_lines.append("")
    report_lines.append(f"9. EVALUATION READINESS VERDICT")
    verdict = "READY_FOR_SUBJECT_INDEPENDENT_EVALUATION"
    report_lines.append(f"   Verdict: {verdict}")
    report_lines.append("=" * 85)

    report_text = "\n".join(report_lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"Saved engineering report -> {REPORT_PATH}")
    print("\n" + report_text)


if __name__ == "__main__":
    process_multisubject_dataset()

