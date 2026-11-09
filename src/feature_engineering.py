
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW_DATA_PATH = os.path.join("data", "raw", "pose_dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "features.csv")
CHART_PATH = os.path.join("results", "activity_distribution.png")

# MediaPipe landmark indices used for distances and angles
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


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance between two 3D points or arrays of points."""
    return np.linalg.norm(point_a - point_b, axis=-1)


def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> np.ndarray:
    """
    Calculate the joint angle (in degrees) at vertex point_b formed by (point_a - point_b)
    and (point_c - point_b).
    """
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    # Dot product and magnitudes
    dot_prod = np.sum(vector_ba * vector_bc, axis=-1)
    norm_ba = np.linalg.norm(vector_ba, axis=-1)
    norm_bc = np.linalg.norm(vector_bc, axis=-1)

    # Avoid division by zero
    denominator = norm_ba * norm_bc
    denominator = np.where(denominator == 0, 1e-7, denominator)

    cosine_angle = np.clip(dot_prod / denominator, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


def get_point(df: pd.DataFrame, index: int) -> np.ndarray:
    """Extract (x, y, z) coordinates for a given landmark index across all rows."""
    x = df[f"landmark_{index}_x"].to_numpy()
    y = df[f"landmark_{index}_y"].to_numpy()
    z = df[f"landmark_{index}_z"].to_numpy()
    return np.column_stack([x, y, z])


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative coordinates, distances, and angles from raw landmark dataframe."""
    new_features = {}

    # Reference point: hip midpoint
    left_hip = get_point(raw_df, LEFT_HIP)
    right_hip = get_point(raw_df, RIGHT_HIP)
    hip_midpoint = (left_hip + right_hip) / 2.0

    # A. Relative landmark coordinates (centered at hip midpoint)
    for i in range(NUM_LANDMARKS):
        pt = get_point(raw_df, i)
        rel_pt = pt - hip_midpoint
        new_features[f"relative_landmark_{i}_x"] = rel_pt[:, 0]
        new_features[f"relative_landmark_{i}_y"] = rel_pt[:, 1]
        new_features[f"relative_landmark_{i}_z"] = rel_pt[:, 2]

    # Pre-extract landmark points for distances and angles
    left_shoulder = get_point(raw_df, LEFT_SHOULDER)
    right_shoulder = get_point(raw_df, RIGHT_SHOULDER)
    left_elbow = get_point(raw_df, LEFT_ELBOW)
    right_elbow = get_point(raw_df, RIGHT_ELBOW)
    left_wrist = get_point(raw_df, LEFT_WRIST)
    right_wrist = get_point(raw_df, RIGHT_WRIST)
    left_knee = get_point(raw_df, LEFT_KNEE)
    right_knee = get_point(raw_df, RIGHT_KNEE)
    left_ankle = get_point(raw_df, LEFT_ANKLE)
    right_ankle = get_point(raw_df, RIGHT_ANKLE)

    # B. Important body distances
    new_features["shoulder_width"] = calculate_distance(left_shoulder, right_shoulder)
    new_features["hip_width"] = calculate_distance(left_hip, right_hip)
    new_features["left_upper_arm_length"] = calculate_distance(left_shoulder, left_elbow)
    new_features["left_forearm_length"] = calculate_distance(left_elbow, left_wrist)
    new_features["right_upper_arm_length"] = calculate_distance(right_shoulder, right_elbow)
    new_features["right_forearm_length"] = calculate_distance(right_elbow, right_wrist)
    new_features["left_thigh_length"] = calculate_distance(left_hip, left_knee)
    new_features["left_shin_length"] = calculate_distance(left_knee, left_ankle)
    new_features["right_thigh_length"] = calculate_distance(right_hip, right_knee)
    new_features["right_shin_length"] = calculate_distance(right_knee, right_ankle)

    # C. Joint angles (in degrees)
    # Elbow angles: shoulder -> elbow -> wrist
    new_features["left_elbow_angle"] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    new_features["right_elbow_angle"] = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Knee angles: hip -> knee -> ankle
    new_features["left_knee_angle"] = calculate_angle(left_hip, left_knee, left_ankle)
    new_features["right_knee_angle"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Shoulder angles: hip -> shoulder -> elbow
    new_features["left_shoulder_angle"] = calculate_angle(left_hip, left_shoulder, left_elbow)
    new_features["right_shoulder_angle"] = calculate_angle(right_hip, right_shoulder, right_elbow)

    # Combine original features (except activity), new engineered features, and activity label (if present)
    raw_feature_cols = [c for c in raw_df.columns if c != "activity"]
    engineered_df = pd.DataFrame(new_features, index=raw_df.index)

    if "activity" in raw_df.columns:
        processed = pd.concat([raw_df[raw_feature_cols], engineered_df, raw_df[["activity"]]], axis=1)
    else:
        processed = pd.concat([raw_df[raw_feature_cols], engineered_df], axis=1)
    return processed


def extract_features_from_landmarks(landmarks) -> pd.DataFrame:

    if landmarks is None or len(landmarks) != NUM_LANDMARKS:
        return None

    raw_dict = {}
    for i, lm in enumerate(landmarks):
        raw_dict[f"landmark_{i}_x"] = lm.x
        raw_dict[f"landmark_{i}_y"] = lm.y
        raw_dict[f"landmark_{i}_z"] = lm.z
        raw_dict[f"landmark_{i}_visibility"] = lm.visibility

    raw_df = pd.DataFrame([raw_dict])
    return engineer_features(raw_df)


def create_distribution_chart(df: pd.DataFrame, output_path: str) -> None:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = df["activity"].value_counts()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=["#2b5c8f", "#3c8dbc", "#46a546", "#f89406"], edgecolor="black")

    plt.title("Sample Distribution by Activity", fontsize=14, pad=12)
    plt.xlabel("Activity", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 8,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved activity distribution chart: {output_path}")


def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] Raw dataset not found: {RAW_DATA_PATH}")
        return

    print(f"Loading raw dataset from: {RAW_DATA_PATH}")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    print(f"Original shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # Validate raw data
    null_count = raw_df.isnull().sum().sum()
    if null_count > 0:
        print(f"[WARNING] Raw dataset contains {null_count} missing values. Dropping rows with nulls.")
        raw_df = raw_df.dropna()
    else:
        print("Data validation: No missing values in raw dataset.")

    # Feature engineering
    print("Extracting relative coordinates, distances, and joint angles...")
    processed_df = engineer_features(raw_df)

    # Check for invalid values in processed data
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(processed_df[numeric_cols].to_numpy()).sum()
    nan_count = processed_df[numeric_cols].isnull().sum().sum()

    if inf_count > 0 or nan_count > 0:
        print(f"[WARNING] Processed data has {inf_count} infinite and {nan_count} NaN values. Cleaning...")
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        print("Data validation: No NaN or Infinite values in engineered features.")

    # Save to data/processed/features.csv
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Successfully saved processed features to: {PROCESSED_DATA_PATH}")
    print(f"Processed shape: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")

    # Generate visualization
    create_distribution_chart(processed_df, CHART_PATH)


if __name__ == "__main__":
    main()
