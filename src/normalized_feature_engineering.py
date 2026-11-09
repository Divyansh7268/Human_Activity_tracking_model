
import numpy as np
import pandas as pd

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

SAMPLING_INTERVAL_S = 0.15


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance between two 3D points or arrays of points."""
    return np.linalg.norm(point_a - point_b, axis=-1)


def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> np.ndarray:
    """
    Calculate the joint angle (in degrees) at vertex point_b formed by
    (point_a - point_b) and (point_c - point_b).
    """
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    dot_prod = np.sum(vector_ba * vector_bc, axis=-1)
    norm_ba = np.linalg.norm(vector_ba, axis=-1)
    norm_bc = np.linalg.norm(vector_bc, axis=-1)

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


def engineer_normalized_spatial_features(raw_df: pd.DataFrame) -> pd.DataFrame:

    new_features = {}

    # 1. Reference Points: Hip Midpoint & Shoulder Midpoint
    left_hip = get_point(raw_df, LEFT_HIP)
    right_hip = get_point(raw_df, RIGHT_HIP)
    hip_midpoint = (left_hip + right_hip) / 2.0

    left_shoulder = get_point(raw_df, LEFT_SHOULDER)
    right_shoulder = get_point(raw_df, RIGHT_SHOULDER)
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0

    # 2. Scale Metric: Torso Length (rigid trunk height)
    torso_length = calculate_distance(shoulder_midpoint, hip_midpoint)
    # Guard against division by zero
    scale = np.where(torso_length < 1e-4, 1e-4, torso_length)

    # 3. Normalized Landmark Coordinates (Centered at hip midpoint, divided by torso length)
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

    # 4. Normalized Body Segment Lengths (scaled by torso length)
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

    # 5. Joint Angles (Degrees, naturally scale- and translation-invariant)
    # Elbow angles: shoulder -> elbow -> wrist
    new_features["left_elbow_angle"] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    new_features["right_elbow_angle"] = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Knee angles: hip -> knee -> ankle
    new_features["left_knee_angle"] = calculate_angle(left_hip, left_knee, left_ankle)
    new_features["right_knee_angle"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Shoulder angles: hip -> shoulder -> elbow
    new_features["left_shoulder_angle"] = calculate_angle(left_hip, left_shoulder, left_elbow)
    new_features["right_shoulder_angle"] = calculate_angle(right_hip, right_shoulder, right_elbow)

    # Hip angles: shoulder -> hip -> knee (vital for sitting vs standing posture)
    new_features["left_hip_angle"] = calculate_angle(left_shoulder, left_hip, left_knee)
    new_features["right_hip_angle"] = calculate_angle(right_shoulder, right_hip, right_knee)

    # 6. Landmark Visibility Scores (Confidence values in [0, 1])
    for i in range(NUM_LANDMARKS):
        vis_col = f"landmark_{i}_visibility"
        if vis_col in raw_df.columns:
            new_features[vis_col] = raw_df[vis_col].to_numpy()

    processed_df = pd.DataFrame(new_features, index=raw_df.index)

    if "activity" in raw_df.columns:
        processed_df["activity"] = raw_df["activity"].values

    return processed_df


def compute_normalized_temporal_features(
    coords_raw: np.ndarray,
    torso_lengths: np.ndarray,
    dt: float = SAMPLING_INTERVAL_S,
) -> dict:
    """
    Compute frame-to-frame movement metrics on scale-normalized coordinates.

    By dividing raw landmark coordinates by torso length at each frame,
    displacement is measured in 'torso lengths moved per frame', making
    movement speed and acceleration completely invariant to subject-to-camera distance.
    """
    if len(coords_raw) < 2:
        return {}

    # Scale coordinates: shape (N, 33, 3)
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
    speed = mean_disp / dt

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

