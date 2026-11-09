

import sys
import os
import time
import csv
import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pose_detection import load_landmarker, draw_pose, MODEL_PATH

# Output CSV file
CSV_PATH = os.path.join("data", "raw", "pose_dataset.csv")

# MediaPipe gives 33 landmarks per pose
NUM_LANDMARKS = 33

# Collect one sample every N milliseconds to avoid flooding with duplicates
SAMPLE_INTERVAL_MS = 150

# Activity key bindings
ACTIVITY_KEYS = {
    ord("1"): "standing",
    ord("2"): "sitting",
    ord("3"): "walking",
    ord("4"): "running",
}


def build_csv_header() -> list[str]:
    """Build the column names: landmark_0_x, landmark_0_y, ... landmark_32_visibility, activity."""
    columns = []
    for i in range(NUM_LANDMARKS):
        columns += [
            f"landmark_{i}_x",
            f"landmark_{i}_y",
            f"landmark_{i}_z",
            f"landmark_{i}_visibility",
        ]
    columns.append("activity")
    return columns


def landmarks_to_row(landmarks, activity: str) -> list:
    """
    Flatten a list of NormalizedLandmark objects into a single row of floats
    followed by the activity label.

    Returns None if the landmark count doesn't match what we expect.
    """
    if len(landmarks) != NUM_LANDMARKS:
        return None

    row = []
    for lm in landmarks:
        row += [lm.x, lm.y, lm.z, lm.visibility]
    row.append(activity)
    return row


def open_csv(path: str) -> tuple[csv.writer, object]:
    """
    Open the CSV file for appending.
    If it doesn't exist, create it and write the header first.
    Returns (writer, file_handle).
    """
    file_exists = os.path.exists(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a", newline="")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(build_csv_header())
        print(f"Created new dataset file: {path}")
    else:
        print(f"Appending to existing dataset: {path}")

    return writer, f


def draw_ui(frame: np.ndarray, activity: str | None, sample_count: int) -> None:
    """Draw the HUD overlay on the frame showing current activity and sample count."""
    h, w = frame.shape[:2]

    # Dark semi-transparent bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Instructions
    instructions = "1=standing  2=sitting  3=walking  4=running  q=quit"
    cv2.putText(frame, instructions, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if activity is None:
        cv2.putText(frame, "Press 1-4 to select an activity", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    else:
        status = f"Activity: {activity}   Samples: {sample_count}"
        cv2.putText(frame, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)


def collect_pose_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("\n=== Pose Data Collection ===")
    print("1 = standing | 2 = sitting | 3 = walking | 4 = running | q = quit")
    print("Press an activity key to start collecting samples.")
    print(f"Saving to: {CSV_PATH}\n")

    writer, csv_file = open_csv(CSV_PATH)

    current_activity = None
    sample_count = 0
    last_sample_time = 0  # milliseconds

    # Monotonically increasing millisecond timestamp tracking for MediaPipe
    start_wall_ms = int(time.perf_counter() * 1000)
    last_ts_ms = -1

    with load_landmarker(MODEL_PATH) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Ensure strictly monotonically increasing timestamp
            calc_ms = int(time.perf_counter() * 1000) - start_wall_ms
            timestamp_ms = max(calc_ms, last_ts_ms + 1)
            last_ts_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw pose skeleton if detected
            if result.pose_landmarks:
                draw_pose(frame, result)
            else:
                cv2.putText(frame, "No pose detected", (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Save a sample only when: an activity is selected, pose is found,
            # and enough time has passed since the last sample
            if (
                current_activity is not None
                and result.pose_landmarks
                and (timestamp_ms - last_sample_time) >= SAMPLE_INTERVAL_MS
            ):
                landmarks = result.pose_landmarks[0]  # first (and only) detected person
                row = landmarks_to_row(landmarks, current_activity)

                if row is not None:
                    writer.writerow(row)
                    csv_file.flush()  # write to disk immediately
                    sample_count += 1
                    last_sample_time = timestamp_ms

            draw_ui(frame, current_activity, sample_count)
            cv2.imshow("Data Collection — press q to quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(f"\nQuitting. Total samples collected this session: {sample_count}")
                break
            elif key in ACTIVITY_KEYS:
                new_activity = ACTIVITY_KEYS[key]
                if new_activity != current_activity:
                    if current_activity is not None:
                        print(f"  Switched from '{current_activity}' to '{new_activity}'")
                    else:
                        print(f"  Started collecting: {new_activity}")
                    current_activity = new_activity

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_pose_data()
