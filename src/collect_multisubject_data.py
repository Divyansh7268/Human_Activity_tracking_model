
import os
import sys
import time
import csv
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Pose landmarker and drawing configuration
sys.path.insert(0, os.path.dirname(__file__))
from pose_detection import load_landmarker, draw_pose, MODEL_PATH

CSV_OUTPUT_PATH = os.path.join("data", "raw", "multisubject_pose_dataset.csv")
NUM_LANDMARKS = 33
MIN_VALID_FRAMES = 50
WARN_VALID_PERCENT = 80.0

ACTIVITIES = {
    1: {"name": "standing", "code": 1},
    2: {"name": "sitting", "code": 2},
    3: {"name": "walking", "code": 3},
    4: {"name": "running", "code": 4},
}

METADATA_HEADER = [
    "subject_id",
    "activity",
    "activity_code",
    "trial_id",
    "sequence_id",
    "frame_id",
    "timestamp",
]


def build_csv_header() -> list[str]:
    """Build full CSV column header (7 metadata columns + 132 landmark columns)."""
    cols = list(METADATA_HEADER)
    for i in range(NUM_LANDMARKS):
        cols.extend([
            f"landmark_{i}_x",
            f"landmark_{i}_y",
            f"landmark_{i}_z",
            f"landmark_{i}_visibility",
        ])
    return cols


def format_sequence_id(subject_id: str, activity_name: str, trial_id: int) -> str:
    """Format standardized sequence ID: e.g. S01_WALKING_T01."""
    sub_clean = subject_id.strip().upper()
    act_clean = activity_name.strip().upper()
    return f"{sub_clean}_{act_clean}_T{trial_id:02d}"


def get_existing_sequences() -> pd.DataFrame | None:
    """Read existing dataset if present to check for duplicate sequences/trials."""
    if os.path.exists(CSV_OUTPUT_PATH):
        try:
            df = pd.read_csv(CSV_OUTPUT_PATH, usecols=["subject_id", "activity", "trial_id", "sequence_id"])
            return df
        except Exception:
            return None
    return None


def sequence_exists(subject_id: str, activity_name: str, trial_id: int) -> bool:
    """Check if subject + activity + trial combination is already in the CSV."""
    df = get_existing_sequences()
    if df is None or len(df) == 0:
        return False
    match = df[
        (df["subject_id"].astype(str).str.upper() == subject_id.strip().upper())
        & (df["activity"].astype(str).str.lower() == activity_name.strip().lower())
        & (df["trial_id"].astype(int) == int(trial_id))
    ]
    return len(match) > 0


def get_next_available_trial(subject_id: str, activity_name: str) -> int:
    """Determine the next unused trial ID for a given subject and activity."""
    df = get_existing_sequences()
    if df is None or len(df) == 0:
        return 1
    sub_df = df[
        (df["subject_id"].astype(str).str.upper() == subject_id.strip().upper())
        & (df["activity"].astype(str).str.lower() == activity_name.strip().lower())
    ]
    if len(sub_df) == 0:
        return 1
    return int(sub_df["trial_id"].max()) + 1


def append_sequence_to_csv(rows: list[list], overwrite_seq_id: str | None = None) -> None:
    """
    Append sequence rows to multisubject_pose_dataset.csv.
    If overwrite_seq_id is provided, removes existing rows matching that sequence_id first.
    """
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    header = build_csv_header()

    if overwrite_seq_id and os.path.exists(CSV_OUTPUT_PATH):
        # Read all rows, filter out overwrite_seq_id, then rewrite and append
        existing_df = pd.read_csv(CSV_OUTPUT_PATH)
        existing_df = existing_df[existing_df["sequence_id"] != overwrite_seq_id]
        existing_df.to_csv(CSV_OUTPUT_PATH, index=False)

    file_exists = os.path.exists(CSV_OUTPUT_PATH)
    with open(CSV_OUTPUT_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(CSV_OUTPUT_PATH) == 0:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def draw_countdown(frame: np.ndarray, seconds_left: int, subject: str, activity: str, trial: int):
    """Draw a clean countdown screen overlay before recording begins."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    info_str = f"Subject: {subject}   Activity: {activity.upper()}   Trial: {trial}"
    cv2.putText(frame, info_str, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Get into position! Recording starts in:", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Big countdown number
    count_text = str(seconds_left)
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
    cx = (w - text_size[0]) // 2
    cy = (h + text_size[1]) // 2
    cv2.putText(frame, count_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 100), 8)


def draw_recording_hud(
    frame: np.ndarray,
    subject: str,
    activity: str,
    trial: int,
    valid_count: int,
    total_count: int,
    elapsed_sec: float,
    pose_detected: bool,
):
    """Draw the recording HUD with status indicators and live frame counts."""
    h, w = frame.shape[:2]

    # Top status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Blinking red RECORDING dot
    is_blink = int(elapsed_sec * 2) % 2 == 0
    dot_color = (0, 0, 255) if is_blink else (50, 50, 150)
    cv2.circle(frame, (25, 30), 10, dot_color, -1)

    cv2.putText(frame, "REC", (45, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    title = f"Subject: {subject} | {activity.upper()} | Trial {trial}"
    cv2.putText(frame, title, (110, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    det_rate = (valid_count / total_count * 100.0) if total_count > 0 else 0.0
    stats = f"Valid: {valid_count} | Total: {total_count} ({det_rate:.1f}%) | Time: {elapsed_sec:.1f}s | Finish: 'q' or SPACE"
    cv2.putText(frame, stats, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)

    # Pose detection indicator at bottom
    if not pose_detected:
        cv2.putText(frame, "WARNING: No human pose detected in frame!", (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def record_sequence(
    cap: cv2.VideoCapture,
    subject_id: str,
    activity_info: dict,
    trial_id: int,
) -> tuple[list[list], dict]:
    """
    Record a single trial sequence from the webcam.
    Returns: (list_of_data_rows, summary_metrics)
    """
    activity_name = activity_info["name"]
    activity_code = activity_info["code"]
    sequence_id = format_sequence_id(subject_id, activity_name, trial_id)

    print(f"\n" + "=" * 65)
    print(f"PREPARING RECORDING: {sequence_id}")
    print(f"Subject: {subject_id} | Activity: {activity_name} (code {activity_code}) | Trial: {trial_id}")
    print(f"Controls: Press 'q' or SPACE in the window to stop recording.")
    print("=" * 65)

    # 3-second countdown
    countdown_start = time.time()
    while True:
        elapsed = time.time() - countdown_start
        remaining = 3 - int(elapsed)
        if remaining <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror for natural webcam interaction
        draw_countdown(frame, remaining, subject_id, activity_name, trial_id)
        cv2.imshow("Multi-Subject Pose Data Collector", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Recording cancelled during countdown.")
            return [], {}

    # Start recording with a fresh PoseLandmarker for this sequence
    print(f"Recording started for {sequence_id}... (Press 'q' or SPACE when finished)")
    recorded_rows = []
    total_frames = 0
    valid_frames = 0
    rec_start_time = time.time()
    last_timestamp_ms = -1
    frame_index = 0

    with load_landmarker(MODEL_PATH) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            total_frames += 1
            elapsed_time = round(time.time() - rec_start_time, 4)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Ensure timestamp is strictly monotonically increasing (> last_timestamp_ms)
            calc_ms = int(elapsed_time * 1000)
            timestamp_ms = max(calc_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_detected = bool(result.pose_landmarks and len(result.pose_landmarks) > 0)

            if pose_detected:
                draw_pose(frame, result)
                landmarks = result.pose_landmarks[0]
                if len(landmarks) == NUM_LANDMARKS:
                    row = [
                        subject_id,
                        activity_name,
                        activity_code,
                        trial_id,
                        sequence_id,
                        valid_frames,  # sequential frame_id starting at 0
                        elapsed_time,
                    ]
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])

                    recorded_rows.append(row)
                    valid_frames += 1

            draw_recording_hud(
                frame,
                subject_id,
                activity_name,
                trial_id,
                valid_frames,
                total_frames,
                elapsed_time,
                pose_detected,
            )

            cv2.imshow("Multi-Subject Pose Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 32):  # 'q' or SPACE to finish
                print(f"Recording stopped by user at {elapsed_time:.1f}s.")
                break

            frame_index += 1

    total_duration = round(time.time() - rec_start_time, 2)
    detection_rate = (valid_frames / total_frames * 100.0) if total_frames > 0 else 0.0

    summary = {
        "subject_id": subject_id,
        "activity": activity_name,
        "trial_id": trial_id,
        "sequence_id": sequence_id,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "invalid_frames": total_frames - valid_frames,
        "duration_sec": total_duration,
        "detection_rate": detection_rate,
    }
    return recorded_rows, summary


def evaluate_quality_and_save(rows: list[list], summary: dict) -> bool:
    """Evaluate quality criteria, alert user, and append to dataset if accepted."""
    if not summary or not rows:
        print("[WARNING] No frames captured. Sequence discarded.")
        return False

    valid = summary["valid_frames"]
    total = summary["total_frames"]
    rate = summary["detection_rate"]
    seq_id = summary["sequence_id"]

    print("\n" + "─" * 65)
    print("SEQUENCE RECORDING SUMMARY:")
    print("─" * 65)
    print(f"  Sequence ID         : {seq_id}")
    print(f"  Subject             : {summary['subject_id']}")
    print(f"  Activity            : {summary['activity']}")
    print(f"  Trial               : {summary['trial_id']}")
    print(f"  Duration            : {summary['duration_sec']} seconds")
    print(f"  Total Frames        : {total}")
    print(f"  Valid Pose Frames   : {valid}")
    print(f"  Invalid Pose Frames : {summary['invalid_frames']}")
    print(f"  Detection Rate      : {rate:.1f}%")
    print("─" * 65)

    # Rule 1: Reject if < 50 valid frames
    if valid < MIN_VALID_FRAMES:
        print(f"\n[QUALITY REJECTION] Sequence contains {valid} valid frames (< {MIN_VALID_FRAMES} required).")
        print("This sequence is too short for temporal feature extraction.")
        choice = input("Discard this trial and re-record? ([y]/n): ").strip().lower()
        if choice != "n":
            return False

    # Rule 2: Warn if detection rate < 80%
    if rate < WARN_VALID_PERCENT:
        print(f"\n[QUALITY WARNING] Pose detection rate is {rate:.1f}% (< {WARN_VALID_PERCENT}% target).")
        print("Camera angle or lighting may have caused occlusion.")
        save_choice = input("Do you still want to save this sequence? (y/[n]): ").strip().lower()
        if save_choice != "y":
            print("Sequence discarded by user.")
            return False

    # Check for existing sequence overwrite
    overwrite_target = None
    if sequence_exists(summary["subject_id"], summary["activity"], summary["trial_id"]):
        print(f"\n[WARNING] Sequence {seq_id} already exists in {CSV_OUTPUT_PATH}.")
        rep = input("Overwrite previous recording for this trial? (y/[n]): ").strip().lower()
        if rep != "y":
            print("Sequence not saved.")
            return False
        overwrite_target = seq_id

    append_sequence_to_csv(rows, overwrite_seq_id=overwrite_target)
    action_str = "Overwritten and saved" if overwrite_target else "Appended and saved"
    print(f"\n[SUCCESS] {action_str} {valid} valid frames to {CSV_OUTPUT_PATH}")
    return True


def print_overall_dataset_summary():
    """Print high-level dataset statistics at exit."""
    if not os.path.exists(CSV_OUTPUT_PATH):
        print("\nNo dataset file created yet.")
        return

    try:
        df = pd.read_csv(CSV_OUTPUT_PATH, usecols=["subject_id", "activity", "sequence_id", "frame_id"])
        print("\n" + "=" * 65)
        print("OVERALL MULTI-SUBJECT DATASET SUMMARY (data/raw/multisubject_pose_dataset.csv):")
        print("=" * 65)
        print(f"Total Rows (Frames)      : {len(df)}")
        print(f"Total Unique Subjects    : {df['subject_id'].nunique()} ({sorted(df['subject_id'].unique().tolist())})")
        print(f"Total Unique Sequences   : {df['sequence_id'].nunique()}")

        print("\nSequences per Activity:")
        seq_per_act = df.groupby("activity")["sequence_id"].nunique()
        for act, cnt in seq_per_act.items():
            print(f"  {act:12s}: {cnt:3d} sequence(s)")

        print("\nFrames per Activity:")
        for act, cnt in df["activity"].value_counts().items():
            print(f"  {act:12s}: {cnt:5d} frames")
        print("=" * 65)
    except Exception as e:
        print(f"Could not read dataset summary: {e}")


def interactive_collection_loop():
    """Main interactive menu loop for data collection."""
    print("=" * 65)
    print("RESEARCH-GRADE MULTI-SUBJECT POSE DATA COLLECTION")
    print("=" * 65)
    print(f"Target dataset: {CSV_OUTPUT_PATH}")
    print("Target activities: 1=standing, 2=sitting, 3=walking, 4=running")
    print("Required protocol: 5+ subjects x 4 activities x 3+ trials")
    print("=" * 65)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure your camera is connected and permitted.")
        return

    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] PoseLandmarker model missing at: {MODEL_PATH}")
        cap.release()
        return

    current_subject = None

    while True:
        # Subject ID input
        if current_subject is None:
            sub_input = input("\nEnter Subject ID (e.g. S01, S02, S03) [or 'q' to quit]: ").strip()
            if not sub_input or sub_input.lower() == "q":
                break
            current_subject = sub_input.upper()

        print(f"\nActive Subject: {current_subject}")
        print("Select Activity:")
        print("  [1] standing")
        print("  [2] sitting")
        print("  [3] walking")
        print("  [4] running")
        print("  [S] Switch Subject ID")
        print("  [Q] Quit program")

        act_choice = input("Enter choice (1-4, S, Q): ").strip()
        if act_choice.upper() == "Q":
            break
        elif act_choice.upper() == "S":
            current_subject = None
            continue

        try:
            act_num = int(act_choice)
            if act_num not in ACTIVITIES:
                print("Invalid selection. Enter 1, 2, 3, or 4.")
                continue
        except ValueError:
            print("Invalid input.")
            continue

        activity_info = ACTIVITIES[act_num]
        suggested_trial = get_next_available_trial(current_subject, activity_info["name"])

        trial_input = input(f"Enter Trial number [press Enter for suggested Trial {suggested_trial}]: ").strip()
        if trial_input == "":
            trial_id = suggested_trial
        else:
            try:
                trial_id = int(trial_input)
                if trial_id < 1:
                    raise ValueError()
            except ValueError:
                print("Trial number must be a positive integer.")
                continue

        # Record sequence
        rows, summary = record_sequence(cap, current_subject, activity_info, trial_id)
        evaluate_quality_and_save(rows, summary)

    cap.release()
    cv2.destroyAllWindows()
    print_overall_dataset_summary()


if __name__ == "__main__":
    interactive_collection_loop()

