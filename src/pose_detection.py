
import sys
import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarksConnections
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec

MODEL_PATH = os.path.join("models", "pose_landmarker_full.task")

# Drawing colors (BGR)
LANDMARK_COLOR = (0, 255, 0)      # green dots
CONNECTION_COLOR = (255, 255, 0)  # cyan lines

LANDMARK_SPEC = DrawingSpec(color=LANDMARK_COLOR, thickness=2, circle_radius=4)
CONNECTION_SPEC = DrawingSpec(color=CONNECTION_COLOR, thickness=2)


def load_landmarker(model_path: str) -> mp_vision.PoseLandmarker:
    """Create and return a PoseLandmarker configured for video frames."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("Download it from:")
        print("  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
        print("and place it at:", model_path)
        sys.exit(1)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def draw_pose(frame: np.ndarray, result: mp_vision.PoseLandmarkerResult) -> None:
    """Draw detected pose landmarks and their connections onto the frame in-place."""
    if not result.pose_landmarks:
        return

    connections = PoseLandmarksConnections.POSE_LANDMARKS
    for pose_landmarks in result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=pose_landmarks,
            connections=connections,
            landmark_drawing_spec=LANDMARK_SPEC,
            connection_drawing_spec=CONNECTION_SPEC,
        )


def run_pose_detection():
    """Main loop: open webcam, detect pose on each frame, display the result."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure it is connected and not in use.")
        sys.exit(1)

    print("Webcam opened. Press 'q' to quit.")

    start_wall_ms = int(time.perf_counter() * 1000)
    last_ts_ms = -1

    with load_landmarker(MODEL_PATH) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Monotonically increasing timestamp in milliseconds
            calc_ms = int(time.perf_counter() * 1000) - start_wall_ms
            timestamp_ms = max(calc_ms, last_ts_ms + 1)
            last_ts_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                cv2.putText(frame, "No pose detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                draw_pose(frame, result)

            cv2.imshow("Pose Detection  —  press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_detection()
