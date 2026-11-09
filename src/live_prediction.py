

import os
import sys
import json
import time
import collections

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import drawing_utils, PoseLandmarksConnections
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec


# Paths (project-relative — run from c:\Human_Activity_Rec)

MODEL_FILE      = os.path.join("models", "final_deployment_model.pkl")
SCHEMA_FILE     = os.path.join("models", "final_feature_schema.json")
LANDMARKER_FILE = os.path.join("models", "pose_landmarker_full.task")
REPORT_FILE     = os.path.join("results", "live_prediction_report.txt")


# MediaPipe landmark indices (matches normalized_feature_engineering.py)

NUM_LANDMARKS    = 33
NOSE             = 0
LEFT_SHOULDER    = 11
RIGHT_SHOULDER   = 12
LEFT_ELBOW       = 13
RIGHT_ELBOW      = 14
LEFT_WRIST       = 15
RIGHT_WRIST      = 16
LEFT_HIP         = 23
RIGHT_HIP        = 24
LEFT_KNEE        = 25
RIGHT_KNEE       = 26
LEFT_ANKLE       = 27
RIGHT_ANKLE      = 28

# Body-part groupings for temporal features (matches normalized_feature_engineering.py)
UPPER_BODY_INDICES = list(range(23))
LOWER_BODY_INDICES = list(range(23, 33))
LEFT_SIDE_INDICES  = [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RIGHT_SIDE_INDICES = [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# Sampling interval used during training (from SAMPLING_INTERVAL_S = 0.15 s)
DT = 0.15


# Temporal buffer and smoothing settings

MIN_BUFFER_FRAMES = 3     # Need at least this many frames before predicting
BUFFER_MAXLEN     = 30    # Rolling window kept in memory
SMOOTH_WINDOW     = 7     # Prediction history for majority-vote smoothing


# Display settings

ACTIVITY_COLORS = {
    "standing": (0,   200, 0),
    "sitting":  (0,   140, 255),
    "walking":  (255, 200, 0),
    "running":  (0,   0,   255),
}
DEFAULT_COLOR = (200, 200, 200)

LANDMARK_SPEC   = DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=3)
CONNECTION_SPEC = DrawingSpec(color=(255, 255, 0),  thickness=2)



# Feature Engineering (mirrors normalized_feature_engineering.py exactly)


def _pt(lm_array, idx):
    """Return (x, y, z) for a single landmark from shape (33, 4)."""
    return lm_array[idx, :3]


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _angle(a, b, c):
    """Angle in degrees at vertex b, formed by a-b-c."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-7:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def extract_spatial_features(lm_array):
    
    feats = {}

    # Reference points
    left_hip_pt      = _pt(lm_array, LEFT_HIP)
    right_hip_pt     = _pt(lm_array, RIGHT_HIP)
    hip_mid          = (left_hip_pt + right_hip_pt) / 2.0

    left_shoulder_pt  = _pt(lm_array, LEFT_SHOULDER)
    right_shoulder_pt = _pt(lm_array, RIGHT_SHOULDER)
    shoulder_mid      = (left_shoulder_pt + right_shoulder_pt) / 2.0

    torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    scale     = max(torso_len, 1e-4)

    # 99 normalized landmark coordinates
    for i in range(NUM_LANDMARKS):
        pt  = lm_array[i, :3]
        rel = pt - hip_mid
        feats[f"norm_landmark_{i}_x"] = rel[0] / scale
        feats[f"norm_landmark_{i}_y"] = rel[1] / scale
        feats[f"norm_landmark_{i}_z"] = rel[2] / scale

    # Convenience variables
    left_elbow_pt  = _pt(lm_array, LEFT_ELBOW)
    right_elbow_pt = _pt(lm_array, RIGHT_ELBOW)
    left_wrist_pt  = _pt(lm_array, LEFT_WRIST)
    right_wrist_pt = _pt(lm_array, RIGHT_WRIST)
    left_knee_pt   = _pt(lm_array, LEFT_KNEE)
    right_knee_pt  = _pt(lm_array, RIGHT_KNEE)
    left_ankle_pt  = _pt(lm_array, LEFT_ANKLE)
    right_ankle_pt = _pt(lm_array, RIGHT_ANKLE)

    # 10 normalized segment distances
    feats["norm_shoulder_width"]  = _dist(left_shoulder_pt,  right_shoulder_pt) / scale
    feats["norm_hip_width"]       = _dist(left_hip_pt,       right_hip_pt)      / scale
    feats["norm_left_upper_arm"]  = _dist(left_shoulder_pt,  left_elbow_pt)     / scale
    feats["norm_left_forearm"]    = _dist(left_elbow_pt,     left_wrist_pt)     / scale
    feats["norm_right_upper_arm"] = _dist(right_shoulder_pt, right_elbow_pt)    / scale
    feats["norm_right_forearm"]   = _dist(right_elbow_pt,    right_wrist_pt)    / scale
    feats["norm_left_thigh"]      = _dist(left_hip_pt,       left_knee_pt)      / scale
    feats["norm_left_shin"]       = _dist(left_knee_pt,      left_ankle_pt)     / scale
    feats["norm_right_thigh"]     = _dist(right_hip_pt,      right_knee_pt)     / scale
    feats["norm_right_shin"]      = _dist(right_knee_pt,     right_ankle_pt)    / scale

    # 8 joint angles (degrees)
    feats["left_elbow_angle"]     = _angle(left_shoulder_pt,  left_elbow_pt,    left_wrist_pt)
    feats["right_elbow_angle"]    = _angle(right_shoulder_pt, right_elbow_pt,   right_wrist_pt)
    feats["left_knee_angle"]      = _angle(left_hip_pt,       left_knee_pt,     left_ankle_pt)
    feats["right_knee_angle"]     = _angle(right_hip_pt,      right_knee_pt,    right_ankle_pt)
    feats["left_shoulder_angle"]  = _angle(left_hip_pt,       left_shoulder_pt, left_elbow_pt)
    feats["right_shoulder_angle"] = _angle(right_hip_pt,      right_shoulder_pt,right_elbow_pt)
    feats["left_hip_angle"]       = _angle(left_shoulder_pt,  left_hip_pt,      left_knee_pt)
    feats["right_hip_angle"]      = _angle(right_shoulder_pt, right_hip_pt,     right_knee_pt)

    # 33 visibility scores
    for i in range(NUM_LANDMARKS):
        feats[f"landmark_{i}_visibility"] = float(lm_array[i, 3])

    return feats, torso_len


def compute_temporal_features(coords_buffer, torso_buffer):
    
    n = len(coords_buffer)
    if n < 2:
        return {}

    coords_raw = np.stack(coords_buffer, axis=0)      # (N, 33, 3)
    torso_arr  = np.array(torso_buffer, dtype=float)  # (N,)

    scale_3d      = np.where(torso_arr < 1e-4, 1e-4, torso_arr)[:, np.newaxis, np.newaxis]
    scaled_coords = coords_raw / scale_3d             # (N, 33, 3)

    # Frame-to-frame displacement: (N-1, 33)
    disp = np.linalg.norm(scaled_coords[1:] - scaled_coords[:-1], axis=-1)

    mean_disp  = disp.mean(axis=-1)                              # (N-1,)
    max_disp   = disp.max(axis=-1)
    upper_disp = disp[:, UPPER_BODY_INDICES].mean(axis=-1)
    lower_disp = disp[:, LOWER_BODY_INDICES].mean(axis=-1)
    left_disp  = disp[:, LEFT_SIDE_INDICES].mean(axis=-1)
    right_disp = disp[:, RIGHT_SIDE_INDICES].mean(axis=-1)
    speed      = mean_disp / DT

    accel = np.zeros_like(speed)
    if len(speed) > 1:
        accel[1:] = np.abs(speed[1:] - speed[:-1]) / DT

    # Use values for the most recent frame
    return {
        "mean_landmark_displacement": float(mean_disp[-1]),
        "max_landmark_displacement":  float(max_disp[-1]),
        "upper_body_movement":        float(upper_disp[-1]),
        "lower_body_movement":        float(lower_disp[-1]),
        "left_side_movement":         float(left_disp[-1]),
        "right_side_movement":        float(right_disp[-1]),
        "movement_speed":             float(speed[-1]),
        "movement_acceleration":      float(accel[-1]),
    }


def build_feature_vector(spatial_feats, temporal_feats, feature_names):
   
    merged = {}
    merged.update(spatial_feats)
    merged.update(temporal_feats)
    vec = []
    for name in feature_names:
        if name not in merged:
            raise KeyError(f"Missing feature in merged dict: {name}")
        vec.append(merged[name])
    return np.array(vec, dtype=np.float64)



# MediaPipe helpers


def load_pose_landmarker(model_path):
    """Create a PoseLandmarker in VIDEO mode (identical to collect_multisubject_data.py)."""
    if not os.path.exists(model_path):
        sys.exit(
            f"[ERROR] Pose landmarker model not found: {model_path}\n"
            "Place pose_landmarker_full.task in the models/ directory."
        )
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def landmarks_to_array(result):
   
    if not result or not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    if len(lms) != NUM_LANDMARKS:
        return None
    arr = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(lms):
        arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr


def draw_pose_on_frame(frame, result):
  
    if not result or not result.pose_landmarks:
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



# Display overlay


def put_text_with_bg(img, text, pos, font_scale=0.7, thickness=2,
                     fg=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with a filled rectangle background for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + baseline + 2), bg, -1)
    cv2.putText(img, text, pos, font, font_scale, fg, thickness, cv2.LINE_AA)


def render_overlay(frame, activity, confidence, fps, buf_len, smooth_full):
    """Render all HUD elements."""
    h, w = frame.shape[:2]

    # --- Activity label (large, top-left) ---
    if activity is not None:
        color = ACTIVITY_COLORS.get(activity, DEFAULT_COLOR)
        label = "Activity: " + activity.upper()
        put_text_with_bg(frame, label, (20, 55), font_scale=1.0, thickness=2,
                         fg=color, bg=(0, 0, 0))
        if confidence is not None:
            put_text_with_bg(frame, f"Confidence: {confidence*100:.1f}%",
                             (20, 95), font_scale=0.65, fg=(220, 220, 220))
    else:
        status = "Buffering..." if buf_len > 0 else "Waiting for pose..."
        put_text_with_bg(frame, status, (20, 55), font_scale=0.9,
                         fg=(180, 180, 180))

    # --- FPS (top-right) ---
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    put_text_with_bg(frame, fps_text, (w - tw - 30, 35),
                     font_scale=0.6, fg=(200, 200, 200))

    # --- Smoothing indicator ---
    sm_text = "Smoothed" if smooth_full else f"Warmup: {buf_len}/{MIN_BUFFER_FRAMES}"
    put_text_with_bg(frame, sm_text, (20, h - 90),
                     font_scale=0.5, fg=(150, 200, 150))

    # --- Buffer size ---
    put_text_with_bg(frame, f"Buffer: {buf_len} frames",
                     (20, h - 60), font_scale=0.5, fg=(180, 180, 180))

    # --- Controls ---
    put_text_with_bg(frame, "q: quit   r: reset",
                     (20, h - 30), font_scale=0.5, fg=(150, 150, 150))


# Startup verification (no camera required)


def verify_startup(feature_names):
    """
    Perform all startup checks without opening the webcam.
    Verifies model loads, schema is 158 features, feature vector builds correctly.
    """
    print("[VERIFY] Feature vector dry-run ...")
    dummy = np.zeros((33, 4), dtype=np.float32)
    dummy[:, 3] = 1.0
    # Give the dummy body a non-zero torso so scale > 0
    dummy[LEFT_HIP]       = [0.45, 0.60, 0.00, 1.0]
    dummy[RIGHT_HIP]      = [0.55, 0.60, 0.00, 1.0]
    dummy[LEFT_SHOULDER]  = [0.40, 0.35, 0.00, 1.0]
    dummy[RIGHT_SHOULDER] = [0.60, 0.35, 0.00, 1.0]
    dummy[LEFT_KNEE]      = [0.44, 0.78, 0.00, 1.0]
    dummy[RIGHT_KNEE]     = [0.56, 0.78, 0.00, 1.0]
    dummy[LEFT_ANKLE]     = [0.44, 0.92, 0.00, 1.0]
    dummy[RIGHT_ANKLE]    = [0.56, 0.92, 0.00, 1.0]
    dummy[LEFT_ELBOW]     = [0.35, 0.50, 0.00, 1.0]
    dummy[RIGHT_ELBOW]    = [0.65, 0.50, 0.00, 1.0]
    dummy[LEFT_WRIST]     = [0.30, 0.65, 0.00, 1.0]
    dummy[RIGHT_WRIST]    = [0.70, 0.65, 0.00, 1.0]

    sp_feats, torso_len = extract_spatial_features(dummy)
    assert len(sp_feats) == 150, f"Expected 150 spatial features, got {len(sp_feats)}"
    print(f"[VERIFY] Spatial features: {len(sp_feats)} ✓")

    # Need 2+ frames for temporal
    coords_buf = [dummy[:, :3].copy(), dummy[:, :3].copy(), dummy[:, :3].copy()]
    torso_buf  = [torso_len, torso_len, torso_len]
    tmp_feats  = compute_temporal_features(coords_buf, torso_buf)
    assert len(tmp_feats) == 8, f"Expected 8 temporal features, got {len(tmp_feats)}"
    print(f"[VERIFY] Temporal features: {len(tmp_feats)} ✓")

    vec = build_feature_vector(sp_feats, tmp_feats, feature_names)
    assert vec.shape[0] == 158, f"Expected 158-feature vector, got {vec.shape[0]}"
    print(f"[VERIFY] Feature vector length: {vec.shape[0]} ✓")
    assert not np.isnan(vec).any(), "NaN in feature vector"
    assert not np.isinf(vec).any(), "Inf in feature vector"
    print("[VERIFY] No NaN/Inf in feature vector ✓")
    return vec



# Main loop


def run(verify_only=False):
    """
    Main entry point.
    verify_only=True: runs startup + feature checks only, no webcam.
    """

    # --- 1. Load ML model ---
    print(f"[INIT] Loading model : {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        sys.exit(f"[ERROR] Model not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    print(f"[INIT] Model loaded. Classes: {model.classes_.tolist()} ✓")

    # --- 2. Load feature schema ---
    print(f"[INIT] Loading schema: {SCHEMA_FILE}")
    if not os.path.exists(SCHEMA_FILE):
        sys.exit(f"[ERROR] Schema not found: {SCHEMA_FILE}")
    with open(SCHEMA_FILE) as fh:
        schema = json.load(fh)
    feature_names = schema["feature_names"]

    if len(feature_names) != 158:
        sys.exit(f"[ERROR] Schema has {len(feature_names)} features, expected 158.")
    print(f"[INIT] Feature schema validated: {len(feature_names)} features ✓")

    # --- 3. Startup verification ---
    test_vec = verify_startup(feature_names)

    # Sample prediction on dummy vector
    test_pred = model.predict(test_vec.reshape(1, -1))
    assert test_pred[0] in model.classes_, "Unknown class in test prediction"
    print(f"[INIT] Sample prediction on dummy frame: '{test_pred[0]}' ✓")

    if verify_only:
        print("\n[VERIFY] All startup checks passed.")
        print("No webcam opened (--verify mode).")
        print("\nLIVE_PREDICTION_IMPLEMENTATION_COMPLETE")
        return

    # --- 4. Load pose landmarker ---
    print(f"[INIT] Loading pose landmarker: {LANDMARKER_FILE}")
    landmarker = load_pose_landmarker(LANDMARKER_FILE)
    print("[INIT] Pose landmarker ready ✓")

    # --- 5. Open webcam ---
    print("[INIT] Opening webcam (index 0) ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        landmarker.close()
        sys.exit("[ERROR] Could not open webcam. Check connection and permissions.")
    print("[INIT] Webcam opened ✓")
    print("[INIT] ===== Live prediction running. Press 'q' to quit, 'r' to reset =====")

    # --- Rolling buffers ---
    coords_buf = collections.deque(maxlen=BUFFER_MAXLEN)   # (33,3) xyz arrays
    torso_buf  = collections.deque(maxlen=BUFFER_MAXLEN)   # float torso lengths
    pred_hist  = collections.deque(maxlen=SMOOTH_WINDOW)   # str predictions

    # FPS tracking
    fps_timer   = time.perf_counter()
    fps_display = 0.0
    frame_count = 0

    # Monotonic timestamp for MediaPipe VIDEO mode
    start_wall_ms = int(time.perf_counter() * 1000)
    last_ts_ms    = -1

    current_activity   = None
    current_confidence = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab webcam frame. Retrying ...")
                time.sleep(0.05)
                continue

            # FPS
            frame_count += 1
            now = time.perf_counter()
            if now - fps_timer >= 0.5:
                fps_display = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now

            # MediaPipe inference
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            calc_ms  = int(time.perf_counter() * 1000) - start_wall_ms
            ts_ms    = max(calc_ms, last_ts_ms + 1)
            last_ts_ms = ts_ms

            try:
                result = landmarker.detect_for_video(mp_img, ts_ms)
            except Exception as exc:
                print(f"[WARN] MediaPipe error: {exc}")
                result = None

            lm_array = landmarks_to_array(result)

            # Draw skeleton
            if result is not None:
                draw_pose_on_frame(frame, result)

            # Feature extraction & prediction
            if lm_array is not None:
                try:
                    sp_feats, torso_len = extract_spatial_features(lm_array)
                    raw_xyz = lm_array[:, :3].copy()

                    coords_buf.append(raw_xyz)
                    torso_buf.append(torso_len)

                    if len(coords_buf) >= MIN_BUFFER_FRAMES:
                        tmp_feats = compute_temporal_features(
                            list(coords_buf), list(torso_buf)
                        )
                        feat_vec = build_feature_vector(sp_feats, tmp_feats, feature_names)

                        proba = model.predict_proba(feat_vec.reshape(1, -1))[0]
                        pred  = model.classes_[int(np.argmax(proba))]
                        conf  = float(np.max(proba))

                        pred_hist.append(pred)

                        # Majority-vote smoothing
                        counts = {}
                        for p in pred_hist:
                            counts[p] = counts.get(p, 0) + 1
                        current_activity   = max(counts, key=counts.get)
                        current_confidence = conf

                except Exception as exc:
                    print(f"[WARN] Feature/prediction error: {exc}")

            # Render HUD
            render_overlay(
                frame,
                activity   = current_activity,
                confidence = current_confidence,
                fps        = fps_display,
                buf_len    = len(coords_buf),
                smooth_full= (len(pred_hist) >= SMOOTH_WINDOW),
            )

            cv2.imshow("Activity Recognition -- Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] 'q' pressed — quitting.")
                break
            elif key == ord("r"):
                coords_buf.clear()
                torso_buf.clear()
                pred_hist.clear()
                current_activity   = None
                current_confidence = None
                print("[INFO] Temporal buffer reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("[INFO] Resources released. Goodbye.")



# Report


REPORT_TEXT = """\
========================================================
LIVE PREDICTION REPORT
========================================================

Script path       : src/live_prediction.py
Model used        : models/final_deployment_model.pkl
Feature schema    : models/final_feature_schema.json
Number of features: 158
Activities        : running, sitting, standing, walking

Feature breakdown (158 total)
-------------------------------
 99  Normalized landmark coordinates
      (hip-centered, torso-length scaled)
      Features: norm_landmark_{i}_{x,y,z} for i = 0..32
 10  Normalized body segment lengths
      (norm_shoulder_width, norm_hip_width,
       norm_left/right_upper_arm, norm_left/right_forearm,
       norm_left/right_thigh, norm_left/right_shin)
  8  Joint angles (degrees)
      (left/right elbow, knee, shoulder, hip angles)
 33  Landmark visibility scores
      (landmark_{i}_visibility for i = 0..32)
  8  Temporal movement features
      (mean_landmark_displacement, max_landmark_displacement,
       upper_body_movement, lower_body_movement,
       left_side_movement, right_side_movement,
       movement_speed, movement_acceleration)

Temporal buffer approach
-------------------------------
- Rolling deque of up to 30 frames maintained in memory.
- Each entry stores raw (33, 3) xyz landmarks and torso length.
- Temporal features computed from entire buffer (normalized displacement),
  matching compute_normalized_temporal_features() in
  src/normalized_feature_engineering.py exactly.
- No prediction is made until at least 3 frames are buffered.
- First frames handled gracefully by suppressing prediction below
  MIN_BUFFER_FRAMES threshold.

Prediction smoothing
-------------------------------
- Last 7 raw predictions held in a smoothing deque (SMOOTH_WINDOW=7).
- Majority vote across the window determines displayed label.
- Raw model probabilities displayed as confidence percentage.
- 'r' key clears all buffers and resets smoothing history.

Keyboard controls
-------------------------------
  q  -- quit the application cleanly
  r  -- reset temporal buffer and smoothing history

Failure handling
-------------------------------
- No pose detected     : skeleton not drawn; buffer not updated.
- Incomplete landmarks : array rejected (len(lms) != 33 check).
- Near-zero torso      : scale clamped to 1e-4 (no division by zero).
- Insufficient history : prediction suppressed until MIN_BUFFER_FRAMES=3.
- Webcam unavailable   : error message, clean exit.
- Prediction errors    : warning printed, frame skipped, loop continues.
- MediaPipe errors     : warning printed, frame skipped, loop continues.

Limitations
-------------------------------
1. LOSO evaluation shows mean accuracy 66.26% +/- 11.34% on unseen subjects.
   Live predictions on new subjects may show similar or lower accuracy,
   especially for standing/walking confusion and subjects with partial
   lower-body visibility.

2. Standing and walking are the most easily confused activities (LOSO result).

3. Camera must capture full body including feet for best accuracy.
   Partial visibility of lower body significantly degrades standing prediction.

4. Only one subject at a time (num_poses=1 in MediaPipe config).

5. Temporal features assume approx 0.15 s frame spacing from training;
   real webcam frame rate may differ, causing slight feature magnitude drift.

6. The model was trained on 4 subjects only. Diversity of training data
   is limited.

DEPLOYMENT DISCLAIMER
-------------------------------
This is a demonstration application trained on the full 4-subject dataset.
Its live accuracy MUST NOT be reported as subject-independent generalization.
Subject-independent performance is documented in:
  results/loso_evaluation_report.txt

  Random Forest Mean Accuracy : 66.26% +/- 11.34%
  Random Forest Mean Macro F1 : 64.84% +/- 13.13%
========================================================
"""


def write_report():
    os.makedirs("results", exist_ok=True)
    with open(REPORT_FILE, "w") as fh:
        fh.write(REPORT_TEXT)
    print(f"[REPORT] Written: {REPORT_FILE}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    write_report()
    verify_only = "--verify" in sys.argv
    run(verify_only=verify_only)
