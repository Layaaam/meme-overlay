"""
Phase 2 — Expression Classification
=====================================
Builds on Phase 1's feature extraction (mouth_open, smile_ratio,
eyebrow_raise, eye_openness, head_tilt_deg) and adds a classify()
function that turns those numbers into a discrete expression label.

Thresholds below are derived directly from real snapshot data collected
in Phase 1 (5 expressions x ~3 snapshots each). See THRESHOLDS section
for the reasoning behind each cutoff, and CALIBRATION_NOTES for the
original data clusters they came from.

Controls:
  q       - quit
  s       - print current feature + label snapshot to console

Setup:
  pip install opencv-python mediapipe

Run:
  python phase2_classification.py
"""

import os
import time
import math
import urllib.request
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
LEFT_EYEBROW = 105
RIGHT_EYEBROW = 334
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

THRESHOLDS = {
    "mouth_open_high": 0.5, 
    "eye_open_wide": 0.35,  
    "eye_open_narrow": 0.28,
    "smile_ratio_high": 0.65,
    "brow_raise_high": 0.30, 
    "head_tilt_deg": 8.0,    
}

SMOOTHING_WINDOW = 6 


def classify_expression(f):
    """
    Takes a feature dict (from compute_features) and returns a single
    expression label based on thresholds derived from real snapshot data.
    Checked in priority order — most distinctive signal first.
    """
    if abs(f["head_tilt_deg"]) > THRESHOLDS["head_tilt_deg"]:
        return "head_tilt"

    if f["mouth_open"] > THRESHOLDS["mouth_open_high"]:
        return "surprised"

    if (f["eye_openness"] < THRESHOLDS["eye_open_narrow"]
            and f["smile_ratio"] > THRESHOLDS["smile_ratio_high"]):
        return "smiling"

    if (f["eye_openness"] > THRESHOLDS["eye_open_wide"]
            and f["eyebrow_raise"] > THRESHOLDS["brow_raise_high"]):
        return "eyebrows_raised"

    return "neutral"


class LabelSmoother:
    """Requires a majority label across a rolling window before it's
    reported as 'stable' — prevents single-frame flicker/misfires."""

    def __init__(self, window=SMOOTHING_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, label):
        self.history.append(label)
        counts = Counter(self.history)
        stable_label, count = counts.most_common(1)[0]
        is_stable = count > len(self.history) / 2
        return stable_label if is_stable else None


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model (one-time, ~4MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded:", MODEL_PATH)


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    p = [get_point(landmarks, i, w, h) for i in eye_idx]
    vertical1 = euclidean(p[1], p[5])
    vertical2 = euclidean(p[2], p[4])
    horizontal = euclidean(p[0], p[3])
    if horizontal == 0:
        return 0
    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_features(landmarks, w, h):
    face_width = euclidean(
        get_point(landmarks, LEFT_EYE_OUTER, w, h),
        get_point(landmarks, RIGHT_EYE_OUTER, w, h),
    )
    if face_width == 0:
        face_width = 1

    mouth_vert = euclidean(
        get_point(landmarks, MOUTH_TOP, w, h),
        get_point(landmarks, MOUTH_BOTTOM, w, h),
    )
    mouth_horiz = euclidean(
        get_point(landmarks, MOUTH_LEFT, w, h),
        get_point(landmarks, MOUTH_RIGHT, w, h),
    )
    mouth_aspect_ratio = mouth_vert / mouth_horiz if mouth_horiz else 0
    smile_ratio = mouth_horiz / face_width

    left_brow_raise = euclidean(
        get_point(landmarks, LEFT_EYEBROW, w, h),
        get_point(landmarks, LEFT_EYE_TOP, w, h),
    ) / face_width
    right_brow_raise = euclidean(
        get_point(landmarks, RIGHT_EYEBROW, w, h),
        get_point(landmarks, RIGHT_EYE_TOP, w, h),
    ) / face_width
    eyebrow_raise = (left_brow_raise + right_brow_raise) / 2.0

    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    eye_openness = (left_ear + right_ear) / 2.0

    left_outer = get_point(landmarks, LEFT_EYE_OUTER, w, h)
    right_outer = get_point(landmarks, RIGHT_EYE_OUTER, w, h)
    dx = right_outer[0] - left_outer[0]
    dy = right_outer[1] - left_outer[1]
    head_tilt_deg = math.degrees(math.atan2(dy, dx))

    return {
        "mouth_open": round(mouth_aspect_ratio, 3),
        "smile_ratio": round(smile_ratio, 3),
        "eyebrow_raise": round(eyebrow_raise, 3),
        "eye_openness": round(eye_openness, 3),
        "head_tilt_deg": round(head_tilt_deg, 1),
    }


def draw_landmarks(frame, landmarks, w, h):
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_panel(frame, features, label, stable_label, fps):
    x, y = 10, 25
    line_height = 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    for i, (key, val) in enumerate(features.items()):
        text = f"{key}: {val}"
        cv2.putText(frame, text, (x, y + (i + 1) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"raw: {label}", (x, y + 7 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    h_frame = frame.shape[0]
    display_label = stable_label if stable_label else "..."
    cv2.putText(frame, f"EXPRESSION: {display_label}", (10, h_frame - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try cv2.VideoCapture(1) instead.")
        return

    smoother = LabelSmoother()
    prev_time = time.time()
    last_features = {}
    last_label = "neutral"
    last_stable = "neutral"
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            draw_landmarks(frame, landmarks, w, h)
            last_features = compute_features(landmarks, w, h)
            last_label = classify_expression(last_features)
            stable = smoother.update(last_label)
            if stable:
                last_stable = stable
            status = "Tracking active"
            status_color = (0, 200, 0)
        else:
            status = "No face detected"
            status_color = (0, 0, 255)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        cv2.putText(frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, status_color, 2)
        if last_features:
            draw_panel(frame, last_features, last_label, last_stable, fps)

        cv2.imshow("Phase 2 - Expression Classification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_features:
            print(f"Snapshot: {last_features}  ->  raw={last_label}  stable={last_stable}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()