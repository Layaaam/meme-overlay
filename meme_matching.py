"""
Phase 3 — Meme Library + Matching
===================================
Builds on Phase 2's classify_expression() output and adds:
  - A MemeLibrary that loads images from disk, organized by expression label
  - get_meme(label) matching with variety logic (avoids repeating the same
    meme twice in a row for a given label)
  - A side-by-side preview window: webcam feed on the left, matched meme
    on the right (full face-overlay compositing is Phase 4 — this phase
    is just about proving the matching logic works)

FOLDER SETUP (do this before running):
  memes/
    neutral/            <- put meme images here
    surprised/
    smiling/
    eyebrows_raised/
    head_tilt/

Drop any .png/.jpg/.gif-as-still meme images into the matching folder for
each label. Empty folders are fine — the panel will just show a "no memes
yet" placeholder for that label until you add some.

Controls:
  q       - quit
  s       - print current feature + label + matched meme snapshot to console

Setup:
  pip install opencv-python mediapipe

Run:
  python phase3_meme_matching.py
"""

import os
import time
import math
import random
import urllib.request
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

MEME_DIR = "memes"
LABELS = ["neutral", "surprised", "smiling", "eyebrows_raised", "head_tilt"]
MEME_PANEL_SIZE = (400, 400) 
RETRIGGER_COOLDOWN_SEC = 1.5

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

class MemeLibrary:
    """Loads meme images from memes/<label>/ folders and hands out matches
    with basic variety logic (avoid immediate repeat per label)."""

    VALID_EXT = (".png", ".jpg", ".jpeg", ".webp")

    def __init__(self, base_dir=MEME_DIR, labels=LABELS):
        self.base_dir = base_dir
        self.images = {label: [] for label in labels}  
        self.last_shown = {label: None for label in labels}
        self._scan()

    def _scan(self):
        for label in self.images:
            folder = os.path.join(self.base_dir, label)
            os.makedirs(folder, exist_ok=True)
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(self.VALID_EXT)
            ]
            self.images[label] = files

    def counts(self):
        return {label: len(files) for label, files in self.images.items()}

    def get_meme(self, label):
        """Returns a file path for the given label, avoiding the immediately
        previous pick when more than one option exists. Returns None if the
        label's folder is empty."""
        files = self.images.get(label, [])
        if not files:
            return None
        if len(files) == 1:
            choice = files[0]
        else:
            candidates = [f for f in files if f != self.last_shown[label]]
            choice = random.choice(candidates)
        self.last_shown[label] = choice
        return choice

    def rescan(self):
        """Call this if you add new images while the app is running."""
        self._scan()


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


def classify_expression(f):
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
    def __init__(self, window=SMOOTHING_WINDOW):
        self.history = deque(maxlen=window)

    def update(self, label):
        self.history.append(label)
        counts = Counter(self.history)
        stable_label, count = counts.most_common(1)[0]
        is_stable = count > len(self.history) / 2
        return stable_label if is_stable else None


def draw_landmarks(frame, landmarks, w, h):
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_meme_panel(meme_path, label, panel_size=MEME_PANEL_SIZE):
    """Returns a BGR image (panel_size) — either the loaded meme, scaled and
    letterboxed to fit, or a placeholder if no meme is available."""
    pw, ph = panel_size
    panel = np.full((ph, pw, 3), 30, dtype=np.uint8) 

    if meme_path is None:
        cv2.putText(panel, "no memes yet", (20, ph // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(panel, f"for '{label}'", (20, ph // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(panel, f"add images to memes/{label}/", (10, ph - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        return panel

    img = cv2.imread(meme_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        cv2.putText(panel, "failed to load", (20, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return panel

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3:4] / 255.0
        bg = np.full_like(bgr, 30)
        img = (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)

    ih, iw = img.shape[:2]
    scale = min(pw / iw, ph / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h))

    x_off = (pw - new_w) // 2
    y_off = (ph - new_h) // 2
    panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    cv2.putText(panel, label, (10, ph - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return panel


def draw_feature_overlay(frame, features, raw_label, stable_label, fps):
    x, y = 10, 25
    line_height = 22
    cv2.putText(frame, f"FPS: {fps:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2)
    for i, (key, val) in enumerate(features.items()):
        cv2.putText(frame, f"{key}: {val}", (x, y + (i + 1) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"raw: {raw_label}", (x, y + 7 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2)
    h_frame = frame.shape[0]
    cv2.putText(frame, f"EXPRESSION: {stable_label or '...'}", (10, h_frame - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def main():
    ensure_model()
    library = MemeLibrary()

    print("Meme library loaded. Image counts per label:")
    for label, count in library.counts().items():
        print(f"  {label}: {count}")
    print("(Add images to the memes/<label>/ folders and restart, or they'll")
    print(" show as placeholders until then.)\n")

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
    start_time = time.time()
    last_features = {}
    last_raw_label = "neutral"
    last_stable_label = "neutral"
    current_meme_path = None
    last_pick_time = 0.0
    last_committed_label = None

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
            last_raw_label = classify_expression(last_features)
            stable = smoother.update(last_raw_label)
            if stable:
                last_stable_label = stable

            now = time.time()
            label_changed = last_stable_label != last_committed_label
            cooldown_elapsed = (now - last_pick_time) > RETRIGGER_COOLDOWN_SEC
            if label_changed or (cooldown_elapsed and label_changed is False and current_meme_path is None):
                current_meme_path = library.get_meme(last_stable_label)
                last_committed_label = last_stable_label
                last_pick_time = now

            status = "Tracking active"
            status_color = (0, 200, 0)
        else:
            status = "No face detected"
            status_color = (0, 0, 255)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        cv2.putText(frame, status, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, status_color, 2)
        if last_features:
            draw_feature_overlay(frame, last_features, last_raw_label,
                                  last_stable_label, fps)

        meme_panel = draw_meme_panel(current_meme_path, last_stable_label)

        target_h = MEME_PANEL_SIZE[1]
        scale = target_h / h
        webcam_resized = cv2.resize(frame, (int(w * scale), target_h))
        combined = np.hstack([webcam_resized, meme_panel])

        cv2.imshow("Phase 3 - Meme Matching", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_features:
            print(f"Snapshot: {last_features}  ->  raw={last_raw_label}  "
                  f"stable={last_stable_label}  meme={current_meme_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()