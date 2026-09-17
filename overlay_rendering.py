"""
Phase 4 — Overlay Rendering
=============================
Builds on Phase 3's meme matching and composites the matched meme directly
onto the live webcam frame — either replacing your face region, or as a
corner "reaction bubble" — instead of showing it in a separate side panel.

This is the last CPU/logic-heavy phase before Phase 5 (virtual camera
output), which just takes whatever frame this phase produces and pipes it
into a virtual webcam device.

Controls:
  q       - quit
  o       - toggle overlay style: face-replacement <-> reaction bubble
  t       - toggle overlay on/off entirely (see your real face only)
  s       - print current feature + label + meme snapshot to console

Setup:
  pip install opencv-python mediapipe numpy

Run:
  python phase4_overlay_rendering.py
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
RETRIGGER_COOLDOWN_SEC = 1.5
FADE_FRAMES = 8          # frames over which a new overlay fades in
BUBBLE_SIZE_RATIO = 0.28  # bubble width as a fraction of frame width
FACE_BBOX_PADDING = 0.35  # extra margin around detected face for face-mode overlay

# --- Landmark indices (unchanged from earlier phases) ---
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
# Wider set of face-oval landmarks, used to compute the overlay bounding box
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

THRESHOLDS = {
    "mouth_open_high": 0.5,
    "eye_open_wide": 0.35,
    "eye_open_narrow": 0.28,
    "smile_ratio_high": 0.65,
    "brow_raise_high": 0.30,
    "head_tilt_deg": 8.0,
}

SMOOTHING_WINDOW = 6


# --------------------------------------------------------------------------
# Meme library (unchanged from Phase 3)
# --------------------------------------------------------------------------
class MemeLibrary:
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


# --------------------------------------------------------------------------
# Feature extraction + classification (unchanged from Phase 1/2)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Overlay rendering — the new part in this phase
# --------------------------------------------------------------------------
def get_face_bbox(landmarks, w, h, padding=FACE_BBOX_PADDING):
    """Bounding box around the face oval landmarks, with padding, clamped
    to frame bounds. Returns (x1, y1, x2, y2)."""
    xs = [landmarks[i].x * w for i in FACE_OVAL]
    ys = [landmarks[i].y * h for i in FACE_OVAL]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * padding
    x2 += bw * padding
    y1 -= bh * padding
    y2 += bh * padding
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def load_rgba(path):
    """Loads an image and guarantees a 4-channel BGRA result (adds a full
    opaque alpha channel if the source has none)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


def alpha_blend(background, overlay_rgba, x, y, target_w, target_h, opacity=1.0):
    """Resizes overlay_rgba to (target_w, target_h) and alpha-blends it onto
    background at position (x, y), clipping safely at frame edges."""
    if target_w <= 0 or target_h <= 0:
        return background

    resized = cv2.resize(overlay_rgba, (target_w, target_h))
    bg_h, bg_w = background.shape[:2]

    x1, y1 = x, y
    x2, y2 = x + target_w, y + target_h

    # Clip destination region to frame bounds
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(bg_w, x2)
    dst_y2 = min(bg_h, y2)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return background  # entirely off-frame

    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    region = resized[src_y1:src_y2, src_x1:src_x2]
    bgr = region[:, :, :3].astype(np.float32)
    alpha = (region[:, :, 3:4].astype(np.float32) / 255.0) * opacity

    bg_region = background[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    blended = bgr * alpha + bg_region * (1 - alpha)
    background[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
    return background


class OverlayState:
    """Tracks the current meme + fade-in progress so switching memes looks
    smooth rather than popping in instantly."""

    def __init__(self):
        self.current_path = None
        self.cached_rgba = None
        self.fade_frame = 0

    def set_meme(self, path):
        if path != self.current_path:
            self.current_path = path
            self.cached_rgba = load_rgba(path) if path else None
            self.fade_frame = 0

    def get_opacity(self):
        self.fade_frame = min(self.fade_frame + 1, FADE_FRAMES)
        return self.fade_frame / FADE_FRAMES


def draw_hud(frame, features, raw_label, stable_label, fps, overlay_mode, overlay_on):
    x, y = 10, 25
    line_height = 20
    cv2.putText(frame, f"FPS: {fps:.1f}  mode: {overlay_mode}  overlay: {'ON' if overlay_on else 'OFF'}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for i, (key, val) in enumerate(features.items()):
        cv2.putText(frame, f"{key}: {val}", (x, y + (i + 1) * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"raw: {raw_label}", (x, y + 7 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    h_frame = frame.shape[0]
    cv2.putText(frame, f"EXPRESSION: {stable_label or '...'}", (10, h_frame - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    ensure_model()
    library = MemeLibrary()

    print("Meme library loaded. Image counts per label:")
    for label, count in library.counts().items():
        print(f"  {label}: {count}")
    print()

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
    overlay_state = OverlayState()
    prev_time = time.time()
    start_time = time.time()
    last_features = {}
    last_raw_label = "neutral"
    last_stable_label = "neutral"
    last_committed_label = None
    overlay_mode = "bubble"  # "bubble" or "face"
    overlay_on = True

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

        face_bbox = None
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            last_features = compute_features(landmarks, w, h)
            last_raw_label = classify_expression(last_features)
            stable = smoother.update(last_raw_label)
            if stable:
                last_stable_label = stable

            if last_stable_label != last_committed_label:
                meme_path = library.get_meme(last_stable_label)
                overlay_state.set_meme(meme_path)
                last_committed_label = last_stable_label

            face_bbox = get_face_bbox(landmarks, w, h)
            status, status_color = "Tracking active", (0, 200, 0)
        else:
            status, status_color = "No face detected", (0, 0, 255)

        # --- Composite overlay onto frame ---
        if overlay_on and overlay_state.cached_rgba is not None:
            opacity = overlay_state.get_opacity()
            if overlay_mode == "face" and face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                frame = alpha_blend(frame, overlay_state.cached_rgba,
                                     x1, y1, x2 - x1, y2 - y1, opacity)
            elif overlay_mode == "bubble":
                bubble_w = int(w * BUBBLE_SIZE_RATIO)
                ih, iw = overlay_state.cached_rgba.shape[:2]
                bubble_h = int(bubble_w * ih / iw)
                margin = 15
                bx = w - bubble_w - margin
                by = h - bubble_h - margin
                # simple border so the bubble reads clearly against busy backgrounds
                cv2.rectangle(frame, (bx - 3, by - 3), (bx + bubble_w + 3, by + bubble_h + 3),
                              (255, 255, 255), 2)
                frame = alpha_blend(frame, overlay_state.cached_rgba,
                                     bx, by, bubble_w, bubble_h, opacity)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        cv2.putText(frame, status, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, status_color, 2)
        if last_features:
            draw_hud(frame, last_features, last_raw_label, last_stable_label,
                      fps, overlay_mode, overlay_on)

        cv2.imshow("Phase 4 - Overlay Rendering", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            overlay_mode = "face" if overlay_mode == "bubble" else "bubble"
            overlay_state.fade_frame = 0  # re-fade on mode switch
        elif key == ord('t'):
            overlay_on = not overlay_on
        elif key == ord('s') and last_features:
            print(f"Snapshot: {last_features}  ->  raw={last_raw_label}  "
                  f"stable={last_stable_label}  meme={overlay_state.current_path}  "
                  f"mode={overlay_mode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()