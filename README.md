# Meme Overlay

A real-time webcam filter that detects your facial expressions and overlays
a matching meme, designed to be piped into video calls (Google Meet,
Messenger, etc.) via a virtual camera.

## How it works

1. **Detection** (`detection.py`) — MediaPipe FaceLandmarker tracks
   468 facial landmarks per frame and derives features like mouth openness,
   smile ratio, eyebrow raise, eye openness, and head tilt.
2. **Classification** (`classification.py`) — turns those features
   into a discrete expression label (`neutral`, `surprised`, `smiling`,
   `eyebrows_raised`, `head_tilt`), with temporal smoothing to avoid
   flickering between labels.
3. **Meme matching** (`meme_matching.py`) — matches the detected
   expression to a meme image and previews it side-by-side with your
   webcam feed.
4. **Face overlay + virtual camera** (planned) — composites the meme
   directly onto your face and streams the output into a virtual camera
   so it appears as your webcam in any video call app.

## Setup

```bash
pip install -r requirements.txt
```

### ⚠️ Add your own meme images before running

This repo ships with **empty** meme folders — the actual meme images are
**not included** (and are gitignored) since they're not something to
commit to source control. Before running any script past classificaiton, you
need to manually add image files into the matching folders:

```
memes/
  neutral/            <- add meme image(s) here
  surprised/
  smiling/
  eyebrows_raised/
  head_tilt/
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`. You can add more than
one image per folder — the matcher will pick between them with basic
variety logic (avoiding the same image twice in a row).

If a folder is left empty, the app still runs fine — it just shows a
"no memes yet for `<label>`" placeholder for that expression until you
add images.

## Running

```bash
python detection.py   # feature extraction only, no classification
python classification.py   # + expression labels
python meme_matching.py    # + meme matching preview
```

Controls (all files): `q` to quit, `s` to print a snapshot of current
values to the console (useful for tuning thresholds in
`classification.py`).
