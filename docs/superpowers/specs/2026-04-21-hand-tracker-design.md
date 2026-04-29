# Hand Tracker — Design Spec
**Date:** 2026-04-21
**Status:** Approved

## Overview

A real-time hand tracker in Python that opens the default webcam, detects up to 2 hands per frame, and overlays 21 landmark dots per hand on the live video feed. The user presses `Q` to exit.

## Architecture

Single-file implementation (`hand_tracker.py`) with a standard capture → detect → render → display loop. No classes required — the script is short and linear enough to be a module-level main loop.

### Dependencies
- `opencv-python` — webcam capture, frame rendering, window display
- `mediapipe` — hand landmark detection (Google's Hands solution)

### Component breakdown

| Component | Responsibility |
|-----------|---------------|
| Webcam capture | `cv2.VideoCapture(0)` reads frames; releases on exit |
| MediaPipe Hands | Detects up to 2 hands; returns 21 normalized (x, y, z) landmarks per hand |
| Landmark renderer | Scales normalized coords to pixel coords; draws filled circles via `cv2.circle` |
| Display loop | Shows annotated frame in OpenCV window; polls for `Q` keypress to exit |

### Detection configuration
- `max_num_hands=2`
- `min_detection_confidence=0.7`
- `min_tracking_confidence=0.5`

### Landmark rendering
- 21 dots per hand, drawn as filled circles
- Distinct color per hand (e.g., green for hand 1, blue for hand 2)
- Dot radius: 5px

## Data Flow

```
Webcam frame (BGR)
  → Convert to RGB
  → mediapipe.Hands.process()
  → multi_hand_landmarks (list of 21 NormalizedLandmark per hand)
  → Scale to frame pixel dimensions
  → Draw cv2.circle for each landmark
  → Display annotated BGR frame in cv2 window
```

## Error Handling

- If webcam fails to open, print an error message and exit with code 1.
- If a frame read fails (returns False), break the loop cleanly.
- MediaPipe processes gracefully when no hands are in frame (returns None).

## Testing

File: `test_hand_tracker.py`

| Test | What it checks |
|------|---------------|
| `test_webcam_opens` | `cv2.VideoCapture(0).isOpened()` returns True |
| `test_mediapipe_blank_frame` | MediaPipe processes a black 480×640 frame without raising |
| `test_landmark_bounds` | Normalized landmark x, y values are in [0.0, 1.0] on a synthetic result |

Tests use `unittest` — no external test framework needed.

## Files

```
hand_tracker.py        # Main script
test_hand_tracker.py   # Unit tests
requirements.txt       # opencv-python, mediapipe
```
