# Hand Tracker

Real-time hand tracking, body pose estimation, face recognition, and ASL fingerspelling — all running locally on a laptop webcam.

---

## Quick Start

```
setup.bat        # install dependencies
run.bat          # launch hand-tracking mode
```

---

## Modes

```
python hand_tracker.py                          # hand tracking (default)
python hand_tracker.py --mode face              # face recognition
python hand_tracker.py --mode enroll            # webcam enrollment
python hand_tracker.py --mode enroll --from-folder ./photos/   # batch enroll
python hand_tracker.py --mode audit             # confusable-pairs report
```

---

## Installation

### 1. Python environment

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Face recognition models (optional — required for `--mode face`)

```
python setup_models.py
```

This checks for InsightFace (antelopev2), AdaFace IR101, ElasticFace-Arc+, and YOLOv8n-face.
InsightFace and YOLOv8 download automatically on first run.
AdaFace and ElasticFace require manual ONNX download — instructions are printed by the script.

If models B or C are missing the system runs with a 2-model ensemble (A + B or A alone).
On CPU without a GPU, Model C is automatically skipped to keep inference fast.

---

## Hand Tracking Mode

```
python hand_tracker.py [options]
```

| Key | Action |
|-----|--------|
| Open hand → pinch | Cycle to next filter |
| Q | Quit |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | 0 | Camera device index |
| `--filter NAME` | — | Start with a specific filter active |
| `--list-filters` | — | Print all filter names and exit |
| `--resolution WxH` | 640x480 | Inference resolution |
| `--model-complexity N` | 1 | MediaPipe quality 0/1/2 |
| `--face` | off | Enable face recognition overlay |
| `--pose` | off | Enable body pose detection + gestures |
| `--blur` | off | Enable middle-finger blur effect |
| `--asl` | off | Enable ASL fingerspelling overlay |
| `--log-level LEVEL` | WARNING | DEBUG / INFO / WARNING / ERROR |

### Available Filters

| Name | Description |
|------|-------------|
| Normal | Passthrough |
| Inverted | Colour inversion |
| Hallucinogenic | Sine-wave distortion + hue shift |
| ASCII | Brightness-mapped ASCII art |
| Upside Down | 180° flip |
| Mosaic | Pixelate |
| Black and White | Greyscale |
| Flat 2D | Simplified flat-colour cartoon |

---

## Face Recognition Mode

```
python hand_tracker.py --mode face
```

Runs the dual-detector + 3-model ensemble pipeline. Bounding boxes are colour-coded by confidence:

| Colour | Status | Meaning |
|--------|--------|---------|
| Green | CONFIRMED | All 3 models agree, all scores above threshold |
| Cyan-yellow | SOFT_CONFIRMED | Majority (2/3) agree, strong scores |
| Orange | LOW_CONFIDENCE | Unanimous but weak scores, clear margin |
| Red | AMBIGUOUS | Models disagree or margins too thin |
| Grey | UNKNOWN | No match found or person not enrolled |

Names are displayed only after a track accumulates 3 consecutive consistent decisions (prevents flash-guesses on first detection).

---

## Enrollment

### From a folder of photos (recommended for 10+ people)

1. Drag photos into the `photos/` folder inside the project directory.
2. Name each photo after the person — `rui.jpg`, `alice_johnson.jpg`, etc.
3. Run:

```
python hand_tracker.py --mode enroll
```

The program detects the photos automatically. Type **a** at the first prompt to enroll the whole folder without stopping.

### Interactive webcam (one person at a time)

```
python hand_tracker.py --mode enroll
```

When `photos/` is empty, this falls through to webcam capture:

1. Enter the person's name at the terminal prompt.
2. Frame their face in the camera window.
3. Press **SPACE** to capture.
4. Type **y** (save), **n** (discard), or **r** (retake).

### Explicit folder path

```
python hand_tracker.py --mode enroll --from-folder C:\my\photos\
```

Supported photo formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### Gallery structure

```
gallery/
├── embeddings/
│   ├── alice_johnson_model_a.npy
│   ├── alice_johnson_model_b.npy
│   └── ...
└── metadata.json
```

---

## Audit Mode

```
python hand_tracker.py --mode audit
```

Computes the full N×N similarity matrix across all enrolled people and prints a confusable-pairs report:

```
=== CONFUSABLE PAIRS REPORT ===
WARNING: Alice Johnson <-> Carol Smith
   model_a: 0.41   model_b: 0.38   model_c: 0.39
   Recommendation: consider enrolling a second photo for both.

HIGH RISK: David Lee <-> Daniel Lee
   model_a: 0.52   model_b: 0.49   model_c: 0.47
   Recommendation: require unanimous consensus for these identities.
```

Pairs flagged as HIGH RISK (similarity > 0.45) are written to `metadata.json`. The arbitration layer automatically requires unanimous agreement for those identities during live recognition.

Run audit after completing all enrollments and before the live event.

---

## Architecture

```
Capture thread  →  raw_queue  →  Inference thread  →  result_queue  →  Render thread (main)
```

- **Capture**: reads frames from VideoCapture, measures FPS
- **Inference**: MediaPipe Hands + Pose on the inference thread (avoids blocking the UI)
- **Render pipeline**: FilterStage → FaceStage → ASLStage → RendererStage

The face recognition pipeline inside FaceStage:

```
DualDetector (RetinaFace + YOLOv8 cross-check)
    ↓ confirmed faces only
Quality gate (size, score, yaw, blur)
    ↓
EnsembleRecognizer (3 independent embedding models)
    ↓
Arbitrator (dynamic-weight voting: CONFIRMED / SOFT_CONFIRMED / LOW_CONFIDENCE / AMBIGUOUS / UNKNOWN)
    ↓
TemporalSmoother (15-frame window, 60% consensus, 3-frame init delay)
    ↓
RendererStage (colour-coded bounding boxes)
```

---

## Tests

```
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 85 tests should pass. No GPU or camera is required — heavy dependencies are stubbed.
