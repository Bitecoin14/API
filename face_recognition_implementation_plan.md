# Face Recognition Mode — Implementation Plan

**Status:** ALL 5 SESSIONS COMPLETE — 2026-04-28

---

## Session Checkpoint

| Session | Description | Status |
|---------|-------------|--------|
| 1 | Detection layer (`face/detector.py`), attribute layer (`face/attributes.py`), config constants, requirements | ✅ Done |
| 2 | Single-model recognition: `face/gallery.py`, `face/enrollment.py`, `--mode enroll`, `--mode face` | ✅ Done |
| 3 | Ensemble: `face/models.py`, `face/arbitration.py`, colour-coded renderer, 3-model voting | ✅ Done |
| 4 | Temporal smoothing (`face/temporal.py`), IoU tracker, `--mode audit`, confusable-pair flagging | ✅ Done |
| 5 | CPU fallback (2-model), `--from-folder` batch enroll, `setup_models.py`, `README.md`, edge-case handling | ✅ Done |

---

## What Was Built

| File | Responsibility |
|------|----------------|
| `face/__init__.py` | Package marker + public re-exports |
| `face/types.py` | `RecognitionStatus`, `DetectedFace`, `ModelVote`, `RecognitionResult`, `FaceAttributes` dataclasses |
| `face/detector.py` | `DualDetector` — RetinaFace + YOLOv8 consensus, quality gate |
| `face/attributes.py` | Glasses detection, makeup estimation, quality score |
| `face/gallery.py` | `FaceGallery` — .npy per-model storage, cosine search, similarity matrix |
| `face/models.py` | `EnsembleRecognizer` — Models A/B/C, L2-normalize, CPU 2-model fallback |
| `face/arbitration.py` | `Arbitrator` — dynamic-weight voting, 5 decision rules |
| `face/temporal.py` | `TemporalSmoother` — IoU tracker, 15-frame window, 60% consensus |
| `face/enrollment.py` | `enroll_from_capture`, `enroll_from_folder`, `run_audit` |
| `stages/face_stage.py` | Pipeline stage — ensemble primary, LBPH legacy fallback |
| `core/config.py` | All face recognition constants added |
| `core/context.py` | `face_results` field added to `FrameContext` |
| `stages/renderer.py` | `_draw_face_box`, `_draw_face_hud`, `draw_enrollment_overlay` added |
| `hand_tracker.py` | `--mode hand/face/enroll/audit`, `--from-folder` wired |
| `setup_models.py` | Model download + verification script |
| `README.md` | Complete usage instructions |
| `tests/test_face_core.py` | 23 unit tests: gallery, temporal smoother, arbitrator |
| `tests/test_face_integration.py` | 6 tests: FaceStage, enrollment helpers, audit |

**All 65 tests pass** (no GPU or camera required).

---

## Context

This document extends the existing `hand_tracker/` project (OpenCV + MediaPipe, real-time webcam pipeline with threaded capture, gesture classification, motion tracking, and renderer). A new **face recognition mode** is added as a switchable mode alongside the existing hand tracking mode. The system must run in real-time at the prom, handle 100+ enrolled people from single photos, and cope with makeup, glasses, and dramatic appearance changes.

---

## Design Constraints

1. **Real-time**: recognition pipeline must sustain ≥ 10 fps on a laptop GPU (RTX 3060+) or ≥ 5 fps on CPU
2. **No per-frame logging**: the system does NOT save or log every recognition event — it only persists data during enrollment
3. **Confirmation loop**: after each enrollment, the system asks the operator "Is this [Name]? (y/n)" and only commits if confirmed
4. **Single-photo enrollment**: each person provides exactly one clear, frontal photo
5. **Ensemble consensus**: three independent models must agree before a name is displayed
6. **Offline / no API**: all models run locally, no cloud calls, no paid dependencies

---

## Architecture: How It Fits Into hand_tracker/

```
hand_tracker/
├── main.py                 # Mode switcher: --mode hand | face | enroll
├── config.py               # Existing + new face recognition constants
├── capture.py              # UNCHANGED — threaded webcam capture
├── tracker.py              # UNCHANGED — hand landmark tracking
├── gestures.py             # UNCHANGED — gesture classification
├── motion.py               # UNCHANGED — motion tracking
├── renderer.py             # EXTENDED — add face bbox + name overlay
├── face/                   # NEW — entire face recognition subsystem
│   ├── __init__.py
│   ├── detector.py         # Dual-detector consensus (RetinaFace + YOLO)
│   ├── models.py           # Ensemble: load and query 3 embedding models
│   ├── gallery.py          # Gallery storage, search, save/load
│   ├── enrollment.py       # Enrollment pipeline with confirmation
│   ├── arbitration.py      # Consensus engine (3-model voting)
│   ├── attributes.py       # Glasses/makeup/quality scoring
│   └── temporal.py         # Track-level smoothing across frames
├── gallery/                # Persisted enrollment data (created at runtime)
│   ├── embeddings/         # .npy files per person per model
│   └── metadata.json       # Name → attributes mapping
├── requirements.txt        # EXTENDED with insightface, onnxruntime, etc.
└── README.md               # EXTENDED with face mode docs
```

---

## Mode Switching in main.py

The existing `main.py` runs the hand tracking loop. Add a CLI argument:

```
python main.py --mode hand      # Default: existing hand tracking
python main.py --mode face      # Real-time face recognition at the prom
python main.py --mode enroll    # Enrollment mode: register new people
```

All three modes share the same `capture.py` threaded frame source. The mode determines which processing pipeline runs on each frame.

---

## Phase 1 — Detection Layer (`face/detector.py`)

### What It Does

Detects faces in each frame using two independent architectures. Only faces confirmed by both proceed to recognition.

### Components

- **Primary**: InsightFace RetinaFace (comes bundled with the `insightface` package)
- **Secondary**: Ultralytics YOLOv8-face (lightweight, different architecture)

### Consensus Protocol

1. Both detectors run on the same frame
2. For each RetinaFace detection, find the YOLOv8 detection with highest IoU overlap
3. If IoU ≥ 0.5 → face is **confirmed** — use RetinaFace bbox + landmarks (higher precision)
4. If no overlap → face is **suppressed** — likely a false positive (poster, decoration, phone screen)
5. Log suppressed detections nowhere — they are silently dropped

### Quality Gate (Applied to Confirmed Faces Only)

Each confirmed face is scored before proceeding:

| Check             | Threshold          | Action if failed                      |
| ----------------- | ------------------ | ------------------------------------- |
| Face width        | ≥ 80px             | Skip — too far away                   |
| Detection score   | ≥ 0.65             | Skip — unreliable detection           |
| Yaw angle         | ≤ 35°              | Skip — profile view, embedding unreliable |
| Blur (Laplacian)  | variance ≥ 50      | Skip — motion blur                    |

Faces that fail the quality gate are drawn with a grey bounding box and "?" label — never forced into a match.

### Real-Time Optimization

- Run RetinaFace at 640×640 input resolution (not full frame) — InsightFace handles the resize internally
- Run YOLOv8 at 320×320 for the cross-check — it only needs to confirm presence, not precise landmarks
- Both detectors share the same pre-resized frame buffer — no duplicate resize
- Expected latency: ~8ms total for both detectors on GPU, ~35ms on CPU

---

## Phase 2 — Attribute Layer (`face/attributes.py`)

### What It Does

Extracts non-identity metadata from each detected face. This layer **never determines identity** — it adjusts confidence thresholds in the arbitration layer.

### Modules

1. **Glasses Detection**: binary (glasses / no glasses) using InsightFace's built-in attribute head (no extra model needed — `face.embedding_norm` and gender/age attributes are already extracted during the embedding pass)
2. **Makeup Estimation**: compare color histogram of the periorbital + lip region against neutral-skin baselines. Three levels: none / light / heavy. Uses OpenCV histogram comparison, no ML model.
3. **Quality Score**: weighted composite of blur, illumination, face size, and pose deviation. Single float 0.0–1.0.

### How It Affects Recognition

The attribute layer outputs a dict per face:

```python
{
    "quality": 0.82,
    "glasses_detected": True,
    "glasses_change": True,    # Compared against enrollment metadata
    "heavy_makeup": True,
    "yaw_degrees": 12.3
}
```

This dict is passed to the Arbitration Layer, which uses it to adjust model weights (see Phase 4).

### Real-Time Cost

Near-zero. Glasses detection reuses the InsightFace embedding pass. Makeup estimation is a histogram computation (~0.5ms). Quality score is arithmetic on already-computed values.

---

## Phase 3 — Recognition Layer / Ensemble (`face/models.py`)

### Three Independent Models

Each model loads once at startup and stays in memory. Each produces a 512-dimensional embedding from the aligned 112×112 face crop.

#### Model A: InsightFace `antelopev2` (ArcFace / Glint360K)

- Role: **Anchor** — highest overall accuracy, default highest weight
- Backbone: ResNet100
- Training data: Glint360K (360K identities, 93M images)
- Why: best-in-class on clean, well-lit faces

#### Model B: AdaFace (IR-ResNet101 / WebFace12M)

- Role: **Quality specialist** — its weight increases when the quality score is low
- Backbone: IR-ResNet101
- Training data: WebFace12M (600K identities, 12M images)
- Why: specifically designed to handle low-quality inputs by adapting loss margin to image quality

#### Model C: ElasticFace-Arc+ (ResNet100 / MS1MV3)

- Role: **Robustness specialist** — its weight increases when glasses change or heavy makeup is detected
- Backbone: ResNet100
- Training data: MS1MV3 (93K identities, 5.1M images)
- Why: elastic margin during training prevents overfitting to specific feature distributions, handles domain shift (makeup, accessories)

### Per-Model Query Protocol

Each model independently:

1. Receives the same aligned face crop
2. Extracts embedding → L2-normalizes
3. Computes cosine similarity against its own gallery column (each person has separate embeddings per model)
4. Returns: `(best_match_name, best_score, runner_up_name, runner_up_score)`

Models do NOT see each other's outputs. Independence is critical.

### Real-Time Optimization

- All three models run sequentially on the same GPU (batch size 1 per face). Total embedding time: ~15ms on GPU, ~80ms on CPU for all three.
- Gallery search is brute-force cosine similarity via numpy — at 100–500 people with 512-d vectors, this is < 1ms.
- Total per-face recognition latency: **~20ms GPU, ~90ms CPU**. At 5 faces per frame, that's ~100ms GPU / ~450ms CPU — still real-time on GPU, borderline on CPU.

### CPU Fallback Strategy

If running on CPU and the per-frame time exceeds 200ms:
- Drop Model C (ElasticFace) — it's the robustness specialist, least critical for clean prom photos
- Run only Model A + Model B — require unanimous agreement between the two (stricter than majority vote)
- This cuts recognition time by ~33%

---

## Phase 4 — Arbitration Layer (`face/arbitration.py`)

### What It Does

Receives three independent votes and resolves them into a single decision. This is the core logic that prevents misidentification.

### Input Per Face

```python
votes = [
    {"model": "A", "match": "Alice", "score": 0.52, "runner_up": "Carol", "ru_score": 0.31},
    {"model": "B", "match": "Alice", "score": 0.48, "runner_up": "Carol", "ru_score": 0.29},
    {"model": "C", "match": "Carol", "score": 0.44, "runner_up": "Alice", "ru_score": 0.43},
]
attributes = {"quality": 0.78, "glasses_change": True, "heavy_makeup": True}
```

### Dynamic Weight Calculation

```
Base weights:     w_A = 0.40,  w_B = 0.30,  w_C = 0.30

If quality < 0.6:    w_B += 0.10,  w_A -= 0.10   (AdaFace handles low quality)
If glasses_change OR heavy_makeup:  w_C += 0.10,  w_A -= 0.10   (ElasticFace handles distribution shift)

Re-normalize so weights sum to 1.0.
```

### Decision Rules

#### Rule 1: Unanimous + Strong → CONFIRMED

All three models return the same name AND all scores ≥ their per-model threshold.

```
Model A threshold: 0.45
Model B threshold: 0.42
Model C threshold: 0.43
```

Display: **green box**, name shown, high confidence.

#### Rule 2: Majority (2/3) + Strong → SOFT CONFIRMED

Two models agree, both above threshold. Third model:
- Returns "Unknown" → accept the majority
- Returns a **different person** → escalate to Rule 4

Display: **yellow box**, name shown, medium confidence.

#### Rule 3: Unanimous + Weak → LOW CONFIDENCE

All three agree but one or more scores below threshold. Check margin (best vs. runner-up) for each model:
- All margins > 0.07 → accept with warning
- Any margin ≤ 0.07 → escalate to Rule 4

Display: **orange box**, name shown with "?" suffix.

#### Rule 4: Disagreement → AMBIGUOUS

No majority. Two or more different names.
- Compute pairwise cosine similarity between the three query embeddings
- If embeddings are consistent (cos > 0.85 pairwise) → gallery ambiguity (two similar-looking people). Show both candidate names.
- If embeddings inconsistent (cos < 0.85) → face quality too poor. Show "Unknown".

Display: **red box**, "?" or both names shown.

#### Rule 5: All Below Threshold → UNKNOWN

No model found a match. Person likely not enrolled.

Display: **grey box**, "Unknown".

### What Is NOT Logged

None of these decisions are written to disk during real-time recognition. The overlay is drawn on the frame and discarded. No audit log, no JSON, no database write. This keeps the pipeline zero-I/O during recognition.

---

## Phase 5 — Temporal Smoothing (`face/temporal.py`)

### Why

A person walking through the prom is recognized across dozens of frames. Single-frame errors (blurry frame, bad lighting angle) should not cause the displayed name to flicker.

### Implementation

1. Assign a persistent **track ID** to each face using a simple IoU-based tracker (no DeepSORT needed at prom scale — faces don't move that fast in a social setting)
2. For each track, maintain a `deque(maxlen=15)` of recent identity decisions
3. Displayed identity = **mode (most common)** of the last 15 decisions
4. If no identity has > 60% of the votes in the window → display "?" (unstable)
5. When a track is first created, require 3 consecutive identical decisions before showing the name (prevents flash-guesses)

### Track Lifecycle

- **Created** when a confirmed face appears with no IoU > 0.4 overlap with existing tracks
- **Updated** each frame with the new face's bbox and identity decision
- **Destroyed** after 30 consecutive frames with no matching face (person walked away)

### Real-Time Cost

Negligible — it's a dict of deques, updated with O(1) operations per face per frame.

---

## Phase 6 — Enrollment Pipeline (`face/enrollment.py`)

### This Is the Only Part That Writes to Disk

Enrollment is a separate mode (`--mode enroll`), run before the prom. It does NOT run during live recognition.

### Enrollment Flow

```
1. Operator starts: python main.py --mode enroll
2. System opens webcam feed with overlay: "ENROLLMENT MODE"
3. Operator types person's name in the terminal
4. Operator frames the person's face in the camera
5. Operator presses SPACE to capture

6. System:
   a. Detects face (dual-detector consensus)
   b. Quality-checks — if too low, displays reason and asks to retry
   c. Augments the single capture into 5 variants:
      - Original
      - Horizontal flip
      - Brightness +15%
      - Brightness -15%
      - Mild Gaussian blur (σ=0.5)
   d. Extracts embeddings from all 5 variants using ALL THREE models
   e. Computes per-model mean embedding (L2-normalized)
   f. Extracts attributes (glasses, estimated skin tone for makeup baseline)

7. System displays the cropped face and asks:
   "Enrolled [Name]. Is this correct? (y/n/r)"
   - y → commit to gallery, save to disk
   - n → discard, re-enter name
   - r → retake photo, keep name

8. On commit:
   - Save per-model mean embeddings as .npy files in gallery/embeddings/
   - Update gallery/metadata.json with name, timestamp, attributes
   - Run similarity check against ALL existing enrolled people
   - If any existing person has cosine similarity > 0.35 with the new person
     in ANY model → display WARNING:
     "[Name] looks similar to [OtherName] (score: 0.41 in Model A).
      Consider enrolling a second photo for both."

9. Operator can also enroll from a folder of pre-taken photos:
   python main.py --mode enroll --from-folder ./photos/
   Each file named "FirstName_LastName.jpg" is enrolled automatically,
   with the confirmation prompt shown for each.
```

### What Gets Saved to Disk

```
gallery/
├── embeddings/
│   ├── alice_johnson_model_a.npy     # 512-d float32 vector
│   ├── alice_johnson_model_b.npy
│   ├── alice_johnson_model_c.npy
│   ├── bob_smith_model_a.npy
│   ├── bob_smith_model_b.npy
│   ├── bob_smith_model_c.npy
│   └── ...
└── metadata.json
```

`metadata.json` structure:

```json
{
  "alice_johnson": {
    "display_name": "Alice Johnson",
    "enrolled_at": "2026-06-10T14:22:00Z",
    "enrollment_quality": 0.92,
    "has_glasses": false,
    "enrollment_photo_hash": "sha256:abc123..."
  },
  "bob_smith": {
    "display_name": "Bob Smith",
    "enrolled_at": "2026-06-10T14:23:15Z",
    "enrollment_quality": 0.88,
    "has_glasses": true,
    "enrollment_photo_hash": "sha256:def456..."
  }
}
```

### Pre-Prom Similarity Audit

After all people are enrolled, run:

```
python main.py --mode audit
```

This computes the full N×N similarity matrix for each model and outputs a report:

```
=== CONFUSABLE PAIRS REPORT ===
WARNING: Alice Johnson ↔ Carol Smith
   Model A: 0.41   Model B: 0.38   Model C: 0.39
   Recommendation: Enroll second photo for both, or flag for operator awareness

WARNING: David Lee ↔ Daniel Lee
   Model A: 0.52   Model B: 0.49   Model C: 0.47
   Recommendation: HIGH RISK — likely siblings. Require unanimous consensus for these identities.

All other pairs: max similarity < 0.30 ✓
```

The audit mode updates `metadata.json` to flag confusable pairs. During live recognition, the Arbitration Layer requires **unanimous agreement** (not majority) for flagged identities.

---

## Phase 7 — Renderer Integration (`renderer.py` Extension)

### Existing Renderer

The current `Renderer` class draws hand landmarks, bounding boxes, gesture labels, velocity vectors, and an FPS HUD. It operates on the frame in-place.

### New Methods for Face Mode

Add to the existing `Renderer` class:

- `draw_face_box(frame, bbox, name, status, confidence)` — draws a colored bounding box with the identity label
  - CONFIRMED: green box, white name
  - SOFT_CONFIRMED: yellow box, white name
  - LOW_CONFIDENCE: orange box, name + "?"
  - AMBIGUOUS: red box, "?" or two candidate names
  - UNKNOWN: grey box, "Unknown"
- `draw_face_hud(frame, num_faces, num_identified, fps)` — top-left HUD showing face count, identification rate, and FPS
- `draw_enrollment_overlay(frame, name, status_text)` — enrollment-mode-specific overlay with the current name being enrolled and status messages

### Color Coding Logic

The color is determined by the Arbitration Layer's `status` field, not by the confidence score directly. This prevents confusion — a high score with a SOFT_CONFIRMED status still shows yellow, not green.

---

## Phase 8 — Config Extension (`config.py`)

Add these constants to the existing config:

```python
# === Face Recognition Config ===

# Detection
FACE_MIN_SIZE = 80                    # Minimum face width in pixels
FACE_MIN_DET_SCORE = 0.65            # Minimum detection confidence
FACE_MAX_YAW = 35.0                  # Maximum yaw angle (degrees)
FACE_MIN_BLUR = 50                   # Minimum Laplacian variance

# Recognition thresholds (per model)
MODEL_A_THRESHOLD = 0.45
MODEL_B_THRESHOLD = 0.42
MODEL_C_THRESHOLD = 0.43
AMBIGUOUS_MARGIN = 0.07              # If top-2 within this margin → ambiguous

# Arbitration weights
BASE_WEIGHT_A = 0.40
BASE_WEIGHT_B = 0.30
BASE_WEIGHT_C = 0.30

# Temporal smoothing
TRACK_WINDOW_SIZE = 15               # Rolling window for identity voting
TRACK_MIN_CONSENSUS = 0.60           # 60% agreement required
TRACK_INIT_FRAMES = 3                # Require N identical before first display
TRACK_TIMEOUT_FRAMES = 30            # Destroy track after N frames without match

# Enrollment
GALLERY_DIR = "gallery"
CONFUSABLE_THRESHOLD = 0.35          # Similarity above this → warn during enrollment
ENROLLMENT_AUGMENT = True

# Performance
FACE_DETECTOR_INPUT_SIZE = 640       # RetinaFace input
YOLO_CROSS_CHECK_SIZE = 320          # YOLOv8 cross-check input
CPU_FALLBACK_TWO_MODELS = True       # Drop Model C on CPU to maintain FPS
```

---

## Phase 9 — Requirements & Dependencies

Add to `requirements.txt`:

```
# Existing
opencv-python>=4.9.0
mediapipe>=0.10.30
numpy>=1.26.0

# Face recognition — NEW
insightface>=0.7.3
onnxruntime-gpu>=1.17.0              # Or onnxruntime for CPU-only
ultralytics>=8.1.0                   # YOLOv8-face
scikit-learn>=1.4.0                  # cosine_similarity utility
```

### Model Downloads (First Run)

- `antelopev2`: downloaded automatically by InsightFace on first use (~300MB)
- AdaFace IR101: download ONNX weights from the AdaFace GitHub release page and place in `models/adaface_ir101.onnx`
- ElasticFace-Arc+: download ONNX weights from the ElasticFace GitHub release page and place in `models/elasticface_arc.onnx`

Create a `setup_models.py` script that downloads and verifies all three model files with SHA256 checksums.

---

## Implementation Order (for Claude Code sessions)

### Session 1: Foundation

**Prompt for Claude Code:**

> Add face detection to the hand_tracker project. Create `face/detector.py` with a `DualDetector` class that runs RetinaFace (via insightface) and YOLOv8-face (via ultralytics) on the same frame and returns only faces confirmed by both (IoU ≥ 0.5). Add a quality gate that filters by face size (≥80px), detection score (≥0.65), yaw (≤35°), and blur (Laplacian variance ≥50). Create `face/attributes.py` with glasses detection (from insightface attributes) and a quality score function. Update `config.py` with all face-related constants. Update `requirements.txt`. The mode is selected via `--mode hand|face` in main.py. In face mode, just draw green boxes around detected faces with the quality score — no recognition yet.

### Session 2: Single-Model Recognition

**Prompt:**

> Add face recognition using InsightFace antelopev2. Create `face/gallery.py` with a `FaceGallery` class that stores L2-normalized embeddings as .npy files and searches by cosine similarity. Create `face/enrollment.py` with an enrollment pipeline: capture face → augment (flip, brightness ±15%, blur) → extract 5 embeddings → compute mean → save. During enrollment, show the cropped face and prompt "Is this [Name]? (y/n/r)" in the terminal — only save on 'y'. Add `--mode enroll` to main.py. In `--mode face`, recognize detected faces against the gallery and display names on the overlay. No logging during recognition — overlay only.

### Session 3: Ensemble (Add Models B and C)

**Prompt:**

> Extend the face recognition system to use three models. Create `face/models.py` with an `EnsembleRecognizer` class that loads antelopev2, AdaFace IR101, and ElasticFace-Arc+ as three independent embedding extractors. Each model has its own gallery column (separate .npy files per person per model). Create `face/arbitration.py` with the consensus engine: unanimous+strong → CONFIRMED, majority+strong → SOFT_CONFIRMED, unanimous+weak → LOW_CONFIDENCE, disagreement → AMBIGUOUS, all below threshold → UNKNOWN. Use dynamic weight adjustment based on quality score, glasses change, and makeup level from the attribute layer. Update the renderer to color-code bounding boxes by status (green/yellow/orange/red/grey). Still no disk writes during recognition.

### Session 4: Temporal Smoothing + Audit

**Prompt:**

> Add temporal smoothing to face recognition. Create `face/temporal.py` with a simple IoU-based face tracker that assigns persistent track IDs across frames. Each track maintains a deque of the last 15 identity decisions. Displayed identity = mode of the deque (require ≥60% agreement). New tracks must have 3 consecutive identical decisions before showing a name. Tracks are destroyed after 30 frames without a matching face. Also add `--mode audit` that loads the gallery, computes the full N×N similarity matrix for each model, and prints a confusable pairs report (any pair with similarity > 0.35 in any model). Flag confusable pairs in metadata.json so the arbitration layer requires unanimous agreement for those identities.

### Session 5: Polish + CPU Fallback

**Prompt:**

> Polish the face recognition system. Add CPU fallback: if `CPU_FALLBACK_TWO_MODELS` is True in config and no GPU is available, only load Models A and B (skip ElasticFace) and require unanimous agreement between the two. Add `--mode enroll --from-folder ./photos/` to batch-enroll from a folder of images named "FirstName_LastName.jpg". Add a `setup_models.py` script that downloads all three model ONNX files with SHA256 verification. Update README.md with complete instructions for enrollment, audit, and live recognition modes. Ensure the system gracefully handles: empty gallery, no faces detected, GPU out of memory (falls back to CPU mid-session).

---

## Performance Budget

| Stage                        | GPU (RTX 3060) | CPU (i7-12700) | Notes                                |
| ---------------------------- | -------------- | -------------- | ------------------------------------ |
| Frame capture (threaded)     | 0ms            | 0ms            | Background thread, non-blocking      |
| RetinaFace detection         | 5ms            | 25ms           | 640×640 input                        |
| YOLOv8 cross-check           | 3ms            | 12ms           | 320×320 input                        |
| Quality gate + attributes    | 1ms            | 1ms            | Arithmetic only                      |
| Model A embedding            | 5ms            | 25ms           | ResNet100 forward pass               |
| Model B embedding            | 5ms            | 28ms           | IR-ResNet101 forward pass            |
| Model C embedding            | 5ms            | 25ms           | ResNet100 forward pass               |
| Gallery search (3 models)    | < 1ms          | < 1ms          | Brute cosine, 100–500 people         |
| Arbitration                  | < 1ms          | < 1ms          | Arithmetic                           |
| Temporal smoothing           | < 1ms          | < 1ms          | Dict + deque operations              |
| Renderer overlay             | 2ms            | 2ms            | cv2 drawing                          |
| **TOTAL (1 face)**           | **~27ms**      | **~119ms**     |                                      |
| **TOTAL (5 faces)**          | **~100ms**     | **~500ms**     |                                      |
| **Effective FPS (5 faces)**  | **~10 fps**    | **~2 fps**     | CPU needs 2-model fallback           |
| **CPU 2-model (5 faces)**    | —              | **~350ms**     | **~3 fps** — usable                  |

---

## Risk Mitigation

| Risk                                       | Mitigation                                                                 |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| Two enrolled people look nearly identical  | Pre-prom audit flags them; arbitration requires unanimous for flagged pairs |
| Heavy prom makeup shifts embedding space   | ElasticFace weight boosted; threshold softened by attribute layer          |
| Strobe/colored lighting on dance floor     | Quality gate rejects unusable frames; temporal smoothing rides through     |
| Glasses on/off between enrollment and event | Attribute layer detects change; threshold adjusted accordingly            |
| GPU runs out of memory                     | Graceful fallback to CPU 2-model mode mid-session                         |
| Person not enrolled                        | All models return below threshold → "Unknown" (correct behavior)          |
| Single model hallucinates a match          | Ensemble voting prevents any single model from overriding consensus       |
| Name flickers between frames               | Temporal smoothing (15-frame window, 60% consensus required)              |

---

## What This System Does NOT Do

- **Does not log every recognition** — no audit log, no database, no JSON files during live mode
- **Does not train a classifier** — uses pretrained embedding models with gallery search
- **Does not require multiple photos** — one photo per person, augmented automatically
- **Does not require internet** — all models run locally
- **Does not force a match** — Unknown is always a valid output
- **Does not make decisions silently** — enrollment requires explicit operator confirmation

---

## IMPLEMENTATION COMPLETE — 2026-04-28

All 5 sessions implemented and tested.

**Test coverage:** 65 tests passing across pipeline, stages, ASL model, face core (gallery/temporal/arbitrator), and face integration (FaceStage, enrollment helpers, audit). No GPU or camera required to run the test suite.

**Known notes:**
- AdaFace IR101 and ElasticFace-Arc+ require manual ONNX download (see `setup_models.py`). The system degrades gracefully to 2-model (A+B) or single-model (A) if files are missing.
- InsightFace antelopev2 (~300 MB) downloads automatically on first `--mode face` run.
- GPU OOM mid-session is handled via per-model try/except in `EnsembleRecognizer.extract_embeddings` — a failing model is silently skipped for that frame without crashing the pipeline.
