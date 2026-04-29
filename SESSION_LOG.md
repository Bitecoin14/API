# Hand Tracker — Development Session Log

> Full history of what was built in each session.

---

## Sessions 1–2 (Initial Build)
**Date:** ~2026-04-21
**Branch:** master

### What Was Built

- **Core webcam loop**: OpenCV `VideoCapture` + MediaPipe Hands
- **Hand skeleton renderer**: 21 landmark points + connections, colored per hand (green/blue)
- **8 visual filters**: Normal, Inverted, Hallucinogenic, ASCII, Upside Down, Mosaic, Black-White, Flat 2D
- **Pinch gesture** to cycle filters (thumb-index distance < 30px)
- **Middle-finger blur** (always-on overlay)
- **Basic logging**: `hand_tracker.log` rotating file handler

**Key files introduced:**
- `hand_tracker.py` (original single-file version)
- `filters/` directory (all 8 filters)

---

## Session 3 (Sign Language + Pose)
**Date:** ~2026-04-22

### What Was Built

- **ASL rule-based recognizer** (`sign_language_module/recognizer.py`):
  - 42-landmark extension (21 real + 21 interpolated midpoints)
  - Geometric decision tree for letters A–Z
  - Per-hand deque history (12 frames) with majority voting
- **Body pose detection** (`pose_module/detector.py`):
  - MediaPipe Pose wrapper with 33 landmarks
  - Color-coded skeleton by body region
- **GestureRecognizer** (`pose_module/gestures.py`):
  - 10 body gestures: waving, clapping, arms stretched, hands up, hand raised, arms crossed, hands on hips, bowing, victory pose, shrugging

---

## Session 4 (Architecture Refactor)
**Date:** 2026-04-23

### What Was Built

The original single-file `hand_tracker.py` was decomposed into a proper package architecture.

**New structure:**
- `core/pipeline.py`: 3-thread pipeline (capture → inference → render) with queue-based backpressure
- `core/context.py`: `FrameContext` dataclass for stage communication
- `core/config.py`: Frozen `Config` dataclass + `build_parser()` CLI
- `core/logging_setup.py`: Rotating file + console logging
- `stages/capture.py`, `stages/inference.py`, `stages/filter_stage.py`, `stages/renderer.py`
- `stages/asl_stage.py`: Wraps ASL recognizer as pipeline stage

**Why:** Original design ran everything in one loop — inference blocked rendering. The 3-thread design allows capture and inference to run asynchronously, preventing frame drops.

**Test coverage introduced:** 35 tests across pipeline, stages, ASL model, config flags.

**Reference:** `docs/REFACTOR_PROGRESS.md`

---

## Session 5 — Face Recognition: Detection Layer
**Date:** 2026-04-24 (estimated)

### What Was Built

- **`face/types.py`**: All shared dataclasses (`DetectedFace`, `RecognitionStatus`, `ModelVote`, `RecognitionResult`, `FaceAttributes`)
- **`face/detector.py`**: `DualDetector` — RetinaFace (InsightFace) + YOLOv8-face consensus detection
  - IoU ≥ 0.5 required between both detectors
  - Quality gate: size, score, yaw, blur
  - Returns aligned 112×112 face crop via `insightface.utils.face_align.norm_crop`
- **`face/attributes.py`**: Glasses detection (Haar cascade), makeup estimation (HSV saturation), quality score
- **`core/config.py`** extended with all face recognition constants
- **`requirements.txt`** extended with insightface, onnxruntime-gpu, ultralytics

---

## Session 6 — Face Recognition: Single-Model Recognition
**Date:** 2026-04-25 (estimated)

### What Was Built

- **`face/gallery.py`**: `FaceGallery` — per-person per-model `.npy` embedding storage, cosine similarity search, `metadata.json`
- **`face/enrollment.py`**: `enroll_from_capture` — interactive webcam enrollment with confirmation loop (y/n/r)
  - 5-variant augmentation per capture (original, flip, brightness ±15%, blur)
  - Mean embedding across variants, L2-normalized
- **`hand_tracker.py`**: `--mode enroll` wired up
- **`stages/face_stage.py`**: Initial version using single-model recognition
- **`stages/renderer.py`**: `_draw_face_box` added

---

## Session 7 — Face Recognition: 3-Model Ensemble
**Date:** 2026-04-26 (estimated)

### What Was Built

- **`face/models.py`**: `EnsembleRecognizer` — loads antelopev2 (Model A), AdaFace IR101 (Model B), ElasticFace-Arc+ (Model C)
  - ONNX Runtime inference for B and C
  - L2 normalization on all embeddings
  - CPU fallback: skips Model C when no CUDA available
- **`face/arbitration.py`**: `Arbitrator` — 5 decision rules with dynamic weight adjustment
  - Weights shift based on face quality, glasses change, makeup
- **`stages/renderer.py`**: Color-coded bounding boxes by status
- **`stages/face_stage.py`**: Updated to full ensemble pipeline

---

## Session 8 — Face Recognition: Temporal Smoothing + Audit
**Date:** 2026-04-27 (estimated)

### What Was Built

- **`face/temporal.py`**: `TemporalSmoother` — IoU-based face tracker
  - 15-frame rolling window per track
  - 60% consensus before displaying a name
  - 3-frame initialization delay
  - 30-frame timeout for lost faces
  - Thread-safe via `threading.Lock`
- **`hand_tracker.py`**: `--mode audit` wired up
- **`face/enrollment.py`**: `run_audit` — N×N similarity matrix, confusable-pair flagging
- **`face/gallery.py`**: `compute_similarity_matrix`, `update_metadata` added

---

## Session 9 — Face Recognition: Polish + CPU Fallback + Batch Enroll
**Date:** 2026-04-28

### What Was Built

- **CPU 2-model fallback**: `CPU_FALLBACK_TWO_MODELS` config flag, applied in `EnsembleRecognizer.__init__`
- **`--from-folder` batch enrollment**: `enroll_from_folder` — reads images named `FirstName_LastName.jpg`, confirms per image
- **`setup_models.py`**: Downloads and verifies all face recognition models, checks GPU availability
- **`README.md`**: Complete usage instructions
- **Edge-case handling**: empty gallery, no faces, GPU OOM graceful fallback
- **Test coverage**: 65 total tests (23 face core + 6 face integration added)

**Status at end of session:** ALL COMPLETE. 65 tests passing.

---

## Session 10 — Documentation + Review
**Date:** 2026-04-29

### What Was Done

- Full codebase analysis and review
- Fixed `README.md`: removed non-existent `--no-face`, `--no-pose`, `--no-blur`, `--no-hand-skeleton` flags (the real flags are opt-in: `--face`, `--pose`, `--blur`, `--asl`)
- Created `CODEBASE_OVERVIEW.md`: full architecture documentation
- Created `SESSION_LOG.md` (this file): development history
- Created `TASKS.md`: current state + pending work
- Created `FACE_RECOGNITION_TUTORIAL.md`: step-by-step photo enrollment guide
