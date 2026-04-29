# Refactor Progress Tracker

**Goal:** Professional-level restructure of the Hand Tracking CV app.  
**Status:** Brainstorming → Design phase complete, awaiting implementation plan.

---

## Decisions Locked In

| # | Decision | Detail |
|---|---|---|
| 1 | ASL model | Pre-trained scikit-learn `.pkl` (landmark-based, free, no API) |
| 2 | Threading | Multi-threaded pipeline — capture / inference / render threads, careful design |
| 3 | CLI | Flags may be updated; `run.bat` and `setup.bat` must continue to work |
| 4 | Architecture | Pipeline + Protocol pattern (see design doc) |

---

## Session Checkpoint: Brainstorming Complete

- [x] Full codebase explored (all 20 source files read)
- [x] Clarifying questions answered (3/3)
- [x] Approaches proposed and approved (Pipeline pattern)
- [x] Design sections presented and approved
- [x] Design doc written → `docs/superpowers/specs/2026-04-23-refactor-design.md`
- [x] Spec self-reviewed and inconsistencies fixed (4 issues resolved)
- [x] **Implementation plan written** → `docs/superpowers/plans/2026-04-23-professional-refactor.md`
- [x] **Implementation executed** (subagent-driven-development) — **ALL 18 TASKS COMPLETE**
- [x] **35/35 tests passing** as of 2026-04-24

---

## What the Refactor Covers

### 1. New Directory Layout
```
hand_tracker/              ← rename root package
  core/
    config.py              ← central Config dataclass (replaces scattered constants)
    pipeline.py            ← Pipeline class: ordered list of Stage objects
    context.py             ← FrameContext dataclass passed between stages
    logging_setup.py       ← one-time logging configuration
  stages/
    capture.py             ← CaptureStage  (reads from camera)
    inference.py           ← InferenceStage (hands + pose on background thread)
    filters.py             ← FilterStage   (applies active visual filter)
    face.py                ← FaceStage     (face recognition overlay)
    sign_language.py       ← ASLStage      (ML-based letter recognition)
    gestures.py            ← GestureStage  (body gestures from pose)
    renderer.py            ← RenderStage   (draws skeleton, HUD, panels)
  models/
    asl_classifier.pkl     ← pre-trained scikit-learn ASL model
    train_asl.py           ← offline training script
  filters/                 ← unchanged filter modules (normal, inverted, …)
  face_recognition_module/ ← unchanged
  pose_module/             ← detector.py + gestures.py (unchanged internals)
  sign_language_module/    ← recognizer.py updated to call ML model
  tests/
    test_pipeline.py
    test_stages.py
    test_asl_model.py
    (existing tests kept)
  hand_tracker.py          ← entry point (thin: parse args → build Pipeline → run)
  requirements.txt
  setup.bat
  run.bat
```

### 2. Threading Model (Three-Thread Pipeline)
```
Thread A — Capture:   VideoCapture.read() → raw_queue
Thread B — Inference: raw_queue → MediaPipe Hands + Pose → result_queue  
Thread C — Render:    result_queue → apply filter → draw → imshow
```
- Queues are `queue.Queue(maxsize=2)` — natural backpressure, drops stale frames
- Inference thread owns both Hands and Pose models (single thread, no GIL contention on C extensions)
- Render thread is the main thread (OpenCV `imshow` must be on main thread on most platforms)

### 3. ASL Model
- Features: 21 landmarks × (x, y, z) = 63 floats, normalized to wrist=origin, scale=wrist→middle-MCP
- Classifier: `sklearn.pipeline.Pipeline(StandardScaler + RandomForestClassifier)` or SVM
- Training data: ASL landmark dataset (to be collected/sourced, script in `models/train_asl.py`)
- Fallback: if `.pkl` missing, falls back to the existing rule-based recognizer with a warning log
- Model size target: < 2 MB

### 4. Central Config (`core/config.py`)
All magic numbers in one place as a frozen dataclass. CLI args override it at startup.

### 5. Logging
`logging` module throughout. Default: `WARNING` to console, full `DEBUG` to `hand_tracker.log`.

### 6. Filter Protocol
Filters become `Protocol`-typed dicts with optional `coord_transform`. No breaking changes to existing filter files.

### 7. Updated CLI
```
--camera N          (unchanged)
--filter NAME       (unchanged)
--list-filters      (unchanged)
--no-hand-skeleton  (was --no-skeleton)
--no-pose           (unchanged)
--no-face           (was --no-face-recognition)
--no-blur           (unchanged)
--asl               (was --sign-language / -sl)
--no-asl            new: disable ASL completely
--log-level LEVEL   new: DEBUG/INFO/WARNING/ERROR
--resolution WxH    new: downscale capture for performance
--model-complexity  new: 0/1/2 for MediaPipe model quality
```
`run.bat` calls `python hand_tracker.py` with no flags — unchanged behaviour.

---

## Known Current Problems (addressed by refactor)

| Problem | Fix |
|---|---|
| 420-line god-object `run()` | Decomposed into Stage classes |
| ASCII filter O(rows×cols) Python loop | Vectorized with numpy |
| Hallucinogenic re-allocates meshgrid every frame | Cached per frame-size |
| No logging | `logging` module, configurable level |
| No type hints on many functions | Full type annotations |
| Scattered magic constants | Central `Config` dataclass |
| Rule-based ASL breaks under rotation | ML model with landmark normalization |
| No tests for filters/pose/ASL | New test suite in `tests/` |
| Single-threaded → display stutters | 3-thread capture/inference/render |
| Filter `__init__.py` must be edited to add overlays | Overlay registry pattern |

---

## IMPLEMENTATION COMPLETE — 2026-04-24

All tasks finished. The refactor is fully implemented and tested.

**What was built:**
- `core/` — Config, FrameContext, Pipeline (3-thread), logging_setup
- `stages/` — CaptureStage, InferenceStage, FilterStage, FaceStage, ASLStage, RendererStage
- `models/train_asl.py` + `models/asl_classifier.pkl` (RandomForest, 84.4% CV accuracy)
- `filters/ascii_art.py` vectorised (20ms/frame, 10x speedup)
- `filters/hallucinogenic.py` meshgrid cached (19ms/frame)
- `tests/` — 35 tests passing (pipeline, stages, ASL model, face, hand tracker)
- `hand_tracker.py` — stripped to 42 lines

**One known note:** `models/asl_classifier.pkl` is ~103 MB (spec said <2 MB — the RandomForest with 200 estimators/depth-20 produces a large model. Can reduce by lowering n_estimators).
