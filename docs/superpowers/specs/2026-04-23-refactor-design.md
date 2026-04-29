# Hand Tracker — Professional Refactor Design Spec
**Date:** 2026-04-23  
**Status:** Approved — ready for implementation planning  
**Approach:** Pipeline + Protocol (Approach A)

---

## 1. Goals

1. Break the 420-line `run()` god-function into independently testable Stage classes.
2. Three-thread pipeline (capture / inference / render) for smooth, lag-free video.
3. ML-based ASL recognition replacing fragile geometric rules.
4. Central `Config` dataclass — no more scattered magic constants.
5. Structured logging (`logging` module) replacing bare `print()` calls.
6. Full type annotations throughout.
7. Vectorised hot-path filters (ASCII, Hallucinogenic).
8. Expanded test suite covering all modules.
9. All existing filter/face/pose module files kept unchanged.

---

## 2. Directory Layout

```
Hand Tracking/
├── core/
│   ├── __init__.py
│   ├── config.py           # Config frozen dataclass + CLI parser
│   ├── context.py          # FrameContext dataclass
│   ├── pipeline.py         # Pipeline class + Stage protocol
│   └── logging_setup.py    # configure_logging()
│
├── stages/
│   ├── __init__.py
│   ├── capture.py          # CaptureStage  — camera → raw_queue
│   ├── inference.py        # InferenceStage — Hands + Pose → result_queue
│   ├── filter_stage.py     # FilterStage   — applies active visual filter
│   ├── face_stage.py       # FaceStage     — LBPH face recognition overlay
│   ├── asl_stage.py        # ASLStage      — ML letter recognition
│   └── renderer.py         # RendererStage — skeleton + HUD + panels
│   # GestureStage removed — gesture recognition runs inside InferenceStage
│
├── models/
│   ├── asl_classifier.pkl  # bundled sklearn model (< 2 MB)
│   └── train_asl.py        # offline training script
│
├── filters/                # UNCHANGED
├── face_recognition_module/# UNCHANGED
├── sign_language_module/   # rule-based recognizer kept as fallback
├── pose_module/            # UNCHANGED
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_stages.py
│   ├── test_asl_model.py
│   ├── test_hand_tracker.py   # existing, moved here
│   └── test_face_recognition.py # existing, moved here
│
├── hand_tracker.py         # ~40 lines: parse Config → build Pipeline → run
├── requirements.txt        # adds scikit-learn
├── setup.bat               # UNCHANGED
└── run.bat                 # UNCHANGED
```

---

## 3. Core Components

### 3.1 `core/context.py` — FrameContext

```python
@dataclass
class FrameContext:
    raw_frame:      np.ndarray          # original BGR from camera
    frame:          np.ndarray          # working copy (modified by stages)
    timestamp:      float               # time.monotonic() at capture
    frame_id:       int                 # monotonically increasing
    config:         Config

    # Inference results (filled by InferenceStage)
    hand_results:   Any | None = None   # mp.solutions.hands result
    pose_results:   Any | None = None   # mp.solutions.pose result

    # Derived (filled by downstream stages)
    asl_letters:    list[tuple[int, str]] = field(default_factory=list)  # (hand_idx, letter)
    body_gestures:  list[str] = field(default_factory=list)
    active_filter:  dict       = field(default_factory=dict)
    coord_transform: Any | None = None
    capture_fps:    float = 0.0   # written by CaptureStage, read by RendererStage
```

All mutable state a stage needs is in `FrameContext`. Stages do not share any other state.

### 3.2 `core/config.py` — Config

```python
@dataclass(frozen=True)
class Config:
    camera:           int   = 0
    resolution:       tuple = (640, 480)   # inference resolution (display may differ)
    model_complexity: int   = 1
    filter_name:      str | None = None
    log_level:        str  = "WARNING"

    show_hand_skeleton: bool = True
    show_pose:          bool = True
    show_face:          bool = True
    show_blur:          bool = True
    show_asl:           bool = False

    hand_detection_conf:  float = 0.7
    hand_tracking_conf:   float = 0.7
    pose_detection_conf:  float = 0.6
    pose_tracking_conf:   float = 0.6

    asl_model_path: Path = Path("models/asl_classifier.pkl")
    known_faces_dir: Path = Path("known_faces")
```

`Config.from_args(namespace)` is a classmethod that builds a Config from parsed CLI args.

### 3.3 `core/pipeline.py` — Stage Protocol + Pipeline

```python
class Stage(Protocol):
    def process(self, ctx: FrameContext) -> FrameContext: ...
    def close(self) -> None: ...

class Pipeline:
    def __init__(self, stages: list[Stage]) -> None: ...
    def run(self) -> None:   # starts threads, runs render loop
    def shutdown(self) -> None:
```

`Pipeline.run()` starts the capture and inference threads, then runs the render loop on the calling (main) thread.

### 3.4 `core/logging_setup.py`

```python
def configure_logging(level: str) -> None:
    # Console: WARNING+  (clean user output)
    # File:    DEBUG+     (hand_tracker.log, rotating, max 5 MB × 3 files)
```

---

## 4. Threading Model

```
CAPTURE THREAD          raw_q (maxsize=2)       INFERENCE THREAD            result_q (maxsize=2)     RENDER THREAD (main)
──────────────          ─────────────────       ────────────────            ────────────────────     ────────────────────
VideoCapture.read()  →  FrameContext(raw)    →  RGB convert                  FrameContext(results) →  FilterStage.process()
drop frame if full      daemon thread           Hands.process()              daemon thread            FaceStage.process()
write capture_fps                               Pose.process()                                        ASLStage.process()
                                                GestureRecognizer.recognize()                         RendererStage.process()
                                                attach all results                                    cv2.imshow()
                                                                                                      cv2.waitKey(1)
```

**Shutdown flow:**
1. User presses `Q` → render thread sets `stop_event`.
2. Capture thread checks `stop_event` each iteration → exits loop → puts `None` sentinel on `raw_q`.
3. Inference thread sees `None` → puts `None` on `result_q` → exits.
4. Render thread sees `None` → exits loop.
5. `Pipeline.shutdown()` joins both daemon threads, releases camera, closes models.

**Frame-drop policy:**  
`raw_q.put_nowait(ctx)` — if `queue.Full`, the frame is silently discarded. This prevents the display from lagging behind real time.

---

## 5. Stage Designs

### 5.1 CaptureStage (capture thread)
- Runs `VideoCapture.read()` in a loop.
- Wraps each frame in a fresh `FrameContext`.
- Applies optional resolution downscale (`cv2.resize`) before queuing.
- Tracks FPS using a rolling window of capture timestamps; writes `ctx.capture_fps` each frame.

### 5.2 InferenceStage (inference thread)
- Converts BGR → RGB once per frame.
- Runs `hands.process(rgb)` then `pose.process(rgb)` sequentially.
- Owns and runs `GestureRecognizer.recognize(pose_results.pose_landmarks)` — writes `ctx.body_gestures`.
- All three objects (Hands, Pose, GestureRecognizer) are owned exclusively by this thread — no locking needed.
- Attaches `hand_results`, `pose_results`, and `body_gestures` to context before queuing.

### 5.3 FilterStage (render thread)
- Holds its own `_filter_index: int` instance state (Config is frozen; cycling state lives here).
- On first frame, initialises `_filter_index` from `config.filter_name` lookup.
- Detects open-hand + pinch from `ctx.hand_results` to advance `_filter_index`.
- Applies `FILTERS[_filter_index]` to `ctx.frame` in place.
- Writes `ctx.active_filter` and `ctx.coord_transform`.

### 5.4 FaceStage (render thread)
- Calls existing `recognize_and_draw(ctx.frame, recognizer, label_map)`.
- Skipped when `config.show_face` is False.
- LBPH recognizer is loaded once at startup.

### 5.5 ASLStage (render thread)
- **Primary path:** loads `models/asl_classifier.pkl` (sklearn pipeline: StandardScaler + RandomForest).
- Input features: 21 landmarks × (x, y, z) = 63 floats, normalised to wrist=origin, scale=wrist→middle-MCP distance.
- Per-hand letter history (deque) for temporal smoothing (same approach as existing).
- **Fallback:** if `.pkl` missing, imports `sign_language_module.recognize_letter` with a `logging.warning`.
- Skipped when `config.show_asl` is False.

### 5.6 GestureStage — removed
- GestureRecognizer is owned by InferenceStage (see 5.2). `ctx.body_gestures` is already populated when the render thread receives the context.
- No separate GestureStage needed; `stages/gesture_stage.py` is not created.

### 5.7 RendererStage (render thread)
- Middle-finger blur (calls existing `ALWAYS_ON_OVERLAY`).
- Pose skeleton (calls `pose_module.PoseDetector.draw()`).
- Hand skeleton (existing `_draw_hand()` logic, extracted to a helper).
- ASL letter overlay (existing `_draw_sign_letter()` logic).
- Gesture panel (existing `_draw_gesture_panel()` logic).
- HUD line (filter name, FPS, open-hand-detected indicator).
- All drawing is on `ctx.frame` in place.

---

## 6. ASL Model

### Training data
- Source: ASL landmark dataset derived from publicly available sign-language image sets.
- Extraction script (`models/train_asl.py`) processes images through MediaPipe Hands and saves landmark CSVs.
- Alternatively, a community-collected landmark CSV dataset (e.g., from Kaggle ASL Alphabet) can be used directly.

### Model
```
sklearn.pipeline.Pipeline([
    ("scaler",  StandardScaler()),
    ("clf",     RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
])
```
- Input: 63 float features (normalised landmarks)
- Output: single character A–Z (or None class for unclear pose)
- Cross-validated accuracy target: ≥ 90% on held-out test set
- Serialised with `joblib.dump` → `models/asl_classifier.pkl`

### Fallback chain
```
asl_classifier.pkl exists?
  YES → sklearn predict
  NO  → sign_language_module.recognize_letter (rule-based) + log warning
```

---

## 7. Performance Improvements

| Issue | Fix |
|---|---|
| ASCII filter: Python loop O(rows×cols) | Vectorise: pre-build char map with `np.take`, render all text via vectorised font LUT |
| Hallucinogenic: meshgrid allocated every frame | Cache `(map_x_base, map_y_base)` keyed by `(h, w)`; only recompute on resolution change |
| Both MediaPipe models block display | Moved to inference thread — display never waits for inference |
| Frame lag under slow inference | Bounded queue + frame-drop policy |
| No FPS cap | Render loop adds `cv2.waitKey` adaptive delay to target 30 FPS cap |
| Haar cascade runs on full-res frame | Downscale to 320×240 for detection, upscale bounding boxes for display |

---

## 8. Updated CLI

```
hand_tracker.py [options]

  --camera N            Camera index (default: 0)
  --filter NAME         Start with named filter active
  --list-filters        Print filter names and exit
  --resolution WxH      Inference resolution, e.g. 640x480 (default: 640x480)
  --model-complexity N  MediaPipe model quality 0/1/2 (default: 1)
  --log-level LEVEL     DEBUG / INFO / WARNING / ERROR (default: WARNING)

  --asl                 Enable ASL fingerspelling recognition
  --no-hand-skeleton    Disable hand skeleton overlay  (was --no-skeleton)
  --no-pose             Disable body pose + gestures    (unchanged)
  --no-face             Disable face recognition        (was --no-face-recognition)
  --no-blur             Disable middle-finger blur       (unchanged)
```

`run.bat` calls `python hand_tracker.py` with no flags — all defaults preserved, behaviour unchanged.

---

## 9. Logging Strategy

| Logger | Level | Output |
|---|---|---|
| `hand_tracker.capture` | DEBUG | Frame timestamps, drop counts |
| `hand_tracker.inference` | DEBUG | Per-frame inference time |
| `hand_tracker.asl` | WARNING | Model load failures, fallback activation |
| `hand_tracker.face` | WARNING | Training image load failures |
| `hand_tracker` (root) | INFO+ | Startup banner, shutdown |

Console shows WARNING+. `hand_tracker.log` (rotating) captures DEBUG+.

---

## 10. Testing Strategy

| Test file | What it covers |
|---|---|
| `tests/test_pipeline.py` | Pipeline init, thread startup/shutdown, sentinel propagation |
| `tests/test_stages.py` | Each Stage.process() with a synthetic FrameContext |
| `tests/test_asl_model.py` | Model loads, predict() returns a char or None, fallback activates |
| `tests/test_hand_tracker.py` | Existing MediaPipe tests (moved) |
| `tests/test_face_recognition.py` | Existing face tests (moved) |

All tests run without a camera (`cv2.VideoCapture` is mocked in pipeline tests).

---

## 11. What Is NOT Changed

- `filters/` — all 8 filter files + `__init__.py` (only hot-path vectorisation applied)
- `face_recognition_module/` — loader and recognizer untouched
- `pose_module/` — detector and gestures untouched
- `sign_language_module/` — kept as fallback
- `setup.bat` / `run.bat` — no changes
- `requirements.txt` — only adds `scikit-learn` and `joblib`
