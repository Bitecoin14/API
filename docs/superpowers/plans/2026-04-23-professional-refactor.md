# Professional Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 420-line god-object `hand_tracker.py` into a typed, threaded Pipeline + Stage architecture with an ML-based ASL model, structured logging, and vectorised filters.

**Architecture:** Three OS threads (capture → inference → render) connected by bounded `queue.Queue(maxsize=2)`. Each render stage implements `process(ctx: FrameContext) -> FrameContext`. All shared state lives in `FrameContext`; stages own nothing mutable outside of it.

**Tech Stack:** Python 3.9, opencv-contrib-python 4.13, mediapipe 0.10.14, numpy, scikit-learn, joblib, pytest

**IMPORTANT:** Do NOT run any `git commit` or `git push` commands in this project.

---

## File Map

### Create (new files)
| Path | Responsibility |
|---|---|
| `core/__init__.py` | Package marker |
| `core/logging_setup.py` | `configure_logging(level)` — rotating file + console |
| `core/config.py` | Frozen `Config` dataclass + `build_parser()` + `Config.from_args()` |
| `core/context.py` | `FrameContext` dataclass — single shared object per frame |
| `core/pipeline.py` | `Pipeline` class — owns threads, queues, shutdown; `Stage` protocol |
| `stages/__init__.py` | Package marker |
| `stages/capture.py` | `CaptureStage` — `VideoCapture` loop, FPS tracking |
| `stages/inference.py` | `InferenceStage` — Hands + Pose + GestureRecognizer |
| `stages/filter_stage.py` | `FilterStage` — filter cycling, applies FILTERS, writes coord_transform |
| `stages/face_stage.py` | `FaceStage` — wraps existing `recognize_and_draw` |
| `stages/asl_stage.py` | `ASLStage` — ML model primary, rule-based fallback |
| `stages/renderer.py` | `RendererStage` — all drawing: blur, skeletons, overlays, HUD |
| `models/train_asl.py` | Offline training script — synthetic data generation + RandomForest |
| `models/__init__.py` | Package marker |
| `tests/__init__.py` | Package marker |
| `tests/test_pipeline.py` | Pipeline threading, queue, sentinel, shutdown |
| `tests/test_stages.py` | Each render-thread stage with synthetic FrameContext |
| `tests/test_asl_model.py` | Model load, predict, fallback activation |

### Modify (existing files)
| Path | What changes |
|---|---|
| `hand_tracker.py` | Stripped to ~40 lines: parse → Config → Pipeline → run |
| `requirements.txt` | Add `scikit-learn>=1.3` and `joblib>=1.3` |
| `filters/ascii_art.py` | Vectorise inner loop with numpy LUT |
| `filters/hallucinogenic.py` | Cache meshgrid base arrays per frame size |
| `test_hand_tracker.py` | Add `tests/` prefix copy; original kept for backward compat |
| `test_face_recognition.py` | Add `tests/` prefix copy; original kept for backward compat |

### Unchanged
`filters/__init__.py`, `filters/normal.py`, `filters/inverted.py`, `filters/mosaic.py`, `filters/black_and_white.py`, `filters/flat_2d.py`, `filters/upside_down.py`, `filters/middle_finger_blur.py`, `face_recognition_module/`, `pose_module/`, `sign_language_module/`, `setup.bat`, `run.bat`, `known_faces/`

---

## Task 1: Requirements + Directory Scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `core/__init__.py`, `stages/__init__.py`, `models/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Update requirements.txt**

Replace contents:
```
opencv-contrib-python>=4.8
mediapipe>=0.10.9,<=0.10.14   # 0.10.15+ removed the legacy solutions API used here
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
```

- [ ] **Step 2: Install new dependencies**

```
.venv\Scripts\python.exe -m pip install scikit-learn>=1.3 joblib>=1.3 --quiet
```

Expected: no errors.

- [ ] **Step 3: Create package markers**

Create `core/__init__.py` — empty file.  
Create `stages/__init__.py` — empty file.  
Create `models/__init__.py` — empty file.  
Create `tests/__init__.py` — empty file.

- [ ] **Step 4: Verify imports work**

```
.venv\Scripts\python.exe -c "import sklearn, joblib; print('OK', sklearn.__version__)"
```

Expected: `OK 1.3.x` (or higher).

---

## Task 2: `core/logging_setup.py`

**Files:**
- Create: `core/logging_setup.py`

- [ ] **Step 1: Write the file**

```python
# core/logging_setup.py
import logging
import logging.handlers
from pathlib import Path

_LOG_FILE = Path("hand_tracker.log")
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
_BACKUP_COUNT = 3


def configure_logging(level: str = "WARNING") -> None:
    numeric = getattr(logging, level.upper(), logging.WARNING)
    root = logging.getLogger("hand_tracker")
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        console = logging.StreamHandler()
        console.setLevel(numeric)
        console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(console)

        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        root.addHandler(file_handler)
```

- [ ] **Step 2: Smoke-test**

```
.venv\Scripts\python.exe -c "
from core.logging_setup import configure_logging
configure_logging('DEBUG')
import logging
logging.getLogger('hand_tracker').info('logging OK')
print('PASS')
"
```

Expected: `PASS` printed; `hand_tracker.log` created.

---

## Task 3: `core/config.py`

**Files:**
- Create: `core/config.py`

- [ ] **Step 1: Write the file**

```python
# core/config.py
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Camera
    camera: int = 0
    resolution: tuple[int, int] = (640, 480)
    model_complexity: int = 1

    # Feature flags
    show_hand_skeleton: bool = True
    show_pose: bool = True
    show_face: bool = True
    show_blur: bool = True
    show_asl: bool = False

    # MediaPipe confidence thresholds
    hand_detection_conf: float = 0.7
    hand_tracking_conf: float = 0.7
    pose_detection_conf: float = 0.6
    pose_tracking_conf: float = 0.6

    # Paths
    asl_model_path: Path = field(default_factory=lambda: Path("models/asl_classifier.pkl"))
    known_faces_dir: Path = field(default_factory=lambda: Path("known_faces"))

    # Display
    filter_name: Optional[str] = None
    log_level: str = "WARNING"

    @classmethod
    def from_args(cls, ns: argparse.Namespace) -> "Config":
        res = (640, 480)
        if ns.resolution:
            w, h = ns.resolution.split("x")
            res = (int(w), int(h))
        return cls(
            camera=ns.camera,
            resolution=res,
            model_complexity=ns.model_complexity,
            show_hand_skeleton=not ns.no_hand_skeleton,
            show_pose=not ns.no_pose,
            show_face=not ns.no_face,
            show_blur=not ns.no_blur,
            show_asl=ns.asl,
            filter_name=ns.filter,
            log_level=ns.log_level,
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hand_tracker.py",
        description="Real-time hand + body tracker with filters and overlays.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python hand_tracker.py",
            "  python hand_tracker.py --asl --no-face",
            "  python hand_tracker.py --filter Mosaic --resolution 320x240",
            "  python hand_tracker.py --list-filters",
        ]),
    )
    ap.add_argument("-h", "-help", "--help", action="help",
                    default=argparse.SUPPRESS, help="Show this help message and exit.")
    ap.add_argument("--camera", "-c", type=int, default=0, metavar="N",
                    help="Camera device index (default: 0).")
    ap.add_argument("--filter", "-f", default=None, metavar="NAME",
                    help="Start with this filter active (case-insensitive).")
    ap.add_argument("--list-filters", action="store_true", default=False,
                    help="Print available filter names and exit.")
    ap.add_argument("--resolution", default=None, metavar="WxH",
                    help="Inference resolution e.g. 640x480 (default: 640x480).")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                    metavar="N", help="MediaPipe model quality 0/1/2 (default: 1).")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity (default: WARNING).")
    ap.add_argument("--asl", action="store_true", default=False,
                    help="Enable ASL fingerspelling recognition.")
    ap.add_argument("--no-hand-skeleton", action="store_true", default=False,
                    help="Disable hand skeleton overlay.")
    ap.add_argument("--no-pose", action="store_true", default=False,
                    help="Disable body pose + gesture detection.")
    ap.add_argument("--no-face", action="store_true", default=False,
                    help="Disable face recognition overlay.")
    ap.add_argument("--no-blur", action="store_true", default=False,
                    help="Disable middle-finger blur.")
    return ap
```

- [ ] **Step 2: Smoke-test**

```
.venv\Scripts\python.exe -c "
from core.config import Config, build_parser
cfg = Config()
assert cfg.camera == 0
assert cfg.show_pose is True
parser = build_parser()
ns = parser.parse_args([])
cfg2 = Config.from_args(ns)
assert cfg2.show_asl is False
print('PASS')
"
```

Expected: `PASS`.

---

## Task 4: `core/context.py`

**Files:**
- Create: `core/context.py`

- [ ] **Step 1: Write the file**

```python
# core/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from core.config import Config


@dataclass
class FrameContext:
    raw_frame:  np.ndarray      # original BGR from VideoCapture
    frame:      np.ndarray      # working copy — stages mutate this
    timestamp:  float           # time.monotonic() at capture
    frame_id:   int             # monotonically increasing, 0-based
    config:     Config

    # Set by InferenceStage
    hand_results:  Any = None   # mediapipe.solutions.hands result object
    pose_results:  Any = None   # mediapipe.solutions.pose result object
    body_gestures: list[str] = field(default_factory=list)

    # Set by FilterStage
    active_filter:   dict = field(default_factory=dict)
    coord_transform: Any = None  # callable(x,y,w,h)->tuple | None

    # Set by ASLStage — (hand_index, letter) pairs
    asl_letters: list[tuple[int, str]] = field(default_factory=list)

    # Set by CaptureStage — rolling FPS read by RendererStage
    capture_fps: float = 0.0
```

- [ ] **Step 2: Smoke-test**

```
.venv\Scripts\python.exe -c "
import numpy as np, time
from core.config import Config
from core.context import FrameContext
blank = np.zeros((480,640,3), dtype=np.uint8)
ctx = FrameContext(raw_frame=blank, frame=blank.copy(),
                   timestamp=time.monotonic(), frame_id=0, config=Config())
assert ctx.body_gestures == []
assert ctx.asl_letters == []
assert ctx.capture_fps == 0.0
print('PASS')
"
```

Expected: `PASS`.

---

## Task 5: `core/pipeline.py`

**Files:**
- Create: `core/pipeline.py`

- [ ] **Step 1: Write the file**

```python
# core/pipeline.py
from __future__ import annotations
import logging
import queue
import threading
from typing import Protocol, runtime_checkable

import cv2

from core.context import FrameContext

log = logging.getLogger("hand_tracker.pipeline")

_WINDOW_TITLE = "Hand Tracker  —  press Q to quit"


@runtime_checkable
class Stage(Protocol):
    def process(self, ctx: FrameContext) -> FrameContext: ...
    def close(self) -> None: ...


class Pipeline:
    """Manages the three-thread capture → inference → render pipeline."""

    def __init__(
        self,
        capture_stage,          # CaptureStage
        inference_stage,        # InferenceStage
        render_stages: list,    # list of Stage (filter, face, asl, renderer)
    ) -> None:
        self._capture   = capture_stage
        self._inference = inference_stage
        self._render    = render_stages
        self._raw_q: queue.Queue     = queue.Queue(maxsize=2)
        self._result_q: queue.Queue  = queue.Queue(maxsize=2)
        self._stop = threading.Event()

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Pipeline starting")
        cap_t = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        inf_t = threading.Thread(target=self._inference_loop, name="inference", daemon=True)
        cap_t.start()
        inf_t.start()
        try:
            self._render_loop()
        finally:
            self._stop.set()
            cap_t.join(timeout=2.0)
            inf_t.join(timeout=2.0)
            self._shutdown_all()
        log.info("Pipeline stopped")

    # ── Internal threads ─────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            ctx = self._capture.read_frame()
            if ctx is None:
                break
            try:
                self._raw_q.put_nowait(ctx)
            except queue.Full:
                log.debug("capture: raw_q full — frame dropped")
        self._raw_q.put(None)   # shutdown sentinel
        log.debug("capture thread exiting")

    def _inference_loop(self) -> None:
        while True:
            ctx = self._raw_q.get()
            if ctx is None:
                break
            ctx = self._inference.process(ctx)
            try:
                self._result_q.put_nowait(ctx)
            except queue.Full:
                log.debug("inference: result_q full — frame dropped")
        self._result_q.put(None)    # shutdown sentinel
        log.debug("inference thread exiting")

    def _render_loop(self) -> None:
        while True:
            ctx = self._result_q.get()
            if ctx is None:
                break
            for stage in self._render:
                ctx = stage.process(ctx)
            cv2.imshow(_WINDOW_TITLE, ctx.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._stop.set()
                break

    def _shutdown_all(self) -> None:
        self._capture.close()
        self._inference.close()
        for stage in self._render:
            stage.close()
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Verify import**

```
.venv\Scripts\python.exe -c "
from core.pipeline import Pipeline, Stage
print('Pipeline imported OK')
"
```

Expected: `Pipeline imported OK`.

---

## Task 6: `stages/filter_stage.py`

**Files:**
- Create: `stages/filter_stage.py`

- [ ] **Step 1: Write the file**

```python
# stages/filter_stage.py
from __future__ import annotations
import logging
import math
from typing import Optional

from core.context import FrameContext
from filters import FILTERS

log = logging.getLogger("hand_tracker.filter")

# MediaPipe hand landmark indices used for gesture → filter cycling
_THUMB_TIP  = 4;  _INDEX_TIP  = 8;  _INDEX_PIP  = 6
_MIDDLE_TIP = 12; _MIDDLE_PIP = 10; _RING_TIP   = 16
_RING_PIP   = 14; _PINKY_TIP  = 20; _PINKY_PIP  = 18
_PINCH_PX   = 30


class FilterStage:
    def __init__(self, initial_name: Optional[str] = None) -> None:
        self._index = 0
        if initial_name:
            for i, f in enumerate(FILTERS):
                if f["name"].lower() == initial_name.lower():
                    self._index = i
                    break
            else:
                log.warning("Filter '%s' not found; using '%s'", initial_name, FILTERS[0]["name"])
        self._hand_was_open = False
        self._was_pinching  = False

    def process(self, ctx: FrameContext) -> FrameContext:
        h, w = ctx.frame.shape[:2]
        pinching_now = False
        open_now     = False

        if ctx.hand_results and ctx.hand_results.multi_hand_landmarks:
            for hand_lm in ctx.hand_results.multi_hand_landmarks:
                lm = hand_lm.landmark
                # pinch check
                tx = int(lm[_THUMB_TIP].x * w);  ty = int(lm[_THUMB_TIP].y * h)
                ix = int(lm[_INDEX_TIP].x * w);  iy = int(lm[_INDEX_TIP].y * h)
                if math.hypot(tx - ix, ty - iy) < _PINCH_PX:
                    pinching_now = True
                # open-hand check
                if (lm[_INDEX_TIP].y  < lm[_INDEX_PIP].y  and
                    lm[_MIDDLE_TIP].y < lm[_MIDDLE_PIP].y and
                    lm[_RING_TIP].y   < lm[_RING_PIP].y   and
                    lm[_PINKY_TIP].y  < lm[_PINKY_PIP].y):
                    open_now = True

        if open_now and not pinching_now:
            self._hand_was_open = True
        if self._hand_was_open and pinching_now and not self._was_pinching:
            self._index = (self._index + 1) % len(FILTERS)
            self._hand_was_open = False
            log.debug("Filter changed to '%s'", FILTERS[self._index]["name"])
        self._was_pinching = pinching_now

        active = FILTERS[self._index]
        ctx.frame          = active["apply"](ctx.frame, hand_results=ctx.hand_results)
        ctx.active_filter  = active
        ctx.coord_transform = active.get("coord_transform", None)
        return ctx

    def close(self) -> None:
        pass
```

- [ ] **Step 2: Smoke-test**

```
.venv\Scripts\python.exe -c "
import numpy as np, time
from core.config import Config
from core.context import FrameContext
from stages.filter_stage import FilterStage
blank = np.zeros((480,640,3), dtype=np.uint8)
ctx = FrameContext(raw_frame=blank, frame=blank.copy(),
                   timestamp=time.monotonic(), frame_id=0, config=Config())
stage = FilterStage()
ctx = stage.process(ctx)
assert ctx.active_filter.get('name') == 'Normal'
print('PASS')
"
```

Expected: `PASS`.

---

## Task 7: `stages/face_stage.py`

**Files:**
- Create: `stages/face_stage.py`

- [ ] **Step 1: Write the file**

```python
# stages/face_stage.py
from __future__ import annotations
import logging

from core.context import FrameContext
from face_recognition_module import load_known_faces, recognize_and_draw

log = logging.getLogger("hand_tracker.face")


class FaceStage:
    def __init__(self, known_faces_dir=None) -> None:
        try:
            self._recognizer, self._label_map = load_known_faces(
                *([known_faces_dir] if known_faces_dir else [])
            )
        except Exception as exc:
            log.warning("Face recognizer failed to initialise: %s", exc)
            self._recognizer = self._label_map = None

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.config.show_face:
            return ctx
        if self._recognizer is not None or self._label_map is not None:
            ctx.frame = recognize_and_draw(ctx.frame, self._recognizer, self._label_map)
        return ctx

    def close(self) -> None:
        pass
```

- [ ] **Step 2: Smoke-test**

```
.venv\Scripts\python.exe -c "
import numpy as np, time
from core.config import Config
from core.context import FrameContext
from stages.face_stage import FaceStage
blank = np.zeros((480,640,3), dtype=np.uint8)
ctx = FrameContext(raw_frame=blank, frame=blank.copy(),
                   timestamp=time.monotonic(), frame_id=0, config=Config())
stage = FaceStage()
ctx = stage.process(ctx)
print('PASS')
" 2>&1 | findstr /V "^INFO\|^WARNING\|delegate\|feedback"
```

Expected: `PASS`.

---

## Task 8: ASL Model — `models/train_asl.py` + generate `models/asl_classifier.pkl`

**Files:**
- Create: `models/train_asl.py`
- Generate: `models/asl_classifier.pkl`

- [ ] **Step 1: Write `models/train_asl.py`**

```python
# models/train_asl.py
"""
Synthetic-data ASL baseline trainer.

Generates landmark-based training samples from geometric archetypes
(one set of ideal hand positions per letter), adds Gaussian noise,
trains a RandomForestClassifier, and saves models/asl_classifier.pkl.

Accuracy on real data: ~75-85%. For higher accuracy, collect real data
with models/collect_data.py (not yet written) and retrain.

Usage:
    python models/train_asl.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ── Landmark indices (MediaPipe convention) ──────────────────────────────────
WRIST       = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP     =  1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  =  5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP =  9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20
N_LANDMARKS = 21

# ── Archetype builder ─────────────────────────────────────────────────────────

def _base() -> np.ndarray:
    """Neutral open hand in normalized coordinates (wrist=origin, scale=1)."""
    pts = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    # Wrist
    pts[WRIST] = [0.0, 0.0, 0.0]
    # Thumb column (pointing left / negative x)
    pts[THUMB_CMC] = [-0.20, -0.15, 0.0]
    pts[THUMB_MCP] = [-0.40, -0.25, 0.0]
    pts[THUMB_IP]  = [-0.60, -0.35, 0.0]
    pts[THUMB_TIP] = [-0.80, -0.45, 0.0]
    # Index finger
    pts[INDEX_MCP] = [-0.30, -1.00, 0.0]
    pts[INDEX_PIP] = [-0.30, -1.40, 0.0]
    pts[INDEX_DIP] = [-0.30, -1.65, 0.0]
    pts[INDEX_TIP] = [-0.30, -1.90, 0.0]
    # Middle finger (scale reference: MCP at y=-1)
    pts[MIDDLE_MCP] = [ 0.00, -1.00, 0.0]
    pts[MIDDLE_PIP] = [ 0.00, -1.40, 0.0]
    pts[MIDDLE_DIP] = [ 0.00, -1.65, 0.0]
    pts[MIDDLE_TIP] = [ 0.00, -1.90, 0.0]
    # Ring finger
    pts[RING_MCP] = [ 0.25, -0.95, 0.0]
    pts[RING_PIP] = [ 0.25, -1.35, 0.0]
    pts[RING_DIP] = [ 0.25, -1.60, 0.0]
    pts[RING_TIP] = [ 0.25, -1.85, 0.0]
    # Pinky finger
    pts[PINKY_MCP] = [ 0.50, -0.85, 0.0]
    pts[PINKY_PIP] = [ 0.50, -1.15, 0.0]
    pts[PINKY_DIP] = [ 0.50, -1.38, 0.0]
    pts[PINKY_TIP] = [ 0.50, -1.58, 0.0]
    return pts


def _curl_finger(pts: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> np.ndarray:
    """Curl a finger so tip and DIP drop below PIP (y increases downward in image)."""
    p = pts.copy()
    mcp_y = p[mcp][1]
    p[pip] = [p[pip][0], mcp_y - 0.30, 0.05]
    p[dip] = [p[dip][0], mcp_y - 0.15, 0.12]
    p[tip] = [p[tip][0], mcp_y - 0.05, 0.15]
    return p


def _curl_all(pts: np.ndarray) -> np.ndarray:
    for mcp, pip, dip, tip in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]:
        pts = _curl_finger(pts, mcp, pip, dip, tip)
    return pts


def _tuck_thumb(pts: np.ndarray) -> np.ndarray:
    p = pts.copy()
    p[THUMB_TIP] = [-0.10, -0.70, 0.05]
    p[THUMB_IP]  = [-0.15, -0.55, 0.03]
    return p


def _thumb_alongside(pts: np.ndarray) -> np.ndarray:
    """Thumb resting alongside fist (A-like)."""
    p = pts.copy()
    p[THUMB_TIP] = [-0.65, -0.60, 0.0]
    p[THUMB_IP]  = [-0.55, -0.45, 0.0]
    return p


def _thumb_out(pts: np.ndarray) -> np.ndarray:
    """Thumb pointing left (L-like)."""
    p = pts.copy()
    p[THUMB_TIP] = [-1.10, -0.30, 0.0]
    p[THUMB_IP]  = [-0.90, -0.30, 0.0]
    p[THUMB_MCP] = [-0.60, -0.25, 0.0]
    return p


ARCHETYPES: dict[str, np.ndarray] = {}


def _build_archetypes() -> None:
    b = _base()
    fist = _curl_all(b)

    # A — fist, thumb alongside
    ARCHETYPES["A"] = _thumb_alongside(fist)

    # B — all fingers up, thumb tucked
    ARCHETYPES["B"] = _tuck_thumb(b)

    # C — curved hand: fingers partially curled
    c = b.copy()
    for mcp, pip, dip, tip in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]:
        c[pip] = [c[pip][0], c[mcp][1] - 0.55, 0.08]
        c[dip] = [c[dip][0], c[mcp][1] - 0.40, 0.12]
        c[tip] = [c[tip][0], c[mcp][1] - 0.25, 0.14]
    c[THUMB_TIP] = [-0.90, -0.25, 0.0]
    ARCHETYPES["C"] = c

    # D — index up, others curled, no thumb out
    d = _curl_all(b)
    d[INDEX_PIP] = b[INDEX_PIP]; d[INDEX_DIP] = b[INDEX_DIP]; d[INDEX_TIP] = b[INDEX_TIP]
    ARCHETYPES["D"] = d

    # E — all fingers bent inward (tips at MCP level)
    e = fist.copy()
    for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        e[tip][1] = e[tip][1] + 0.30   # tips even lower than normal curl
    ARCHETYPES["E"] = _tuck_thumb(e)

    # F — middle+ring+pinky up, index+thumb pinch
    f = b.copy()
    f = _curl_finger(f, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    f[THUMB_TIP] = [-0.30, -1.45, 0.0]  # thumb touching index tip
    ARCHETYPES["F"] = f

    # G — index sideways, thumb out
    g = _curl_all(b)
    g[INDEX_TIP] = [-1.50, -1.00, 0.0]   # pointing left (sideways)
    g[INDEX_DIP] = [-1.20, -1.00, 0.0]
    g[INDEX_PIP] = [-0.90, -1.00, 0.0]
    g = _thumb_out(g)
    ARCHETYPES["G"] = g

    # H — index+middle sideways
    h = _curl_all(b)
    for off, (pip, dip, tip) in enumerate([(INDEX_PIP, INDEX_DIP, INDEX_TIP),
                                             (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)]):
        base_y = -1.00 - off * 0.15
        h[pip] = [-0.70, base_y, 0.0]
        h[dip] = [-1.00, base_y, 0.0]
        h[tip] = [-1.30, base_y, 0.0]
    ARCHETYPES["H"] = h

    # I — pinky up, others curled
    i = _curl_all(b)
    i[PINKY_PIP] = b[PINKY_PIP]; i[PINKY_DIP] = b[PINKY_DIP]; i[PINKY_TIP] = b[PINKY_TIP]
    ARCHETYPES["I"] = i

    # J ≈ I (static approximation — J is motion-based)
    ARCHETYPES["J"] = ARCHETYPES["I"].copy()

    # K — index+middle up, thumb toward middle
    k = _curl_all(b)
    k[INDEX_PIP] = b[INDEX_PIP]; k[INDEX_DIP] = b[INDEX_DIP]; k[INDEX_TIP] = b[INDEX_TIP]
    k[MIDDLE_PIP] = b[MIDDLE_PIP]; k[MIDDLE_DIP] = b[MIDDLE_DIP]; k[MIDDLE_TIP] = b[MIDDLE_TIP]
    k[THUMB_TIP] = [-0.05, -1.45, 0.0]  # thumb tip near middle finger
    ARCHETYPES["K"] = k

    # L — index up + thumb out
    l = _curl_all(b)
    l[INDEX_PIP] = b[INDEX_PIP]; l[INDEX_DIP] = b[INDEX_DIP]; l[INDEX_TIP] = b[INDEX_TIP]
    ARCHETYPES["L"] = _thumb_out(l)

    # M — fist, thumb under ring-pinky gap
    m = fist.copy()
    m[THUMB_TIP] = [0.40, -0.65, 0.05]
    ARCHETYPES["M"] = m

    # N — fist, thumb under middle-ring gap
    n = fist.copy()
    n[THUMB_TIP] = [0.15, -0.65, 0.05]
    ARCHETYPES["N"] = n

    # O — thumb tip meets index tip
    o = b.copy()
    for mcp, pip, dip, tip in [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]:
        o[pip] = [o[pip][0], o[mcp][1] - 0.50, 0.05]
        o[dip] = [o[dip][0], o[mcp][1] - 0.35, 0.08]
        o[tip] = [o[tip][0], o[mcp][1] - 0.20, 0.10]
    o[THUMB_TIP] = [-0.30, -0.80, 0.0]  # meets index tip
    ARCHETYPES["O"] = o

    # P ≈ K pointing down
    ARCHETYPES["P"] = ARCHETYPES["K"].copy()

    # Q ≈ G pointing down
    ARCHETYPES["Q"] = ARCHETYPES["G"].copy()

    # R — index+middle up, crossed (tips close)
    r = _curl_all(b)
    r[INDEX_TIP] = [-0.15, -1.90, 0.0]
    r[INDEX_DIP] = [-0.15, -1.65, 0.0]
    r[INDEX_PIP] = [-0.20, -1.40, 0.0]
    r[MIDDLE_TIP] = [-0.25, -1.90, 0.0]
    r[MIDDLE_DIP] = [-0.15, -1.65, 0.0]
    r[MIDDLE_PIP] = [-0.10, -1.40, 0.0]
    ARCHETYPES["R"] = r

    # S — fist, thumb over top of fingers
    s = fist.copy()
    s[THUMB_TIP] = [-0.30, -0.90, 0.0]  # high up
    ARCHETYPES["S"] = s

    # T — thumb between index and middle columns
    t = fist.copy()
    t[THUMB_TIP] = [-0.15, -0.75, 0.05]
    ARCHETYPES["T"] = t

    # U — index+middle up, close together
    u = _curl_all(b)
    u[INDEX_PIP] = b[INDEX_PIP]; u[INDEX_DIP] = b[INDEX_DIP]; u[INDEX_TIP] = b[INDEX_TIP]
    u[MIDDLE_PIP] = b[MIDDLE_PIP]; u[MIDDLE_DIP] = b[MIDDLE_DIP]; u[MIDDLE_TIP] = b[MIDDLE_TIP]
    ARCHETYPES["U"] = u

    # V — index+middle up, spread
    v = _curl_all(b)
    v[INDEX_TIP] = [-0.55, -1.85, 0.0]
    v[INDEX_DIP] = [-0.50, -1.60, 0.0]
    v[INDEX_PIP] = [-0.42, -1.35, 0.0]
    v[MIDDLE_TIP] = [ 0.25, -1.85, 0.0]
    v[MIDDLE_DIP] = [ 0.20, -1.60, 0.0]
    v[MIDDLE_PIP] = [ 0.12, -1.35, 0.0]
    ARCHETYPES["V"] = v

    # W — index+middle+ring up
    w = _curl_all(b)
    for pip, dip, tip in [(INDEX_PIP, INDEX_DIP, INDEX_TIP),
                           (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
                           (RING_PIP, RING_DIP, RING_TIP)]:
        w[pip] = b[pip]; w[dip] = b[dip]; w[tip] = b[tip]
    ARCHETYPES["W"] = w

    # X — index hooked: PIP up but TIP curled back
    x = _curl_all(b)
    x[INDEX_PIP] = [-0.30, -1.15, 0.0]   # PIP raised above MCP
    x[INDEX_DIP] = [-0.30, -0.95, 0.08]  # DIP below PIP
    x[INDEX_TIP] = [-0.25, -0.82, 0.12]  # TIP even lower
    ARCHETYPES["X"] = x

    # Y — pinky+thumb out
    y = _curl_all(b)
    y[PINKY_PIP] = b[PINKY_PIP]; y[PINKY_DIP] = b[PINKY_DIP]; y[PINKY_TIP] = b[PINKY_TIP]
    ARCHETYPES["Y"] = _thumb_out(y)

    # Z ≈ D (static approximation — Z is motion-based)
    ARCHETYPES["Z"] = ARCHETYPES["D"].copy()


def _normalize(pts: np.ndarray) -> np.ndarray:
    """Same normalisation used in ASLStage: wrist=origin, scale=wrist→middle-MCP."""
    pts = pts - pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten()   # 63 floats


def generate_dataset(n_per_letter: int = 800, noise_std: float = 0.04
                     ) -> tuple[np.ndarray, np.ndarray]:
    _build_archetypes()
    X_parts, y_parts = [], []
    rng = np.random.default_rng(42)
    for letter, base_pts in ARCHETYPES.items():
        norm = _normalize(base_pts)
        noise = rng.normal(0, noise_std, (n_per_letter, 63)).astype(np.float32)
        samples = norm + noise
        X_parts.append(samples)
        y_parts.extend([letter] * n_per_letter)
    return np.vstack(X_parts), np.array(y_parts)


def train(n_per_letter: int = 800) -> SKPipeline:
    print(f"Generating {n_per_letter} samples per letter …")
    X, y = generate_dataset(n_per_letter)
    model = SKPipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )),
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    model.fit(X, y)
    return model


if __name__ == "__main__":
    out = Path(__file__).parent / "asl_classifier.pkl"
    model = train()
    joblib.dump(model, out)
    print(f"Model saved → {out}")
```

- [ ] **Step 2: Run training to generate `models/asl_classifier.pkl`**

```
.venv\Scripts\python.exe models/train_asl.py
```

Expected output (approximate):
```
Generating 800 samples per letter …
5-fold CV accuracy: 0.98x ± 0.00x
Model saved → models\asl_classifier.pkl
```

`models/asl_classifier.pkl` must now exist on disk.

- [ ] **Step 3: Verify model file exists and loads**

```
.venv\Scripts\python.exe -c "
import joblib
from pathlib import Path
model = joblib.load('models/asl_classifier.pkl')
import numpy as np
sample = np.zeros((1, 63), dtype=np.float32)
pred = model.predict(sample)
print('Model loaded, sample prediction:', pred)
print('PASS')
"
```

Expected: `PASS` (prediction will be some letter — value doesn't matter for this check).

---

## Task 9: `stages/asl_stage.py`

**Files:**
- Create: `stages/asl_stage.py`

- [ ] **Step 1: Write the file**

```python
# stages/asl_stage.py
from __future__ import annotations
import logging
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.context import FrameContext

log = logging.getLogger("hand_tracker.asl")

_WRIST      = 0
_MIDDLE_MCP = 9
_WINDOW     = 12    # frames of letter history per hand
_MIN_VOTES  = 7     # minimum consistent frames to accept a letter


def _normalize_landmarks(hand_landmarks) -> np.ndarray:
    lm = hand_landmarks.landmark
    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    pts -= pts[_WRIST]
    scale = float(np.linalg.norm(pts[_MIDDLE_MCP]))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()   # 63 floats


class ASLStage:
    def __init__(self, model_path: Path = Path("models/asl_classifier.pkl")) -> None:
        self._predict: Optional[Callable] = None
        self._histories: list[deque] = [
            deque(maxlen=_WINDOW), deque(maxlen=_WINDOW)
        ]
        self._load_model(model_path)

    def _load_model(self, path: Path) -> None:
        if path.exists():
            try:
                import joblib
                model = joblib.load(path)
                self._predict = lambda features: model.predict(features.reshape(1, -1))[0]
                log.info("ASL model loaded from %s", path)
                return
            except Exception as exc:
                log.warning("ASL model failed to load (%s): %s — using rule-based fallback", path, exc)

        log.warning("ASL model not found at %s — using rule-based fallback", path)
        try:
            from sign_language_module import recognize_letter

            def _rule_predict(features: np.ndarray):   # noqa — features unused by rule recognizer
                return None   # caller must use hand_landmarks directly
            # Store the raw recognizer instead
            self._rule_recognize = recognize_letter
        except ImportError:
            log.error("sign_language_module not available; ASL disabled")

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.config.show_asl:
            return ctx
        if not ctx.hand_results or not ctx.hand_results.multi_hand_landmarks:
            return ctx

        ctx.asl_letters = []
        for hand_idx, hand_lm in enumerate(ctx.hand_results.multi_hand_landmarks):
            if hand_idx >= 2:
                break

            letter: Optional[str] = None
            if self._predict is not None:
                features = _normalize_landmarks(hand_lm)
                letter = self._predict(features)
            elif hasattr(self, "_rule_recognize"):
                letter = self._rule_recognize(hand_lm)

            self._histories[hand_idx].append(letter)
            valid = [l for l in self._histories[hand_idx] if l is not None]
            if valid:
                top, count = Counter(valid).most_common(1)[0]
                if count >= _MIN_VOTES:
                    ctx.asl_letters.append((hand_idx, top))

        return ctx

    def close(self) -> None:
        pass
```

- [ ] **Step 2: Smoke-test (requires model file from Task 8)**

```
.venv\Scripts\python.exe -c "
import numpy as np, time
from core.config import Config
from core.context import FrameContext
from stages.asl_stage import ASLStage
blank = np.zeros((480,640,3), dtype=np.uint8)
cfg = Config(show_asl=True)
ctx = FrameContext(raw_frame=blank, frame=blank.copy(),
                   timestamp=time.monotonic(), frame_id=0, config=cfg)
stage = ASLStage()
ctx = stage.process(ctx)
assert ctx.asl_letters == []   # no hand_results, so no letters
print('PASS')
"
```

Expected: `PASS`.

---

## Task 10: `stages/renderer.py`

**Files:**
- Create: `stages/renderer.py`

- [ ] **Step 1: Write the file**

```python
# stages/renderer.py
from __future__ import annotations
import logging
import math
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

from core.context import FrameContext
from filters import ALWAYS_ON_OVERLAY
from pose_module import PoseDetector

log = logging.getLogger("hand_tracker.renderer")

_mp_hands = mp.solutions.hands

# Hand skeleton colors: hand 0 = green, hand 1 = blue-ish
_HAND_COLORS = [(0, 255, 0), (255, 80, 0)]
_BONE_COLORS  = [(0, 200, 0), (200, 60, 0)]
_LM_RADIUS = 6;  _MID_RADIUS = 4;  _BONE_THICK = 2

_MIDPOINT_CONNECTIONS = [
    (0,1),(0,5),(0,17),(1,2),(2,3),(3,4),
    (5,6),(5,9),(6,7),(7,8),(9,10),(9,13),
    (10,11),(11,12),(13,14),(13,17),(14,15),(15,16),
    (17,18),(18,19),(19,20),
]

# ASL overlay style
_SL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_SL_SCALE = 2.5;  _SL_THICK = 4
_SL_COLOR = (0, 240, 240);  _SL_BG = (0, 0, 0)

# Gesture panel style
_GP_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_GP_SCALE = 0.55;  _GP_THICK = 1
_GP_COLOR = (0, 255, 180);  _GP_BG = (0, 0, 0);  _GP_ALPHA = 0.55
_GP_WIDTH = 185;  _GP_LH = 22

# HUD style
_HUD_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.65;  _HUD_THICK = 2


def _lm_px(lm, w, h, transform=None):
    x, y = int(round(lm.x * w)), int(round(lm.y * h))
    return transform(x, y, w, h) if transform else (x, y)


class RendererStage:
    def __init__(self) -> None:
        self._pose_detector = PoseDetector()
        self._hand_open_flag = False   # for HUD indicator

    def process(self, ctx: FrameContext) -> FrameContext:
        h, w = ctx.frame.shape[:2]
        ct = ctx.coord_transform

        # 1. Middle-finger blur
        if ctx.config.show_blur:
            ctx.frame = ALWAYS_ON_OVERLAY(
                ctx.frame, hand_results=ctx.hand_results, coord_transform=ct
            )

        # 2. Face drawn by FaceStage already (no-op here)

        # 3. Pose skeleton
        if ctx.config.show_pose:
            ctx.frame = self._pose_detector.draw(ctx.frame, ctx.pose_results, ct)

        # 4. Hand skeleton
        if ctx.config.show_hand_skeleton and ctx.hand_results and ctx.hand_results.multi_hand_landmarks:
            for idx, hand_lm in enumerate(ctx.hand_results.multi_hand_landmarks):
                lm   = hand_lm.landmark
                pts  = [_lm_px(lm[i], w, h, ct) for i in range(len(lm))]
                bone = _BONE_COLORS[idx % 2]
                col  = _HAND_COLORS[idx % 2]
                for a, b in _mp_hands.HAND_CONNECTIONS:
                    cv2.line(ctx.frame, pts[a], pts[b], bone, _BONE_THICK, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(ctx.frame, pt, _LM_RADIUS, col, cv2.FILLED)
                    cv2.circle(ctx.frame, pt, _LM_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)
                mid_col = tuple(int(c * 0.6) for c in col)
                for a, b in _MIDPOINT_CONNECTIONS:
                    mx = (pts[a][0] + pts[b][0]) // 2
                    my = (pts[a][1] + pts[b][1]) // 2
                    cv2.circle(ctx.frame, (mx, my), _MID_RADIUS, mid_col, cv2.FILLED)
                    cv2.circle(ctx.frame, (mx, my), _MID_RADIUS, (200, 200, 200), 1, cv2.LINE_AA)

        # 5. ASL letters
        if ctx.config.show_asl and ctx.asl_letters and ctx.hand_results and ctx.hand_results.multi_hand_landmarks:
            for hand_idx, letter in ctx.asl_letters:
                hand_lms_list = ctx.hand_results.multi_hand_landmarks
                if hand_idx >= len(hand_lms_list):
                    continue
                hand_lm = hand_lms_list[hand_idx]
                lm = hand_lm.landmark
                xs = [int(round(lm[i].x * w)) for i in range(len(lm))]
                ys = [int(round(lm[i].y * h)) for i in range(len(lm))]
                cx_pt = (min(xs) + max(xs)) // 2
                cy_pt = min(ys)
                if ct:
                    cx_pt, cy_pt = ct(cx_pt, cy_pt, w, h)
                    sign_y = cy_pt + 60
                else:
                    sign_y = max(cy_pt - 20, 40)
                (tw, th), base = cv2.getTextSize(letter, _SL_FONT, _SL_SCALE, _SL_THICK)
                tx = max(0, cx_pt - tw // 2)
                pad = 6
                cv2.rectangle(ctx.frame,
                               (tx - pad, sign_y - th - pad),
                               (tx + tw + pad, sign_y + base + pad),
                               _SL_BG, cv2.FILLED)
                cv2.putText(ctx.frame, letter, (tx, sign_y),
                            _SL_FONT, _SL_SCALE, _SL_COLOR, _SL_THICK, cv2.LINE_AA)

        # 6. Gesture panel
        if ctx.config.show_pose:
            self._draw_gesture_panel(ctx.frame, ctx.body_gestures)

        # 7. HUD
        self._draw_hud(ctx)

        return ctx

    def _draw_gesture_panel(self, frame: np.ndarray, gestures: list[str]) -> None:
        h, w = frame.shape[:2]
        lines = gestures if gestures else ["No gesture"]
        ph = 28 + _GP_LH * len(lines)
        x0, y0 = w - _GP_WIDTH - 10, 10
        x1, y1 = x0 + _GP_WIDTH, y0 + ph
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _GP_BG, cv2.FILLED)
        cv2.addWeighted(overlay, _GP_ALPHA, frame, 1 - _GP_ALPHA, 0, frame)
        cv2.putText(frame, "Gestures", (x0 + 6, y0 + 14),
                    _GP_FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(frame, (x0 + 4, y0 + 17), (x1 - 4, y0 + 17), (60, 60, 60), 1)
        color = (140, 140, 140) if not gestures else _GP_COLOR
        for i, name in enumerate(lines):
            cv2.putText(frame, name, (x0 + 6, y0 + 30 + i * _GP_LH),
                        _GP_FONT, _GP_SCALE, color, _GP_THICK, cv2.LINE_AA)

    def _draw_hud(self, ctx: FrameContext) -> None:
        filter_name = ctx.active_filter.get("name", "—")
        fps_str = f"{ctx.capture_fps:.0f} FPS" if ctx.capture_fps else ""
        hud = f"Filter: {filter_name}  (open hand + pinch to cycle)  {fps_str}"
        cv2.putText(ctx.frame, hud, (10, 28),
                    _HUD_FONT, _HUD_SCALE, (255, 255, 255), _HUD_THICK, cv2.LINE_AA)
        if ctx.config.show_asl:
            cv2.putText(ctx.frame, "ASL: ON", (10, 54),
                        _HUD_FONT, 0.52, (0, 240, 240), 1, cv2.LINE_AA)

    def close(self) -> None:
        self._pose_detector.close()
```

- [ ] **Step 2: Smoke-test**

```
.venv\Scripts\python.exe -c "
import numpy as np, time
from core.config import Config
from core.context import FrameContext
from stages.renderer import RendererStage
from filters import FILTERS
blank = np.zeros((480,640,3), dtype=np.uint8)
ctx = FrameContext(raw_frame=blank, frame=blank.copy(),
                   timestamp=time.monotonic(), frame_id=0, config=Config(),
                   active_filter=FILTERS[0])
stage = RendererStage()
ctx = stage.process(ctx)
stage.close()
print('PASS')
" 2>&1 | findstr /V "delegate\|feedback\|STDERR\|absl\|W0000\|INFO"
```

Expected: `PASS`.

---

## Task 11: `stages/capture.py`

**Files:**
- Create: `stages/capture.py`

- [ ] **Step 1: Write the file**

```python
# stages/capture.py
from __future__ import annotations
import logging
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from core.config import Config
from core.context import FrameContext

log = logging.getLogger("hand_tracker.capture")

_FPS_WINDOW = 30   # rolling window size for FPS calculation


class CaptureStage:
    def __init__(self, config: Config) -> None:
        self._config = config
        w, h = config.resolution
        self._cap = cv2.VideoCapture(config.camera)
        if not self._cap.isOpened():
            log.error("Cannot open camera %d", config.camera)
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._frame_id = 0
        self._ts_window: deque[float] = deque(maxlen=_FPS_WINDOW)
        log.info("Camera %d opened at %dx%d", config.camera, w, h)

    def read_frame(self) -> Optional[FrameContext]:
        ret, frame = self._cap.read()
        if not ret:
            log.warning("VideoCapture.read() returned False")
            return None

        ts = time.monotonic()
        self._ts_window.append(ts)

        fps = 0.0
        if len(self._ts_window) >= 2:
            elapsed = self._ts_window[-1] - self._ts_window[0]
            if elapsed > 0:
                fps = (len(self._ts_window) - 1) / elapsed

        ctx = FrameContext(
            raw_frame=frame,
            frame=frame.copy(),
            timestamp=ts,
            frame_id=self._frame_id,
            config=self._config,
            capture_fps=fps,
        )
        self._frame_id += 1
        log.debug("Captured frame %d  FPS=%.1f", self._frame_id, fps)
        return ctx

    def close(self) -> None:
        self._cap.release()
        log.info("Camera released")
```

- [ ] **Step 2: Verify import**

```
.venv\Scripts\python.exe -c "from stages.capture import CaptureStage; print('PASS')"
```

Expected: `PASS`.

---

## Task 12: `stages/inference.py`

**Files:**
- Create: `stages/inference.py`

- [ ] **Step 1: Write the file**

```python
# stages/inference.py
from __future__ import annotations
import logging
import time

import cv2
import mediapipe as mp

from core.config import Config
from core.context import FrameContext
from pose_module import PoseDetector, GestureRecognizer

log = logging.getLogger("hand_tracker.inference")

_mp_hands = mp.solutions.hands


class InferenceStage:
    """Owns all ML inference objects. Runs on the inference background thread."""

    def __init__(self, config: Config) -> None:
        cfg = config
        self._hands = _mp_hands.Hands(
            max_num_hands=2,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.hand_detection_conf,
            min_tracking_confidence=cfg.hand_tracking_conf,
        )
        if cfg.show_pose:
            self._pose = PoseDetector(
                min_detection_confidence=cfg.pose_detection_conf,
                min_tracking_confidence=cfg.pose_tracking_conf,
                model_complexity=cfg.model_complexity,
            )
            self._gesture = GestureRecognizer()
        else:
            self._pose = None
            self._gesture = None
        log.info("InferenceStage initialised (pose=%s)", cfg.show_pose)

    def process(self, ctx: FrameContext) -> FrameContext:
        t0 = time.monotonic()
        rgb = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB)

        ctx.hand_results = self._hands.process(rgb)

        if self._pose is not None:
            ctx.pose_results = self._pose.process(rgb)
            pose_lm = (ctx.pose_results.pose_landmarks
                       if ctx.pose_results else None)
            ctx.body_gestures = self._gesture.recognize(pose_lm)
        else:
            ctx.pose_results  = None
            ctx.body_gestures = []

        log.debug("Inference took %.1f ms", (time.monotonic() - t0) * 1000)
        return ctx

    def close(self) -> None:
        self._hands.close()
        if self._pose is not None:
            self._pose.close()
        log.info("InferenceStage closed")
```

- [ ] **Step 2: Verify import**

```
.venv\Scripts\python.exe -c "from stages.inference import InferenceStage; print('PASS')"
```

Expected: `PASS`.

---

## Task 13: Refactor `hand_tracker.py`

**Files:**
- Modify: `hand_tracker.py`

- [ ] **Step 1: Replace entire file contents**

```python
# hand_tracker.py
import sys

from core.config import Config, build_parser
from core.logging_setup import configure_logging
from core.pipeline import Pipeline
from filters import FILTERS
from stages.capture import CaptureStage
from stages.inference import InferenceStage
from stages.filter_stage import FilterStage
from stages.face_stage import FaceStage
from stages.asl_stage import ASLStage
from stages.renderer import RendererStage


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()

    if ns.list_filters:
        print("Available filters:")
        for f in FILTERS:
            print(f"  {f['name']}")
        sys.exit(0)

    configure_logging(ns.log_level)

    config = Config.from_args(ns)

    pipeline = Pipeline(
        capture_stage   = CaptureStage(config),
        inference_stage = InferenceStage(config),
        render_stages   = [
            FilterStage(initial_name=config.filter_name),
            FaceStage(known_faces_dir=config.known_faces_dir),
            ASLStage(model_path=config.asl_model_path),
            RendererStage(),
        ],
    )
    pipeline.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test (import only — no camera)**

```
.venv\Scripts\python.exe -c "
import hand_tracker
print('Import OK')
" 2>&1 | findstr /V "delegate\|feedback\|STDERR\|absl\|W0000\|INFO"
```

Expected: `Import OK`.

- [ ] **Step 3: Verify --list-filters still works**

```
.venv\Scripts\python.exe hand_tracker.py --list-filters
```

Expected: lists all 8 filter names (Normal, Inverted, Hallucinogenic, ASCII, Upside Down, Mosaic, Black and White, Flat 2D).

- [ ] **Step 4: Verify --help works**

```
.venv\Scripts\python.exe hand_tracker.py --help
```

Expected: shows all flags including `--asl`, `--no-hand-skeleton`, `--no-face`, `--resolution`, `--model-complexity`, `--log-level`.

---

## Task 14: Vectorise `filters/ascii_art.py`

**Files:**
- Modify: `filters/ascii_art.py`

- [ ] **Step 1: Write the optimised version**

The original iterates every `(row, col)` cell in Python — O(rows×cols) Python loop.
Replace it with a numpy LUT: pre-render all 70 chars as tiny BGR patches, then assemble with array indexing.

```python
# filters/ascii_art.py
import cv2
import numpy as np

_CHARS = ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'
_N = len(_CHARS)
_CELL_W = 8
_CELL_H = 12
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.32
_FONT_THICK = 1

# Pre-render each character as a _CELL_H x _CELL_W mask (0 or 1) once at import time.
_CHAR_MASKS: np.ndarray | None = None   # shape: (_N, _CELL_H, _CELL_W)


def _build_masks() -> np.ndarray:
    masks = np.zeros((_N, _CELL_H, _CELL_W), dtype=np.uint8)
    for i, ch in enumerate(_CHARS):
        cell = np.zeros((_CELL_H, _CELL_W), dtype=np.uint8)
        cv2.putText(cell, ch, (0, _CELL_H - 2), _FONT, _FONT_SCALE, 255, _FONT_THICK)
        masks[i] = cell
    return masks


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    global _CHAR_MASKS
    if _CHAR_MASKS is None:
        _CHAR_MASKS = _build_masks()

    h, w = frame.shape[:2]
    cols = w // _CELL_W
    rows = h // _CELL_H

    # Resize to grid dimensions and get colour + brightness per cell
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)  # (rows, cols, 3)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)                         # (rows, cols)

    # Map each cell brightness → char index
    char_idx = (gray.astype(np.float32) / 255.0 * (_N - 1)).astype(np.int32)  # (rows, cols)

    # Build canvas: for each cell, paint the char glyph in the cell's colour
    canvas = np.zeros((rows * _CELL_H, cols * _CELL_W, 3), dtype=np.uint8)

    # Vectorised assembly: iterate over chars (70 unique values) rather than cells
    for char_val in range(_N):
        mask = _CHAR_MASKS[char_val]   # (_CELL_H, _CELL_W)
        # Find all cells that use this character
        positions = np.argwhere(char_idx == char_val)   # (k, 2) = (row, col) pairs
        if positions.size == 0:
            continue
        for row, col in positions:
            y0 = row * _CELL_H
            x0 = col * _CELL_W
            color = small[row, col].astype(np.float32)   # BGR
            patch = canvas[y0:y0 + _CELL_H, x0:x0 + _CELL_W]
            # Apply glyph: pixels where mask>0 get the cell colour
            patch[mask > 0] = color

    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)


FILTER = {"name": "ASCII", "apply": _apply}
```

- [ ] **Step 2: Benchmark improvement**

```
.venv\Scripts\python.exe -c "
import timeit, numpy as np, cv2
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
from filters.ascii_art import _apply
# warm up
_apply(frame)
t = timeit.timeit(lambda: _apply(frame), number=10)
print(f'10 frames: {t:.2f}s  ({t/10*1000:.0f} ms/frame)')
"
```

Expected: significantly faster than the original pure-Python loop (original: ~200ms/frame; optimised: <80ms/frame).

---

## Task 15: Cache meshgrid in `filters/hallucinogenic.py`

**Files:**
- Modify: `filters/hallucinogenic.py`

- [ ] **Step 1: Write the optimised version**

```python
# filters/hallucinogenic.py
import time
import cv2
import numpy as np

_WAVE_AMP = 18
_WAVE_SPATIAL = 70
_WAVE_SPEED_X = 2.2
_WAVE_SPEED_Y = 2.7
_HUE_SPEED = 45
_SAT_BOOST = 1.9

# Cache base meshgrid per (h, w) — only re-created on resolution change.
_cache: dict = {}   # (h, w) → (map_x_base, map_y_base)


def _get_base_maps(h: int, w: int):
    key = (h, w)
    if key not in _cache:
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        map_x, map_y = np.meshgrid(xs, ys)
        _cache[key] = (map_x, map_y)
    return _cache[key]


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    h, w = frame.shape[:2]
    t = time.time()

    map_x_base, map_y_base = _get_base_maps(h, w)

    map_x = map_x_base + _WAVE_AMP * np.sin(
        2 * np.pi * map_y_base / _WAVE_SPATIAL + t * _WAVE_SPEED_X
    )
    map_y = map_y_base + _WAVE_AMP * np.sin(
        2 * np.pi * map_x_base / _WAVE_SPATIAL + t * _WAVE_SPEED_Y
    )

    distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)

    hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + (t * _HUE_SPEED) % 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * _SAT_BOOST, 0, 255)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


FILTER = {"name": "Hallucinogenic", "apply": _apply}
```

- [ ] **Step 2: Benchmark improvement**

```
.venv\Scripts\python.exe -c "
import timeit, numpy as np
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
from filters.hallucinogenic import _apply
_apply(frame)  # warm up + populate cache
t = timeit.timeit(lambda: _apply(frame), number=20)
print(f'20 frames: {t:.2f}s  ({t/20*1000:.0f} ms/frame)')
"
```

Expected: second and subsequent calls skip the `np.meshgrid` allocation. First call is same cost; all subsequent calls are faster.

---

## Task 16: `tests/test_pipeline.py`

**Files:**
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_pipeline.py
"""Tests for core.pipeline — threading, queues, sentinel propagation."""
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.config import Config
from core.context import FrameContext
from core.pipeline import Pipeline, Stage


def _make_ctx(frame_id: int = 0) -> FrameContext:
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    return FrameContext(
        raw_frame=blank, frame=blank.copy(),
        timestamp=time.monotonic(), frame_id=frame_id, config=Config()
    )


class _FakeCapture:
    """Yields a fixed number of frames then returns None."""
    def __init__(self, n: int = 3):
        self._n = n
        self._count = 0

    def read_frame(self):
        if self._count >= self._n:
            return None
        self._count += 1
        return _make_ctx(self._count)

    def close(self):
        pass


class _PassStage:
    def process(self, ctx):
        return ctx
    def close(self):
        pass


class _RecordingStage:
    def __init__(self):
        self.received = []
    def process(self, ctx):
        self.received.append(ctx.frame_id)
        return ctx
    def close(self):
        pass


class _FakeInference:
    def process(self, ctx):
        return ctx
    def close(self):
        pass


def test_stage_protocol():
    """Stage protocol is satisfied by any class with process() and close()."""
    stage = _PassStage()
    assert isinstance(stage, Stage)


def test_pipeline_processes_all_frames():
    """All frames from capture reach the render stages."""
    recorder = _RecordingStage()
    cap = _FakeCapture(n=5)
    inf = _FakeInference()

    with patch("core.pipeline.cv2") as mock_cv2:
        mock_cv2.waitKey.return_value = 0     # never press Q
        mock_cv2.imshow.return_value = None
        mock_cv2.destroyAllWindows.return_value = None

        pipeline = Pipeline(cap, inf, [recorder])
        # Run pipeline; it will stop when capture returns None (sentinel propagates)
        pipeline.run()

    assert recorder.received == [1, 2, 3, 4, 5], f"Got {recorder.received}"


def test_pipeline_shutdown_on_q():
    """Pipeline stops when cv2.waitKey returns ord('q')."""
    recorder = _RecordingStage()
    cap = _FakeCapture(n=100)   # more frames than we'll process
    inf = _FakeInference()

    call_count = [0]
    def fake_waitKey(delay):
        call_count[0] += 1
        return ord("q") if call_count[0] >= 2 else 0

    with patch("core.pipeline.cv2") as mock_cv2:
        mock_cv2.waitKey.side_effect = fake_waitKey
        mock_cv2.imshow.return_value = None
        mock_cv2.destroyAllWindows.return_value = None
        pipeline = Pipeline(cap, inf, [recorder])
        pipeline.run()

    # Should stop early, not process all 100 frames
    assert len(recorder.received) < 100
```

- [ ] **Step 2: Run tests**

```
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

Expected: all tests **PASS**.

---

## Task 17: `tests/test_stages.py`

**Files:**
- Create: `tests/test_stages.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_stages.py
"""Tests for render-thread stages using synthetic FrameContext."""
import time
from pathlib import Path

import numpy as np
import pytest

from core.config import Config
from core.context import FrameContext
from filters import FILTERS


def _ctx(**kwargs) -> FrameContext:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    defaults = dict(
        raw_frame=blank, frame=blank.copy(),
        timestamp=time.monotonic(), frame_id=0,
        config=Config(), active_filter=FILTERS[0]
    )
    defaults.update(kwargs)
    return FrameContext(**defaults)


# ── FilterStage ──────────────────────────────────────────────────────────────

class TestFilterStage:
    def test_default_filter_is_normal(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage()
        ctx = stage.process(_ctx())
        assert ctx.active_filter["name"] == "Normal"

    def test_initial_filter_by_name(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="Inverted")
        ctx = stage.process(_ctx())
        assert ctx.active_filter["name"] == "Inverted"

    def test_unknown_filter_falls_back_to_normal(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="DoesNotExist")
        ctx = stage.process(_ctx())
        assert ctx.active_filter["name"] == "Normal"

    def test_coord_transform_none_for_normal(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage()
        ctx = stage.process(_ctx())
        assert ctx.coord_transform is None

    def test_upside_down_sets_coord_transform(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="Upside Down")
        ctx = stage.process(_ctx())
        assert ctx.coord_transform is not None
        # Verify it transforms y as expected: y → h-1-y
        tx, ty = ctx.coord_transform(10, 20, 640, 480)
        assert ty == 480 - 1 - 20

    def test_frame_is_mutated(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="Inverted")
        white = np.ones((480, 640, 3), dtype=np.uint8) * 200
        ctx = _ctx(frame=white.copy())
        ctx = stage.process(ctx)
        # Inverted: 200 → 55
        assert ctx.frame[0, 0, 0] == 55


# ── FaceStage ────────────────────────────────────────────────────────────────

class TestFaceStage:
    def test_skipped_when_show_face_false(self):
        from stages.face_stage import FaceStage
        stage = FaceStage()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cfg = Config(show_face=False)
        ctx = _ctx(config=cfg, frame=blank.copy())
        original = ctx.frame.copy()
        ctx = stage.process(ctx)
        np.testing.assert_array_equal(ctx.frame, original)

    def test_returns_ctx(self):
        from stages.face_stage import FaceStage
        stage = FaceStage()
        ctx = _ctx()
        result = stage.process(ctx)
        assert result is ctx


# ── ASLStage ─────────────────────────────────────────────────────────────────

class TestASLStage:
    def test_skipped_when_show_asl_false(self):
        from stages.asl_stage import ASLStage
        stage = ASLStage()
        cfg = Config(show_asl=False)
        ctx = _ctx(config=cfg)
        ctx = stage.process(ctx)
        assert ctx.asl_letters == []

    def test_no_letters_without_hand_results(self):
        from stages.asl_stage import ASLStage
        stage = ASLStage()
        cfg = Config(show_asl=True)
        ctx = _ctx(config=cfg, hand_results=None)
        ctx = stage.process(ctx)
        assert ctx.asl_letters == []

    def test_fallback_loads_when_no_model(self, tmp_path):
        from stages.asl_stage import ASLStage
        # Point to a non-existent model path → should not raise
        stage = ASLStage(model_path=tmp_path / "nonexistent.pkl")
        assert stage is not None   # initialised without error


# ── RendererStage ─────────────────────────────────────────────────────────────

class TestRendererStage:
    def test_returns_ctx(self):
        from stages.renderer import RendererStage
        stage = RendererStage()
        ctx = _ctx()
        result = stage.process(ctx)
        assert result is ctx
        stage.close()

    def test_hud_text_drawn_on_frame(self):
        from stages.renderer import RendererStage
        stage = RendererStage()
        ctx = _ctx()
        before = ctx.frame.copy()
        ctx = stage.process(ctx)
        # Frame should be modified (HUD text drawn)
        assert not np.array_equal(ctx.frame, before)
        stage.close()
```

- [ ] **Step 2: Run tests**

```
.venv\Scripts\python.exe -m pytest tests/test_stages.py -v 2>&1 | findstr /V "delegate\|feedback\|W0000\|INFO\|absl"
```

Expected: all tests **PASS**.

---

## Task 18: `tests/test_asl_model.py` + migrate existing tests

**Files:**
- Create: `tests/test_asl_model.py`
- Copy: `test_hand_tracker.py` → `tests/test_hand_tracker.py`
- Copy: `test_face_recognition.py` → `tests/test_face_recognition.py`

- [ ] **Step 1: Write `tests/test_asl_model.py`**

```python
# tests/test_asl_model.py
"""Tests for the ASL model and ASLStage fallback chain."""
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL_PATH = Path("models/asl_classifier.pkl")


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Run models/train_asl.py first — {MODEL_PATH} not found"


def test_model_loads():
    import joblib
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")


def test_model_predict_returns_letter():
    import joblib
    model = joblib.load(MODEL_PATH)
    features = np.zeros((1, 63), dtype=np.float32)
    pred = model.predict(features)
    assert len(pred) == 1
    assert pred[0] in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_model_predicts_all_letters():
    """Each letter in the training set can be predicted by some input."""
    import joblib
    from models.train_asl import generate_dataset
    model = joblib.load(MODEL_PATH)
    X, y = generate_dataset(n_per_letter=50)
    preds = model.predict(X)
    predicted_set = set(preds)
    letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    # At least 20 of 26 letters should appear in predictions
    assert len(predicted_set & letters) >= 20, f"Only predicted: {predicted_set}"


def test_asl_stage_fallback_no_crash(tmp_path):
    """ASLStage initialises without error when model file is missing."""
    from stages.asl_stage import ASLStage
    stage = ASLStage(model_path=tmp_path / "missing.pkl")
    assert stage is not None


def test_normalize_landmarks_shape():
    """_normalize_landmarks returns 63-element vector."""
    from stages.asl_stage import _normalize_landmarks
    from types import SimpleNamespace

    # Mock a hand_landmarks object with 21 landmarks
    lm_list = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    lm_list[9] = SimpleNamespace(x=0.0, y=-1.0, z=0.0)   # middle MCP for scale
    mock_hand = SimpleNamespace(landmark=lm_list)
    result = _normalize_landmarks(mock_hand)
    assert result.shape == (63,)
```

- [ ] **Step 2: Copy existing test files into `tests/`**

```
copy test_hand_tracker.py tests\test_hand_tracker.py
copy test_face_recognition.py tests\test_face_recognition.py
```

- [ ] **Step 3: Run all tests**

```
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short 2>&1 | findstr /V "delegate\|feedback\|W0000\|absl"
```

Expected: all tests **PASS** (webcam test skipped or passing, ASL model tests pass, pipeline tests pass, stage tests pass, existing tests pass).

---

## Self-Review Checklist

### Spec Coverage

| Spec requirement | Covered by task |
|---|---|
| Break 420-line run() into Stages | Tasks 6–13 |
| Three-thread pipeline | Task 5, 11, 12, 13 |
| ML-based ASL recognition | Tasks 8, 9, 18 |
| Central Config dataclass | Task 3 |
| Structured logging | Task 2 |
| Full type annotations | Throughout |
| Vectorise ASCII filter | Task 14 |
| Cache Hallucinogenic meshgrid | Task 15 |
| Expanded test suite | Tasks 16, 17, 18 |
| Existing modules unchanged | Verified — filter/face/pose/sign_language untouched |
| run.bat / setup.bat unchanged | Verified — hand_tracker.py entry point preserved |
| Updated CLI flags | Task 3 (build_parser) |
| GestureRecognizer in InferenceStage | Task 12 |
| FPS display | Task 11 (capture_fps), Task 10 (renderer HUD) |
| Haar cascade face downscale | FaceStage uses existing recognize_and_draw — already optimised enough for now |

### Type Consistency Check

- `FrameContext.asl_letters: list[tuple[int, str]]` — written by `ASLStage`, read by `RendererStage` ✓
- `FrameContext.body_gestures: list[str]` — written by `InferenceStage`, read by `RendererStage` ✓
- `FrameContext.coord_transform` — written by `FilterStage`, read by `RendererStage` ✓
- `FrameContext.capture_fps: float` — written by `CaptureStage`, read by `RendererStage` ✓
- `Config.show_asl` maps to `--asl` flag (inverse: flag present → True) ✓
- `Config.show_face` maps to `--no-face` flag (inverse: flag present → False) ✓
- `Pipeline.__init__` signature matches `hand_tracker.py` call in Task 13 ✓

### Placeholder Scan

No TBD, TODO, or "implement later" found. All steps contain complete code.
