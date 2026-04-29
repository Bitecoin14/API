# Opt-In Feature Flags + Face Recognition Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Invert all feature flags from opt-out (`--no-face`, `--no-pose`, etc.) to opt-in (`--face`, `--pose`, etc.), with each mode auto-enabling only its defining feature, while confirming the existing multi-face real-time recognition pipeline wires correctly under the new system.

**Architecture:** `core/config.py` is the sole source of truth for CLI flags and defaults; `hand_tracker.py` assembles the render pipeline conditionally based on config; all `face/` modules and `stages/face_stage.py` are untouched. Mode `hand` auto-enables hand skeleton; mode `face` auto-enables face recognition; all other features require explicit flags.

**Tech Stack:** Python 3.10+, argparse, pytest, OpenCV, MediaPipe, InsightFace (ONNX), YOLOv8

---

## File Map

| File | Action | What changes |
|---|---|---|
| `tests/test_config_flags.py` | **Create** | New test module for Config.from_args opt-in behavior |
| `core/config.py` | **Modify** | Defaults, `from_args`, `build_parser` |
| `hand_tracker.py` | **Modify** | `_run_hand_mode` conditionally assembles render stages |

---

## Task 1: Write Failing Tests for New CLI Behavior

**Files:**
- Create: `tests/test_config_flags.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_config_flags.py
"""Tests for opt-in feature flag behavior introduced in 2026-04-29 redesign."""
import pytest
from core.config import Config, build_parser


def _parse(*args) -> Config:
    parser = build_parser()
    ns = parser.parse_args(list(args))
    return Config.from_args(ns)


class TestHandModeDefaults:
    def test_hand_mode_enables_hand_skeleton(self):
        cfg = _parse()
        assert cfg.show_hand_skeleton is True

    def test_hand_mode_pose_off_by_default(self):
        cfg = _parse()
        assert cfg.show_pose is False

    def test_hand_mode_face_off_by_default(self):
        cfg = _parse()
        assert cfg.show_face is False

    def test_hand_mode_blur_off_by_default(self):
        cfg = _parse()
        assert cfg.show_blur is False

    def test_hand_mode_asl_off_by_default(self):
        cfg = _parse()
        assert cfg.show_asl is False


class TestOptInFlags:
    def test_face_flag_enables_face(self):
        cfg = _parse("--face")
        assert cfg.show_face is True

    def test_pose_flag_enables_pose(self):
        cfg = _parse("--pose")
        assert cfg.show_pose is True

    def test_blur_flag_enables_blur(self):
        cfg = _parse("--blur")
        assert cfg.show_blur is True

    def test_asl_flag_enables_asl(self):
        cfg = _parse("--asl")
        assert cfg.show_asl is True

    def test_multiple_flags_together(self):
        cfg = _parse("--face", "--pose", "--blur")
        assert cfg.show_face is True
        assert cfg.show_pose is True
        assert cfg.show_blur is True
        assert cfg.show_hand_skeleton is True  # auto-on in hand mode

    def test_hand_skeleton_still_on_with_other_flags(self):
        cfg = _parse("--face", "--asl")
        assert cfg.show_hand_skeleton is True


class TestFaceMode:
    def test_face_mode_auto_enables_face(self):
        cfg = _parse("--mode", "face")
        assert cfg.show_face is True

    def test_face_mode_disables_hand_skeleton(self):
        cfg = _parse("--mode", "face")
        assert cfg.show_hand_skeleton is False

    def test_face_mode_pose_still_off_by_default(self):
        cfg = _parse("--mode", "face")
        assert cfg.show_pose is False


class TestRemovedFlags:
    """Ensure the old --no-* flags are gone (argparse exits with error code 2)."""

    def test_no_hand_skeleton_flag_removed(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--no-hand-skeleton"])
        assert exc_info.value.code == 2

    def test_no_pose_flag_removed(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--no-pose"])
        assert exc_info.value.code == 2

    def test_no_face_flag_removed(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--no-face"])
        assert exc_info.value.code == 2

    def test_no_blur_flag_removed(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--no-blur"])
        assert exc_info.value.code == 2
```

- [ ] **Step 2: Run the tests and confirm they all fail**

```
cd "C:\Users\New User\OneDrive\Ambiente de Trabalho\Claude\Hand Tracking"
python -m pytest tests/test_config_flags.py -v
```

Expected: All tests **FAIL**. Failures like `AssertionError: assert True is False` (defaults wrong) and `Failed: DID NOT RAISE` (old flags still present) confirm the tests are exercising the right behavior.

---

## Task 2: Update `core/config.py`

**Files:**
- Modify: `core/config.py`

- [ ] **Step 1: Change Config dataclass defaults to False**

In `core/config.py`, change the `Config` dataclass feature flag defaults:

```python
# Feature flags — all off by default; mode logic in from_args sets implied defaults
show_hand_skeleton: bool = False
show_pose: bool = False
show_face: bool = False
show_blur: bool = False
show_asl: bool = False
```

(Lines 17–21 currently read `True` for hand_skeleton, pose, face, blur. Change all four to `False`.)

- [ ] **Step 2: Update `Config.from_args` to use mode-implied logic**

Replace the `from_args` body (lines 46–64) with:

```python
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
        show_hand_skeleton=(ns.mode == "hand"),
        show_face=ns.face or (ns.mode == "face"),
        show_pose=ns.pose,
        show_blur=ns.blur,
        show_asl=ns.asl,
        mode=ns.mode,
        enroll_folder=getattr(ns, "from_folder", None),
        filter_name=ns.filter,
        log_level=ns.log_level,
    )
```

- [ ] **Step 3: Update `build_parser` — remove `--no-*` flags, add `--face`, `--pose`, `--blur`**

Replace the `build_parser` function (lines 104–152) with:

```python
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hand_tracker.py",
        description="Real-time hand + body tracker with face recognition, filters, and overlays.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python hand_tracker.py                        # hand skeleton only",
            "  python hand_tracker.py --face                 # hand skeleton + face recognition",
            "  python hand_tracker.py --face --pose --blur   # hand + face + pose + blur",
            "  python hand_tracker.py --mode face            # face recognition only",
            "  python hand_tracker.py --mode enroll          # enroll new person",
            "  python hand_tracker.py --mode enroll --from-folder ./photos/",
            "  python hand_tracker.py --mode audit           # audit gallery for confusable pairs",
            "  python hand_tracker.py --filter Mosaic --resolution 320x240",
            "  python hand_tracker.py --list-filters",
        ]),
    )
    ap.add_argument("-h", "-help", "--help", action="help",
                    default=argparse.SUPPRESS, help="Show this help message and exit.")
    ap.add_argument("--mode", default="hand",
                    choices=["hand", "face", "enroll", "audit"],
                    help="Operating mode (default: hand).")
    ap.add_argument("--from-folder", default=None, metavar="PATH",
                    help="Batch-enroll from a folder of images (enroll mode only).")
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
    ap.add_argument("--face", action="store_true", default=False,
                    help="Enable face recognition overlay (colored square + name label).")
    ap.add_argument("--pose", action="store_true", default=False,
                    help="Enable body pose detection + gesture control.")
    ap.add_argument("--blur", action="store_true", default=False,
                    help="Enable middle-finger blur effect.")
    ap.add_argument("--asl", action="store_true", default=False,
                    help="Enable ASL fingerspelling recognition.")
    return ap
```

- [ ] **Step 4: Run new tests and confirm they pass**

```
python -m pytest tests/test_config_flags.py -v
```

Expected: All 18 tests **PASS**.

---

## Task 3: Update `hand_tracker.py` — Conditional Stage Assembly

**Files:**
- Modify: `hand_tracker.py`

- [ ] **Step 1: Replace `_run_hand_mode` with conditional stage assembly**

Replace the `_run_hand_mode` function (lines 19–35) with:

```python
def _run_hand_mode(config: Config) -> None:
    """Standard hand-tracking pipeline. Stages added only when their feature is enabled."""
    render_stages = [FilterStage(initial_name=config.filter_name)]
    if config.show_face:
        render_stages.append(FaceStage(
            gallery_dir=config.gallery_dir,
            models_dir=config.models_dir,
            known_faces_dir=config.known_faces_dir,
        ))
    if config.show_asl:
        render_stages.append(ASLStage(model_path=config.asl_model_path))
    render_stages.append(RendererStage())

    pipeline = Pipeline(
        capture_stage=CaptureStage(config),
        inference_stage=InferenceStage(config),
        render_stages=render_stages,
    )
    pipeline.run()
```

- [ ] **Step 2: Run the full test suite to confirm no regressions**

```
python -m pytest tests/ -v
```

Expected: All tests **PASS** (65 existing + 18 new = 83 total).

If `test_stages.py::TestFaceStage::test_returns_ctx` fails: it uses `Config()` default which now has `show_face=False`, so `FaceStage.process()` returns early — `result is ctx` should still be `True`. If it fails for another reason, inspect the error.

---

## Task 4: Verify End-to-End Integration

**Files:** none (manual verification)

- [ ] **Step 1: Verify `--help` output shows correct flags**

```
python hand_tracker.py --help
```

Expected output includes:
```
  --face        Enable face recognition overlay (colored square + name label).
  --pose        Enable body pose detection + gesture control.
  --blur        Enable middle-finger blur effect.
  --asl         Enable ASL fingerspelling recognition.
```

Expected output does NOT include: `--no-hand-skeleton`, `--no-pose`, `--no-face`, `--no-blur`

- [ ] **Step 2: Confirm default mode shows hand skeleton only**

```
python hand_tracker.py
```

Expected: webcam opens, hand skeleton overlaid, no face boxes, no pose, no blur.
Press `q` to quit.

- [ ] **Step 3: Confirm face recognition draws colored squares with names**

```
python hand_tracker.py --mode face
```

Expected: webcam opens, colored bounding squares appear around detected faces (green = confirmed, grey = unknown), name label displayed above each square. Multiple faces handled simultaneously.
Press `q` to quit.

- [ ] **Step 4: Confirm combined hand + face mode**

```
python hand_tracker.py --face
```

Expected: hand skeleton overlay + colored face squares both active simultaneously.
Press `q` to quit.
