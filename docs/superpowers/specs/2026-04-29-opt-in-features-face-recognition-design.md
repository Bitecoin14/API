# Design: Opt-In Feature Flags + Face Recognition Integration

**Date:** 2026-04-29  
**Status:** Approved

---

## Overview

Invert the feature-flag system so all features are disabled by default and must be explicitly enabled via CLI arguments. Confirm that the existing face recognition pipeline (multi-face, real-time, bounding-box rendering) is correctly wired under the new opt-in model.

---

## Problem Statement

Currently, all features (hand skeleton, pose, face recognition, blur) are **ON by default** and disabled via `--no-*` flags. The user must actively suppress features they don't want. The desired behavior is the opposite: features are **OFF by default** and enabled via explicit flags.

---

## Approach: Mode-Implied Defaults + Opt-In Flags

Each mode implies exactly one "free" feature (its defining capability). All other features require an explicit flag. All `--no-*` flags are removed.

### Mode Defaults

| Mode | Auto-enabled | Requires explicit flag |
|---|---|---|
| `--mode hand` (default) | hand skeleton | `--pose`, `--face`, `--blur`, `--asl` |
| `--mode face` | face recognition | nothing extra |
| `--mode enroll` | enrollment UI | — |
| `--mode audit` | gallery audit | — |

### New CLI Flags

**Added:**
- `--face` — Enable face recognition overlay (colored square + name label)
- `--pose` — Enable body pose detection + gesture control
- `--blur` — Enable middle-finger blur effect
- `--asl` — (unchanged, already opt-in)

**Removed:**
- `--no-hand-skeleton`
- `--no-pose`
- `--no-face`
- `--no-blur`

### Example Invocations

```
python hand_tracker.py                          # hand skeleton only
python hand_tracker.py --face                   # hand skeleton + face recognition
python hand_tracker.py --face --pose --blur     # hand skeleton + face + pose + blur
python hand_tracker.py --mode face              # face recognition only (no hands)
python hand_tracker.py --mode enroll            # enrollment (unchanged)
python hand_tracker.py --mode audit             # gallery audit (unchanged)
```

---

## Face Recognition Behavior

The `face/` subsystem is fully implemented and requires no changes. Behavior under the new opt-in system:

- **Multi-face**: `DualDetector` runs RetinaFace + YOLOv8 in parallel; handles multiple simultaneous faces per frame
- **Bounding box**: Each detected face gets a colored square:
  - Green = CONFIRMED
  - Yellow = SOFT_CONFIRMED
  - Orange = LOW_CONFIDENCE
  - Red = AMBIGUOUS
  - Grey = UNKNOWN
- **Name label**: Displayed above the square
- **Real-time temporal smoothing**: 15-frame voting window, 60% consensus threshold
- **Three-model ensemble**: antelopev2 (anchor) + AdaFace (quality specialist) + ElasticFace (robustness specialist)

---

## Architecture

### Config Changes (`core/config.py`)

`Config` dataclass: all `show_*` fields default to `False`.

`Config.from_args` mode-implied logic:
```python
show_hand_skeleton = (ns.mode == "hand")
show_face          = ns.face or (ns.mode == "face")
show_pose          = ns.pose
show_blur          = ns.blur
show_asl           = ns.asl
```

### Pipeline Changes (`hand_tracker.py`)

`_run_hand_mode` conditionally adds stages based on config:
```python
render_stages = [FilterStage(...)]
if config.show_face:
    render_stages.append(FaceStage(...))
if config.show_asl:
    render_stages.append(ASLStage(...))
render_stages.append(RendererStage())
```

---

## Change Surface

| File | Change |
|---|---|
| `core/config.py` | Defaults, `from_args`, `build_parser` |
| `hand_tracker.py` | Conditional stage assembly in `_run_hand_mode` |

All `face/` modules, `stages/face_stage.py`, `stages/renderer.py`, `core/pipeline.py`, `core/context.py` — **unchanged**.

---

## Testing

- Existing 65-test suite runs unchanged
- `Config.from_args` mode-implied logic is pure Python and unit-testable without camera/GPU
- Manual smoke test: `python hand_tracker.py --mode face` → confirm colored squares appear around detected faces in real time
