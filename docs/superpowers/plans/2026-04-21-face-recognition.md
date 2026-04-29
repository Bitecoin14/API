# Face Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time face recognition to the hand tracker that draws a thin labelled box around each detected face, with a `known_faces/` folder the user populates with named sub-folders of photos to register new people.

**Architecture:** A standalone `face_recognition_module/` package handles all face logic: `loader.py` scans `known_faces/<Name>/` directories at startup to build an in-memory list of encodings, and `recognizer.py` detects faces in each frame, matches them, and draws the overlay. `hand_tracker.py` calls these two functions in its existing loop with minimal changes.

**Tech Stack:** Python 3, `face_recognition` (dlib-backed), OpenCV, NumPy, `pathlib`

---

## File Map

| Path | Status | Responsibility |
|------|--------|----------------|
| `known_faces/` | Create (dir) | User-managed; one sub-folder per person, images inside |
| `known_faces/README.txt` | Create | Instructions for adding new faces |
| `face_recognition_module/__init__.py` | Create | Public API: `load_known_faces`, `recognize_and_draw` |
| `face_recognition_module/loader.py` | Create | Scans `known_faces/`, returns `list[tuple[ndarray, str]]` |
| `face_recognition_module/recognizer.py` | Create | Detects, matches, draws boxes+tags on a frame |
| `hand_tracker.py` | Modify | Import and call face recognition in the main loop |
| `requirements.txt` | Modify | Add `face_recognition` |
| `test_face_recognition.py` | Create | Unit tests for loader and recognizer (no webcam required) |

---

## Prerequisites — install dlib + face_recognition

> **Windows note:** `face_recognition` depends on `dlib` which requires a C++ compiler.
> The easiest path is:
> ```
> pip install cmake
> pip install dlib
> pip install face_recognition
> ```
> If dlib fails to build, download a pre-built wheel from
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x and install it with
> `pip install <wheel_file>.whl` before running `pip install face_recognition`.

---

### Task 1: Update requirements and create folder skeleton

**Files:**
- Modify: `requirements.txt`
- Create: `known_faces/README.txt`
- Create: `face_recognition_module/__init__.py` (empty placeholder)
- Create: `face_recognition_module/loader.py` (empty placeholder)
- Create: `face_recognition_module/recognizer.py` (empty placeholder)

- [ ] **Step 1: Add face_recognition to requirements.txt**

Replace the contents of `requirements.txt` with:

```
opencv-python
mediapipe==0.10.9
face_recognition
```

- [ ] **Step 2: Create known_faces/README.txt**

Create `known_faces/README.txt` with the content below. The program reads this directory at startup — no code changes needed when adding new people.

```
HOW TO ADD A NEW PERSON
=======================

1. Create a sub-folder inside known_faces/ using the person's name.
   Examples:
     known_faces/Alice/
     known_faces/Bob_Smith/

2. Copy one or more clear, front-facing photos of that person into the folder.
   Supported formats: .jpg  .jpeg  .png

3. Restart hand_tracker.py. The program loads encodings at startup.

TIPS
----
- More photos (3-10) → better accuracy.
- Photos should show the face clearly, without heavy shadows or extreme angles.
- File names inside the folder don't matter.
- To remove a person, delete their folder and restart.
```

- [ ] **Step 3: Create placeholder module files**

Create `face_recognition_module/__init__.py`:
```python
from .loader import load_known_faces
from .recognizer import recognize_and_draw

__all__ = ["load_known_faces", "recognize_and_draw"]
```

Create `face_recognition_module/loader.py` (empty stub — filled in Task 2):
```python
from pathlib import Path
from typing import List, Tuple
import numpy as np

KNOWN_FACES_DIR = Path(__file__).parent.parent / "known_faces"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_known_faces(faces_dir: Path = KNOWN_FACES_DIR) -> List[Tuple[np.ndarray, str]]:
    return []
```

Create `face_recognition_module/recognizer.py` (empty stub — filled in Task 3):
```python
import numpy as np


def recognize_and_draw(frame: np.ndarray, known: list) -> np.ndarray:
    return frame
```

- [ ] **Step 4: Commit skeleton**

```bash
git add requirements.txt known_faces/README.txt face_recognition_module/
git commit -m "feat: scaffold face_recognition_module and known_faces directory"
```

---

### Task 2: Implement loader.py

**Files:**
- Modify: `face_recognition_module/loader.py`
- Test: `test_face_recognition.py`

- [ ] **Step 1: Write failing tests for the loader**

Create `test_face_recognition.py`:

```python
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestLoadKnownFaces(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_empty_dir_returns_empty_list(self):
        from face_recognition_module.loader import load_known_faces
        result = load_known_faces(self.tmp)
        self.assertEqual(result, [])

    def test_nonexistent_dir_is_created_and_returns_empty(self):
        from face_recognition_module.loader import load_known_faces
        new_dir = self.tmp / "faces"
        result = load_known_faces(new_dir)
        self.assertEqual(result, [])
        self.assertTrue(new_dir.exists())

    def test_ignores_non_image_files(self):
        from face_recognition_module.loader import load_known_faces
        person = self.tmp / "Alice"
        person.mkdir()
        (person / "notes.txt").write_text("not an image")
        result = load_known_faces(self.tmp)
        self.assertEqual(result, [])

    def test_ignores_non_directory_entries(self):
        from face_recognition_module.loader import load_known_faces
        (self.tmp / "stray_file.jpg").write_bytes(b"")
        result = load_known_faces(self.tmp)
        self.assertEqual(result, [])

    def test_result_is_list_of_tuples(self):
        # We can only verify structure here without a real face image.
        # A real integration test would need a known face photo.
        from face_recognition_module.loader import load_known_faces
        result = load_known_faces(self.tmp)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            enc, name = item
            self.assertIsInstance(enc, np.ndarray)
            self.assertIsInstance(name, str)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to confirm they fail (or pass on stub)**

```bash
python -m pytest test_face_recognition.py -v
```

Expected: `test_empty_dir_returns_empty_list` PASS (stub returns []), others also PASS on stub — all should pass except `test_result_is_list_of_tuples` which trivially passes on empty list. That's fine; real validation happens via integration.

- [ ] **Step 3: Implement loader.py**

Replace `face_recognition_module/loader.py` with:

```python
from pathlib import Path
from typing import List, Tuple

import face_recognition
import numpy as np

KNOWN_FACES_DIR = Path(__file__).parent.parent / "known_faces"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_known_faces(faces_dir: Path = KNOWN_FACES_DIR) -> List[Tuple[np.ndarray, str]]:
    """Scan faces_dir for sub-folders; each sub-folder name is the person's name.
    Returns a list of (128-d encoding, name) pairs ready for comparison."""
    if not faces_dir.exists():
        faces_dir.mkdir(parents=True)
        return []

    known: List[Tuple[np.ndarray, str]] = []

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known.append((encodings[0], name))
            else:
                print(f"[face loader] No face found in {img_path.name} — skipping.")

    print(f"[face loader] Loaded {len(known)} encoding(s) for "
          f"{len({n for _, n in known})} person(s).")
    return known
```

- [ ] **Step 4: Run tests again**

```bash
python -m pytest test_face_recognition.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add face_recognition_module/loader.py test_face_recognition.py
git commit -m "feat: implement known-faces loader with per-person folder scanning"
```

---

### Task 3: Implement recognizer.py

**Files:**
- Modify: `face_recognition_module/recognizer.py`
- Modify: `test_face_recognition.py` (add recognizer tests)

- [ ] **Step 1: Write failing tests for the recognizer**

Append to `test_face_recognition.py` (inside the file, before `if __name__ == "__main__"`):

```python
class TestRecognizeAndDraw(unittest.TestCase):
    def _blank(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_ndarray_same_shape(self):
        from face_recognition_module.recognizer import recognize_and_draw
        frame = self._blank()
        result = recognize_and_draw(frame, [])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (480, 640, 3))

    def test_no_crash_on_empty_known_list(self):
        from face_recognition_module.recognizer import recognize_and_draw
        frame = self._blank()
        try:
            recognize_and_draw(frame, [])
        except Exception as e:
            self.fail(f"recognize_and_draw raised on empty known list: {e}")

    def test_no_crash_on_no_faces_in_frame(self):
        from face_recognition_module.recognizer import recognize_and_draw
        dummy_enc = np.zeros(128, dtype=np.float64)
        known = [(dummy_enc, "Alice")]
        frame = self._blank()
        try:
            recognize_and_draw(frame, known)
        except Exception as e:
            self.fail(f"recognize_and_draw raised with no faces in frame: {e}")

    def test_frame_is_not_mutated_when_no_faces(self):
        from face_recognition_module.recognizer import recognize_and_draw
        frame = self._blank()
        original = frame.copy()
        recognize_and_draw(frame, [])
        np.testing.assert_array_equal(frame, original)
```

- [ ] **Step 2: Run tests to confirm new tests fail**

```bash
python -m pytest test_face_recognition.py::TestRecognizeAndDraw -v
```

Expected: `test_returns_ndarray_same_shape` PASS (stub returns frame), others PASS — all pass on stub. That is expected; the real guarantee is that the full implementation doesn't break the contract.

- [ ] **Step 3: Implement recognizer.py**

Replace `face_recognition_module/recognizer.py` with:

```python
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np

# Visual style
_BOX_COLOR = (200, 200, 200)   # BGR light-grey — subtle, doesn't clash with hand markers
_BOX_THICKNESS = 1
_TAG_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TAG_SCALE = 0.45
_TAG_THICKNESS = 1
_TAG_PAD = 4                   # pixels between box top and text baseline

# Resize factor for face detection — smaller = faster, less accurate at distance
_DETECT_SCALE = 0.5
_MATCH_TOLERANCE = 0.55        # lower = stricter. 0.6 is face_recognition default


def recognize_and_draw(
    frame: np.ndarray,
    known: List[Tuple[np.ndarray, str]],
) -> np.ndarray:
    """Detect faces in frame, match against known encodings, draw labelled boxes.

    Does NOT modify frame in-place when no faces are found.
    Returns the annotated frame (same object, modified when faces detected).
    """
    h, w = frame.shape[:2]
    sh, sw = int(h * _DETECT_SCALE), int(w * _DETECT_SCALE)
    small = cv2.resize(frame, (sw, sh))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_small, model="hog")
    if not locations:
        return frame

    encodings = face_recognition.face_encodings(rgb_small, locations)
    known_encs = [k[0] for k in known] if known else []

    inv = 1.0 / _DETECT_SCALE
    for enc, (top, right, bottom, left) in zip(encodings, locations):
        # Scale coordinates back to original frame size
        top    = int(top    * inv)
        right  = int(right  * inv)
        bottom = int(bottom * inv)
        left   = int(left   * inv)

        name = _match(enc, known_encs, known)

        cv2.rectangle(frame, (left, top), (right, bottom), _BOX_COLOR, _BOX_THICKNESS)
        label_y = max(top - _TAG_PAD, 12)
        cv2.putText(
            frame, name, (left, label_y),
            _TAG_FONT, _TAG_SCALE, _BOX_COLOR, _TAG_THICKNESS, cv2.LINE_AA,
        )

    return frame


def _match(enc: np.ndarray, known_encs: list, known: list) -> str:
    if not known_encs:
        return "Unknown"
    matches = face_recognition.compare_faces(known_encs, enc, tolerance=_MATCH_TOLERANCE)
    if True not in matches:
        return "Unknown"
    distances = face_recognition.face_distance(known_encs, enc)
    best = int(np.argmin(distances))
    return known[best][1] if matches[best] else "Unknown"
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest test_face_recognition.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add face_recognition_module/recognizer.py test_face_recognition.py
git commit -m "feat: implement face recognizer with thin box and name tag overlay"
```

---

### Task 4: Integrate face recognition into hand_tracker.py

**Files:**
- Modify: `hand_tracker.py`

- [ ] **Step 1: Add imports and load known faces at startup**

In `hand_tracker.py`, add the import at the top (after existing imports):

```python
from face_recognition_module import load_known_faces, recognize_and_draw
```

In the `run()` function, after `hands = mp.solutions.hands.Hands(...)`, add:

```python
known_faces = load_known_faces()
```

- [ ] **Step 2: Call recognize_and_draw in the main loop**

In the main loop, after `frame = FILTERS[filter_index]["apply"](frame)` and **before** the hand-landmark drawing block, add:

```python
# --- Face recognition overlay ---
frame = recognize_and_draw(frame, known_faces)
```

The full modified `run()` function body should look like this (only the two changed sections shown in context):

```python
    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    known_faces = load_known_faces()          # <-- NEW

    filter_index = 0
    was_pinching  = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # --- Pinch detection (rising-edge only) ---
            pinching_now = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if _is_pinching(hand_landmarks, w, h):
                        pinching_now = True
                        break

            if pinching_now and not was_pinching:
                filter_index = (filter_index + 1) % len(FILTERS)
            was_pinching = pinching_now

            # --- Apply active filter ---
            frame = FILTERS[filter_index]["apply"](frame)

            # --- Face recognition overlay ---          # <-- NEW
            frame = recognize_and_draw(frame, known_faces)   # <-- NEW

            # --- Draw landmarks on top of the filtered frame ---
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    color = HAND_COLORS[hand_idx % len(HAND_COLORS)]
                    for lm in hand_landmarks.landmark:
                        cx, cy = _landmark_px(lm, w, h)
                        cv2.circle(frame, (cx, cy), LANDMARK_RADIUS, color, cv2.FILLED)

            # --- HUD ---
            label = f"Filter: {FILTERS[filter_index]['name']}  (pinch to cycle)"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Hand Tracker — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
```

- [ ] **Step 3: Run existing tests to check nothing broke**

```bash
python -m pytest test_hand_tracker.py test_face_recognition.py -v
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add hand_tracker.py
git commit -m "feat: integrate face recognition overlay into hand tracker main loop"
```

---

### Task 5: Smoke-test the full program

> This task is manual — no automated test can cover live webcam + face recognition together.

- [ ] **Step 1: Add a test photo of yourself**

```
mkdir known_faces\YourName
# Copy one clear front-facing photo of your face into known_faces\YourName\
```

- [ ] **Step 2: Run the program**

```bash
python hand_tracker.py
```

Expected:
- Console prints `[face loader] Loaded N encoding(s) for 1 person(s).`
- A thin grey box appears around your face with your name above it.
- Hand landmark dots still appear correctly over the filtered video.
- Pinch gesture still cycles filters.
- Q quits cleanly.

- [ ] **Step 3: Test "Unknown" label**

Cover the known face with your hand or present an unregistered face. Expected: box appears labelled `Unknown`.

- [ ] **Step 4: Add a second person and restart**

```
mkdir known_faces\FriendName
# Copy a photo
python hand_tracker.py
```

Expected: both names appear correctly without recompiling anything.

- [ ] **Step 5: Final commit if any last-minute fixes were made**

```bash
git add -A
git commit -m "chore: post-integration tweaks"
```
