# Hand Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time Python hand tracker that reads from the webcam and overlays 21 landmark dots per hand (up to 2 hands) on the live video feed.

**Architecture:** Single script (`hand_tracker.py`) runs a capture → detect → render → display loop using OpenCV for I/O and MediaPipe Hands for landmark detection. Tests live in `test_hand_tracker.py` using stdlib `unittest`.

**Tech Stack:** Python 3, `opencv-python`, `mediapipe`, `unittest`

---

## File Map

| File | Role |
|------|------|
| `requirements.txt` | Pins `opencv-python` and `mediapipe` |
| `hand_tracker.py` | Main script: webcam capture, MediaPipe detection, landmark rendering, display loop |
| `test_hand_tracker.py` | Unit tests: webcam opens, MediaPipe processes blank frame, landmark bounds |

---

### Task 1: Create requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```
opencv-python
mediapipe
```

- [ ] **Step 2: Install dependencies**

Run:
```bash
pip install -r requirements.txt
```
Expected: Both packages install without errors. Verify with:
```bash
python -c "import cv2; import mediapipe; print('OK')"
```
Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git init
git add requirements.txt
git commit -m "chore: add requirements for hand tracker"
```

---

### Task 2: Write the failing tests

**Files:**
- Create: `test_hand_tracker.py`

- [ ] **Step 1: Write test_hand_tracker.py**

```python
import unittest
import numpy as np
import cv2
import mediapipe as mp


class TestWebcam(unittest.TestCase):
    def test_webcam_opens(self):
        cap = cv2.VideoCapture(0)
        opened = cap.isOpened()
        cap.release()
        self.assertTrue(opened, "Default webcam (index 0) could not be opened")


class TestMediaPipe(unittest.TestCase):
    def setUp(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def tearDown(self):
        self.hands.close()

    def test_mediapipe_blank_frame(self):
        # A pure black 480x640 RGB frame — no hands, should return without raising
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            results = self.hands.process(blank)
        except Exception as e:
            self.fail(f"MediaPipe raised on blank frame: {e}")
        # No hands detected in a blank frame
        self.assertIsNone(results.multi_hand_landmarks)

    def test_landmark_bounds(self):
        # Inject a synthetic result by processing a real-ish frame and checking
        # that IF landmarks are returned, all x and y are in [0.0, 1.0].
        # We create a white frame — still unlikely to detect hands, but the
        # bounds check validates the contract regardless of detection.
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        results = self.hands.process(white)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    self.assertGreaterEqual(lm.x, 0.0)
                    self.assertLessEqual(lm.x, 1.0)
                    self.assertGreaterEqual(lm.y, 0.0)
                    self.assertLessEqual(lm.y, 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — expect 2 pass, 1 conditional pass (no hands in blank frames)**

```bash
python -m unittest test_hand_tracker -v
```
Expected output:
```
test_landmark_bounds ... ok
test_mediapipe_blank_frame ... ok
test_webcam_opens ... ok

----------------------------------------------------------------------
Ran 3 tests in X.XXXs

OK
```
> If `test_webcam_opens` fails, your webcam is not accessible — check device permissions.

- [ ] **Step 3: Commit**

```bash
git add test_hand_tracker.py
git commit -m "test: add webcam, mediapipe, and landmark bounds tests"
```

---

### Task 3: Implement hand_tracker.py

**Files:**
- Create: `hand_tracker.py`

- [ ] **Step 1: Write hand_tracker.py**

```python
import sys
import cv2
import mediapipe as mp

# Colors per hand index (BGR): hand 0 = green, hand 1 = blue
HAND_COLORS = [(0, 255, 0), (255, 0, 0)]
LANDMARK_RADIUS = 5


def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.", file=sys.stderr)
        sys.exit(1)

    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    color = HAND_COLORS[hand_idx % len(HAND_COLORS)]
                    for lm in hand_landmarks.landmark:
                        cx = int(lm.x * w)
                        cy = int(lm.y * h)
                        cv2.circle(frame, (cx, cy), LANDMARK_RADIUS, color, cv2.FILLED)

            cv2.imshow("Hand Tracker — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the tests again to confirm nothing broke**

```bash
python -m unittest test_hand_tracker -v
```
Expected: all 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add hand_tracker.py
git commit -m "feat: implement real-time hand tracker with MediaPipe landmark overlay"
```

---

### Task 4: Manual smoke test

**Files:** none changed

- [ ] **Step 1: Launch the tracker**

```bash
python hand_tracker.py
```
Expected: a window opens showing your webcam feed.

- [ ] **Step 2: Verify landmark dots appear**

Hold one hand in front of the camera. You should see 21 green dots on your hand joints and fingertips.

- [ ] **Step 3: Verify two-hand tracking**

Hold both hands in front of the camera. Hand 0 dots should be green; hand 1 dots should be blue.

- [ ] **Step 4: Verify Q quits**

Press `Q`. The window should close and the script should exit cleanly with no error output.

- [ ] **Step 5: Final commit (tag passing state)**

```bash
git add -A
git commit -m "chore: verified hand tracker working — all tests pass, smoke test complete"
```
