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
        # If landmarks are returned, all x and y must be in [0.0, 1.0].
        # White frame — unlikely to detect hands, but validates contract if detected.
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
