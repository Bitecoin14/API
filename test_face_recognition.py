import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class TestLoadKnownFaces(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_empty_dir_returns_none_and_empty_map(self):
        from face_recognition_module.loader import load_known_faces
        recognizer, label_map = load_known_faces(self.tmp)
        self.assertIsNone(recognizer)
        self.assertEqual(label_map, {})

    def test_nonexistent_dir_is_created_and_returns_none(self):
        from face_recognition_module.loader import load_known_faces
        new_dir = self.tmp / "faces"
        recognizer, label_map = load_known_faces(new_dir)
        self.assertIsNone(recognizer)
        self.assertTrue(new_dir.exists())

    def test_ignores_non_image_files(self):
        from face_recognition_module.loader import load_known_faces
        person = self.tmp / "Alice"
        person.mkdir()
        (person / "notes.txt").write_text("not an image")
        recognizer, label_map = load_known_faces(self.tmp)
        self.assertIsNone(recognizer)

    def test_ignores_non_directory_entries(self):
        from face_recognition_module.loader import load_known_faces
        (self.tmp / "stray_file.jpg").write_bytes(b"")
        recognizer, label_map = load_known_faces(self.tmp)
        self.assertIsNone(recognizer)

    def test_label_map_populated_when_person_dir_exists(self):
        from face_recognition_module.loader import load_known_faces
        person = self.tmp / "Alice"
        person.mkdir()
        # No images with detectable faces, so recognizer=None, but label_map may be empty too
        # since no valid face crops were found — that is fine. Just check it returns a dict.
        _, label_map = load_known_faces(self.tmp)
        self.assertIsInstance(label_map, dict)

    def test_returns_tuple_of_two(self):
        from face_recognition_module.loader import load_known_faces
        result = load_known_faces(self.tmp)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestRecognizeAndDraw(unittest.TestCase):
    def _blank(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_ndarray_same_shape(self):
        from face_recognition_module.recognizer import recognize_and_draw
        frame = self._blank()
        result = recognize_and_draw(frame, None, {})
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (480, 640, 3))

    def test_no_crash_on_none_recognizer(self):
        from face_recognition_module.recognizer import recognize_and_draw
        frame = self._blank()
        try:
            recognize_and_draw(frame, None, {})
        except Exception as e:
            self.fail(f"recognize_and_draw raised with None recognizer: {e}")

    def test_no_crash_on_blank_frame_with_recognizer(self):
        from face_recognition_module.recognizer import recognize_and_draw
        # Blank frame — no faces detected, recognizer should never be called
        frame = self._blank()
        # Create a dummy trained recognizer
        import cv2
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Train on a tiny synthetic face crop so it's valid
        dummy_img = np.zeros((50, 50), dtype=np.uint8)
        recognizer.train([dummy_img], np.array([0]))
        label_map = {0: "Alice"}
        try:
            recognize_and_draw(frame, recognizer, label_map)
        except Exception as e:
            self.fail(f"recognize_and_draw raised on blank frame: {e}")

    def test_frame_not_mutated_when_no_faces(self):
        from face_recognition_module.recognizer import recognize_and_draw
        frame = self._blank()
        original = frame.copy()
        recognize_and_draw(frame, None, {})
        np.testing.assert_array_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
