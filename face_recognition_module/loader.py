from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

KNOWN_FACES_DIR = Path(__file__).parent.parent / "known_faces"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
assert not _face_cascade.empty(), "Haar cascade failed to load — check OpenCV data path"


def load_known_faces(
    faces_dir: Path = KNOWN_FACES_DIR,
) -> Tuple[Optional[cv2.face.LBPHFaceRecognizer], Dict[int, str]]:
    """Scan faces_dir sub-folders (one per person) and train an LBPH recognizer.

    Returns (recognizer, label_map).
    recognizer is None when no usable face images are found.
    label_map maps integer label -> person name.
    """
    if not faces_dir.exists():
        faces_dir.mkdir(parents=True)
        return None, {}

    images, labels, label_map = [], [], {}
    label_id = 0

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        person_id = label_id
        label_id += 1
        label_map[person_id] = name

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                images.append(gray[y : y + h, x : x + w])
                labels.append(person_id)
                label_map[person_id] = name

    if not images:
        print("[face loader] No face crops found in training images.")
        return None, label_map

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    print(
        f"[face loader] Trained on {len(images)} face crop(s) "
        f"for {len(label_map)} person(s)."
    )
    return recognizer, label_map
