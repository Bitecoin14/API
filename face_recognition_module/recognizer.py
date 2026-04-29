from typing import Dict, Optional

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
assert not _face_cascade.empty(), "Haar cascade failed to load — check OpenCV data path"

_BOX_COLOR = (200, 200, 200)    # BGR light-grey — subtle against filters
_BOX_THICKNESS = 1
_TAG_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TAG_SCALE = 0.45
_TAG_THICKNESS = 1
_TAG_PAD = 4                     # px gap between box top and text baseline
_MAX_DISTANCE = 80.0             # LBPH distance — lower = better match; reject above this


def recognize_and_draw(
    frame: np.ndarray,
    recognizer: Optional[cv2.face.LBPHFaceRecognizer],
    label_map: Dict[int, str],
) -> np.ndarray:
    """Detect faces in frame, optionally match them, draw thin labelled boxes.

    When no faces are detected the frame is returned unmodified.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    faces = np.array(faces)

    if faces.size == 0:
        return frame

    for (x, y, w, h) in faces:
        name = _identify(gray[y : y + h, x : x + w], recognizer, label_map)
        cv2.rectangle(frame, (x, y), (x + w, y + h), _BOX_COLOR, _BOX_THICKNESS)
        label_y = max(y - _TAG_PAD, 12)
        (text_w, _), _ = cv2.getTextSize(name, _TAG_FONT, _TAG_SCALE, _TAG_THICKNESS)
        label_x = max(0, min(x, frame.shape[1] - text_w))
        cv2.putText(
            frame, name, (label_x, label_y),
            _TAG_FONT, _TAG_SCALE, _BOX_COLOR, _TAG_THICKNESS, cv2.LINE_AA,
        )

    return frame


def _identify(
    face_crop: np.ndarray,
    recognizer: Optional[cv2.face.LBPHFaceRecognizer],
    label_map: Dict[int, str],
) -> str:
    if recognizer is None or not label_map:
        return "Unknown"
    label, confidence = recognizer.predict(face_crop)
    if confidence > _MAX_DISTANCE:
        return "Unknown"
    return label_map.get(label, "Unknown")
