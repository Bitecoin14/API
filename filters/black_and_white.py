import cv2
import numpy as np


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


FILTER = {"name": "Black & White", "apply": _apply}
