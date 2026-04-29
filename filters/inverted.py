import cv2
import numpy as np


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    return cv2.bitwise_not(frame)


FILTER = {"name": "Inverted", "apply": _apply}
