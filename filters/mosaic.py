import cv2
import numpy as np

_BLOCK = 16  # pixel block size


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // _BLOCK, h // _BLOCK), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


FILTER = {"name": "Mosaic", "apply": _apply}
