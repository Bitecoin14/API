import cv2
import numpy as np


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    return cv2.flip(frame, 0)


def _coord_transform(x: int, y: int, w: int, h: int) -> tuple:
    return x, h - 1 - y


FILTER = {"name": "Upside Down", "apply": _apply, "coord_transform": _coord_transform}
