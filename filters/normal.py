import numpy as np


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    return frame


FILTER = {"name": "Normal", "apply": _apply}
