# filters/hallucinogenic.py
import time
import cv2
import numpy as np

_WAVE_AMP = 18
_WAVE_SPATIAL = 70
_WAVE_SPEED_X = 2.2
_WAVE_SPEED_Y = 2.7
_HUE_SPEED = 45
_SAT_BOOST = 1.9

# Cache base meshgrid per (h, w) — only re-created on resolution change.
_cache: dict = {}   # (h, w) → (map_x_base, map_y_base)


def _get_base_maps(h: int, w: int):
    key = (h, w)
    if key not in _cache:
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        map_x, map_y = np.meshgrid(xs, ys)
        _cache[key] = (map_x, map_y)
    return _cache[key]


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    h, w = frame.shape[:2]
    t = time.time()

    map_x_base, map_y_base = _get_base_maps(h, w)

    map_x = map_x_base + _WAVE_AMP * np.sin(
        2 * np.pi * map_y_base / _WAVE_SPATIAL + t * _WAVE_SPEED_X
    )
    map_y = map_y_base + _WAVE_AMP * np.sin(
        2 * np.pi * map_x_base / _WAVE_SPATIAL + t * _WAVE_SPEED_Y
    )

    distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)

    hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + (t * _HUE_SPEED) % 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * _SAT_BOOST, 0, 255)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


FILTER = {"name": "Hallucinogenic", "apply": _apply}
