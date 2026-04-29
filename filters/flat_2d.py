import cv2
import numpy as np


# Controls how flat/simplified the colors look (lower = fewer distinct colors)
_COLOR_LEVELS = 6
# Bilateral filter parameters: higher d = stronger smoothing
_BILATERAL_D = 9
_BILATERAL_SIGMA = 75
# Edge thickness via dilation
_EDGE_DILATE = 1


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    # Smooth colors with bilateral filter to preserve hard edges
    smooth = cv2.bilateralFilter(frame, _BILATERAL_D, _BILATERAL_SIGMA, _BILATERAL_SIGMA)

    # Quantize colors so they look flat/solid like a 2D illustration
    step = 256 // _COLOR_LEVELS
    flat = (smooth // step * step + step // 2).astype(np.uint8)

    # Build a dark outline from edges detected on the grayscale smoothed image
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=9, C=2,
    )
    # Slightly thicken the edges
    kernel = np.ones((_EDGE_DILATE, _EDGE_DILATE), np.uint8)
    edges = cv2.erode(edges, kernel)

    # Composite: keep flat colors where edges are white, black where edges are dark
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(flat, edges_bgr)
    return result


FILTER = {"name": "Flat 2D", "apply": _apply}
