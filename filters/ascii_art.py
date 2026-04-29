# filters/ascii_art.py
import cv2
import numpy as np
from typing import Optional

_CHARS = ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'
_N = len(_CHARS)
_CELL_W = 8
_CELL_H = 12
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.32
_FONT_THICK = 1

# Pre-render each character as a _CELL_H x _CELL_W mask (0 or 1) once at import time.
_CHAR_MASKS: Optional[np.ndarray] = None   # shape: (_N, _CELL_H, _CELL_W)


def _build_masks() -> np.ndarray:
    masks = np.zeros((_N, _CELL_H, _CELL_W), dtype=np.uint8)
    for i, ch in enumerate(_CHARS):
        cell = np.zeros((_CELL_H, _CELL_W), dtype=np.uint8)
        cv2.putText(cell, ch, (0, _CELL_H - 2), _FONT, _FONT_SCALE, 255, _FONT_THICK)
        masks[i] = cell
    return masks


def _apply(frame: np.ndarray, **kwargs) -> np.ndarray:
    global _CHAR_MASKS
    if _CHAR_MASKS is None:
        _CHAR_MASKS = _build_masks()

    h, w = frame.shape[:2]
    cols = w // _CELL_W
    rows = h // _CELL_H

    # Resize to grid dimensions and get colour + brightness per cell
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)  # (rows, cols, 3)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)                         # (rows, cols)

    # Map each cell brightness → char index
    char_idx = (gray.astype(np.float32) / 255.0 * (_N - 1)).astype(np.int32)  # (rows, cols)

    # Build canvas: for each cell, paint the char glyph in the cell's colour
    canvas = np.zeros((rows * _CELL_H, cols * _CELL_W, 3), dtype=np.uint8)

    # Vectorised assembly: iterate over chars (70 unique values) rather than cells
    for char_val in range(_N):
        mask = _CHAR_MASKS[char_val]   # (_CELL_H, _CELL_W)
        # Find all cells that use this character
        positions = np.argwhere(char_idx == char_val)   # (k, 2) = (row, col) pairs
        if positions.size == 0:
            continue
        for row, col in positions:
            y0 = row * _CELL_H
            x0 = col * _CELL_W
            color = small[row, col].astype(np.float32)   # BGR
            patch = canvas[y0:y0 + _CELL_H, x0:x0 + _CELL_W]
            # Apply glyph: pixels where mask>0 get the cell colour
            patch[mask > 0] = color

    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)


FILTER = {"name": "ASCII", "apply": _apply}
