from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .types import DetectedFace, FaceAttributes

log = logging.getLogger("hand_tracker.face.attributes")

# ---------------------------------------------------------------------------
# Module-level Haar cascade (loaded once)
# ---------------------------------------------------------------------------
_GLASSES_CASCADE: Optional[cv2.CascadeClassifier] = None
_GLASSES_CASCADE_LOADED = False


def _get_glasses_cascade() -> Optional[cv2.CascadeClassifier]:
    global _GLASSES_CASCADE, _GLASSES_CASCADE_LOADED
    if _GLASSES_CASCADE_LOADED:
        return _GLASSES_CASCADE
    _GLASSES_CASCADE_LOADED = True
    try:
        path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        cc = cv2.CascadeClassifier(path)
        if cc.empty():
            log.warning(
                "haarcascade_eye_tree_eyeglasses.xml could not be loaded "
                "from cv2.data.haarcascades — glasses detection disabled."
            )
            _GLASSES_CASCADE = None
        else:
            _GLASSES_CASCADE = cc
    except Exception:
        log.warning("Failed to load glasses Haar cascade.", exc_info=True)
        _GLASSES_CASCADE = None
    return _GLASSES_CASCADE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_attributes(
    face: DetectedFace,
    frame: np.ndarray,
    insight_face_obj=None,
    enrollment_metadata: dict | None = None,
) -> FaceAttributes:
    """
    Derive FaceAttributes from a DetectedFace.

    Parameters
    ----------
    face:
        The DetectedFace (must have .bbox and ideally .crop).
    frame:
        The original BGR frame (used for Haar-cascade glasses check on the
        face ROI rather than the 112×112 aligned crop, which is more reliable
        for the cascade).
    insight_face_obj:
        Raw InsightFace face object (may be None).
    enrollment_metadata:
        Dict loaded from gallery metadata.json (may be None).
    """
    quality = compute_quality_score(face)

    glasses_detected = _detect_glasses(face, frame)

    glasses_change = False
    if enrollment_metadata is not None:
        enrolled_glasses = bool(enrollment_metadata.get("has_glasses", False))
        if enrolled_glasses != glasses_detected:
            glasses_change = True

    heavy_makeup = _detect_heavy_makeup(face)

    return FaceAttributes(
        quality=quality,
        glasses_detected=glasses_detected,
        glasses_change=glasses_change,
        heavy_makeup=heavy_makeup,
        yaw_degrees=face.yaw,
    )


def compute_quality_score(face: DetectedFace) -> float:
    x1, y1, x2, y2 = face.bbox
    face_width = float(x2 - x1)

    q_size = min(face_width / 200.0, 1.0) * 0.3
    q_det  = min((face.det_score - 0.5) / 0.5, 1.0) * 0.3
    q_blur = min(face.blur_var / 200.0, 1.0) * 0.25
    q_yaw  = max(0.0, (1.0 - abs(face.yaw) / 35.0)) * 0.15
    return min(q_size + q_det + q_blur + q_yaw, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_glasses(face: DetectedFace, frame: np.ndarray) -> bool:
    """
    Attempt glasses detection via Haar cascade on the face ROI from the
    original frame.  Falls back to False on any failure.
    """
    cascade = _get_glasses_cascade()
    if cascade is None:
        return False

    try:
        x1, y1, x2, y2 = face.bbox
        h_frame, w_frame = frame.shape[:2]
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w_frame, x2)
        y2c = min(h_frame, y2)

        roi = frame[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return False

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        glasses_rects = cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 10),
        )
        return len(glasses_rects) > 0
    except Exception:
        log.debug("Glasses detection failed.", exc_info=True)
        return False


def _detect_heavy_makeup(face: DetectedFace) -> bool:
    """
    Heuristic makeup detection on the 112×112 aligned crop.
    Uses mean HSV saturation in the lip region (rows 80-105, cols 30-82).
    Returns True if mean saturation exceeds threshold.
    """
    crop = face.crop
    if crop is None or crop.shape[:2] != (112, 112):
        return False

    try:
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Saturation channel of lip region
        lip_region = crop_hsv[80:105, 30:82, 1].astype(np.float32)
        if lip_region.size == 0:
            return False
        mean_sat = float(lip_region.mean())
        return mean_sat > 75.0
    except Exception:
        log.debug("Makeup detection failed.", exc_info=True)
        return False
