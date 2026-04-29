from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .types import DetectedFace, FaceAttributes

log = logging.getLogger("hand_tracker.face.detector")

# ---------------------------------------------------------------------------
# Optional heavy dependencies — degrade gracefully if missing
# ---------------------------------------------------------------------------
try:
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop
    _INSIGHTFACE_OK = True
except ImportError:
    FaceAnalysis = None  # type: ignore[assignment,misc]
    norm_crop = None     # type: ignore[assignment]
    _INSIGHTFACE_OK = False
    log.warning(
        "insightface is not installed. "
        "Install it with: pip install insightface onnxruntime"
    )

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_OK = True
except ImportError:
    _YOLO = None  # type: ignore[assignment,misc]
    _ULTRALYTICS_OK = False
    log.warning(
        "ultralytics is not installed. "
        "Install it with: pip install ultralytics"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _laplacian_blur(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quality_score(face_width: float, det_score: float,
                   blur_var: float, yaw: float) -> float:
    q_size = min(face_width / 200.0, 1.0) * 0.3
    q_det  = min((det_score - 0.5) / 0.5, 1.0) * 0.3
    q_blur = min(blur_var / 200.0, 1.0) * 0.25
    q_yaw  = max(0.0, (1.0 - abs(yaw) / 35.0)) * 0.15
    return min(q_size + q_det + q_blur + q_yaw, 1.0)


def _yaw_from_landmarks(kps: np.ndarray, face_width: float) -> float:
    """Rough yaw proxy from 5-point landmarks (left_eye=0, right_eye=1)."""
    left_eye_x  = float(kps[0, 0])
    right_eye_x = float(kps[1, 0])
    if face_width > 0:
        return (right_eye_x - left_eye_x) / face_width * 50.0
    return 0.0


# ---------------------------------------------------------------------------
# DualDetector
# ---------------------------------------------------------------------------

class DualDetector:
    def __init__(
        self,
        det_size: int = 640,
        cross_check_size: int = 320,
        min_face_px: int = 80,
        min_det_score: float = 0.65,
        max_yaw: float = 35.0,
        min_blur: float = 50.0,
        use_gpu: bool = True,
    ) -> None:
        self._det_size = det_size
        self._cross_check_size = cross_check_size
        self._min_face_px = min_face_px
        self._min_det_score = min_det_score
        self._max_yaw = max_yaw
        self._min_blur = min_blur

        # Logged-once flags to avoid log spam
        self._warned_no_insightface = False
        self._warned_no_yolo = False

        ctx_id = 0 if use_gpu else -1

        # --- Primary: InsightFace RetinaFace (buffalo_l) ---
        self._app: Optional[FaceAnalysis] = None
        if _INSIGHTFACE_OK:
            try:
                self._app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection"],
                )
                self._app.prepare(
                    ctx_id=ctx_id,
                    det_size=(det_size, det_size),
                )
                log.info("InsightFace buffalo_l detector ready (ctx_id=%d).", ctx_id)
            except Exception:
                log.exception(
                    "Failed to initialise InsightFace buffalo_l. "
                    "Ensure the model is downloaded."
                )
                self._app = None

        # --- Secondary: YOLOv8-face ---
        self._yolo = None
        if _ULTRALYTICS_OK:
            try:
                import os
                model_path = "yolov8n-face.pt"
                if not os.path.exists(model_path):
                    try:
                        from ultralytics.utils.downloads import attempt_download_asset
                        attempt_download_asset(model_path)
                    except Exception:
                        log.warning(
                            "Could not download yolov8n-face.pt — "
                            "running in single-detector mode."
                        )
                        model_path = None

                if model_path and os.path.exists(model_path):
                    self._yolo = _YOLO(model_path)
                    log.info("YOLOv8-face cross-checker loaded.")
                else:
                    log.warning(
                        "yolov8n-face.pt not found — "
                        "running in single-detector mode."
                    )
            except Exception:
                log.warning(
                    "Failed to load YOLOv8-face model — "
                    "running in single-detector mode.",
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        try:
            return self._detect_impl(frame)
        except Exception:
            log.exception("Unexpected error in DualDetector.detect().")
            return []

    def _detect_impl(self, frame: np.ndarray) -> list[DetectedFace]:
        if self._app is None:
            if not self._warned_no_insightface:
                log.warning(
                    "InsightFace is unavailable; returning no detections."
                )
                self._warned_no_insightface = True
            return []

        h, w = frame.shape[:2]

        # Resize once — shared buffer for both detectors
        scale = self._cross_check_size / max(h, w)
        small_w = int(w * scale)
        small_h = int(h * scale)
        frame_small = cv2.resize(frame, (small_w, small_h))

        # ---- Primary detection (InsightFace operates on full frame) ----
        insight_faces = self._app.get(frame)

        if not insight_faces:
            return []

        # ---- Secondary detection (YOLO on small frame) ----
        yolo_boxes: list[tuple[float, float, float, float]] = []
        if self._yolo is not None:
            try:
                results = self._yolo(
                    frame_small,
                    imgsz=self._cross_check_size,
                    verbose=False,
                    conf=0.35,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes.xyxy.cpu().numpy():
                        x1 = box[0] / scale
                        y1 = box[1] / scale
                        x2 = box[2] / scale
                        y2 = box[3] / scale
                        yolo_boxes.append((float(x1), float(y1),
                                           float(x2), float(y2)))
            except Exception:
                log.warning("YOLO inference failed; using single-detector mode.",
                            exc_info=True)

        use_yolo = self._yolo is not None and len(yolo_boxes) >= 0

        # ---- IoU consensus + quality gate ----
        results_out: list[DetectedFace] = []

        for iface in insight_faces:
            bbox_arr = iface.bbox  # float array [x1, y1, x2, y2]
            x1 = int(bbox_arr[0])
            y1 = int(bbox_arr[1])
            x2 = int(bbox_arr[2])
            y2 = int(bbox_arr[3])
            ibox = (float(x1), float(y1), float(x2), float(y2))

            # IoU consensus
            if use_yolo and yolo_boxes:
                best_iou = max(_iou(ibox, yb) for yb in yolo_boxes)
                if best_iou < 0.5:
                    continue  # not confirmed by YOLO — drop silently

            det_score = float(getattr(iface, "det_score", 0.0))
            face_width = x2 - x1

            # ---- Quality gate ----
            if face_width < self._min_face_px:
                continue
            if det_score < self._min_det_score:
                continue

            # Yaw
            pose = getattr(iface, "pose", None)
            kps  = getattr(iface, "kps", None)

            if pose is not None and len(pose) >= 1:
                yaw = float(pose[0])
            elif kps is not None and kps.shape[0] >= 2:
                yaw = _yaw_from_landmarks(kps, float(face_width))
            else:
                yaw = 0.0

            if abs(yaw) > self._max_yaw:
                continue

            # Face crop for blur estimation
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)
            crop_region = frame[y1c:y2c, x1c:x2c]

            if crop_region.size == 0:
                continue

            blur_var = _laplacian_blur(crop_region)
            if blur_var < self._min_blur:
                continue

            # Aligned crop (112×112)
            aligned_crop: Optional[np.ndarray] = None
            if norm_crop is not None and kps is not None and kps.shape == (5, 2):
                try:
                    aligned_crop = norm_crop(frame, kps.astype(np.float32))
                except Exception:
                    aligned_crop = None

            quality = _quality_score(face_width, det_score, blur_var, yaw)

            landmarks: Optional[np.ndarray] = (
                kps.astype(np.float32) if kps is not None else None
            )

            face = DetectedFace(
                bbox=(x1, y1, x2, y2),
                det_score=det_score,
                yaw=yaw,
                blur_var=blur_var,
                quality_score=quality,
                attributes=FaceAttributes(
                    quality=quality,
                    yaw_degrees=yaw,
                ),
                landmarks=landmarks,
                crop=aligned_crop,
            )
            results_out.append(face)

        return results_out

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._app = None
        self._yolo = None
        log.debug("DualDetector closed.")
