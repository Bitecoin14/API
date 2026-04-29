# stages/capture.py
from __future__ import annotations
import logging
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from core.config import Config
from core.context import FrameContext

log = logging.getLogger("hand_tracker.capture")

_FPS_WINDOW = 30   # rolling window size for FPS calculation


class CaptureStage:
    def __init__(self, config: Config) -> None:
        self._config = config
        w, h = config.resolution
        self._cap = cv2.VideoCapture(config.camera)
        if not self._cap.isOpened():
            log.error("Cannot open camera %d", config.camera)
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._frame_id = 0
        self._ts_window: deque[float] = deque(maxlen=_FPS_WINDOW)
        log.info("Camera %d opened at %dx%d", config.camera, w, h)

    def read_frame(self) -> Optional[FrameContext]:
        ret, frame = self._cap.read()
        if not ret:
            log.warning("VideoCapture.read() returned False")
            return None

        ts = time.monotonic()
        self._ts_window.append(ts)

        fps = 0.0
        if len(self._ts_window) >= 2:
            elapsed = self._ts_window[-1] - self._ts_window[0]
            if elapsed > 0:
                fps = (len(self._ts_window) - 1) / elapsed

        ctx = FrameContext(
            raw_frame=frame,
            frame=frame.copy(),
            timestamp=ts,
            frame_id=self._frame_id,
            config=self._config,
            capture_fps=fps,
        )
        self._frame_id += 1
        log.debug("Captured frame %d  FPS=%.1f", self._frame_id, fps)
        return ctx

    def close(self) -> None:
        self._cap.release()
        log.info("Camera released")
