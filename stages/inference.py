# stages/inference.py
from __future__ import annotations
import logging
import time

import cv2
import mediapipe as mp

from core.config import Config
from core.context import FrameContext
from pose_module import PoseDetector, GestureRecognizer

log = logging.getLogger("hand_tracker.inference")

_mp_hands = mp.solutions.hands


class InferenceStage:
    """Owns all ML inference objects. Runs on the inference background thread."""

    def __init__(self, config: Config) -> None:
        cfg = config
        self._hands = _mp_hands.Hands(
            max_num_hands=2,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.hand_detection_conf,
            min_tracking_confidence=cfg.hand_tracking_conf,
        )
        if cfg.show_pose:
            self._pose = PoseDetector(
                min_detection_confidence=cfg.pose_detection_conf,
                min_tracking_confidence=cfg.pose_tracking_conf,
                model_complexity=cfg.model_complexity,
            )
            self._gesture = GestureRecognizer()
        else:
            self._pose = None
            self._gesture = None
        log.info("InferenceStage initialised (pose=%s)", cfg.show_pose)

    def process(self, ctx: FrameContext) -> FrameContext:
        t0 = time.monotonic()
        rgb = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB)

        ctx.hand_results = self._hands.process(rgb)

        if self._pose is not None:
            ctx.pose_results = self._pose.process(rgb)
            pose_lm = (ctx.pose_results.pose_landmarks
                       if ctx.pose_results else None)
            ctx.body_gestures = self._gesture.recognize(pose_lm)
        else:
            ctx.pose_results  = None
            ctx.body_gestures = []

        log.debug("Inference took %.1f ms", (time.monotonic() - t0) * 1000)
        return ctx

    def close(self) -> None:
        self._hands.close()
        if self._pose is not None:
            self._pose.close()
        log.info("InferenceStage closed")
