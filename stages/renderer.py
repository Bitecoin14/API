# stages/renderer.py
from __future__ import annotations
import logging
import math
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

from core.context import FrameContext
from filters import ALWAYS_ON_OVERLAY
from pose_module import PoseDetector

log = logging.getLogger("hand_tracker.renderer")

_mp_hands = mp.solutions.hands

# Hand skeleton colors: hand 0 = green, hand 1 = blue-ish
_HAND_COLORS = [(0, 255, 0), (255, 80, 0)]
_BONE_COLORS  = [(0, 200, 0), (200, 60, 0)]
_LM_RADIUS = 6;  _MID_RADIUS = 4;  _BONE_THICK = 2

# Face recognition overlay colors (BGR) keyed by status name
_FACE_STATUS_COLORS = {
    "CONFIRMED":       (0, 220, 0),      # green
    "SOFT_CONFIRMED":  (0, 210, 240),    # yellow-ish (cyan-yellow in BGR)
    "LOW_CONFIDENCE":  (0, 140, 255),    # orange
    "AMBIGUOUS":       (0, 0, 220),      # red
    "UNKNOWN":         (130, 130, 130),  # grey
}
_FACE_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FACE_SCALE = 0.65
_FACE_THICK = 2

_MIDPOINT_CONNECTIONS = [
    (0,1),(0,5),(0,17),(1,2),(2,3),(3,4),
    (5,6),(5,9),(6,7),(7,8),(9,10),(9,13),
    (10,11),(11,12),(13,14),(13,17),(14,15),(15,16),
    (17,18),(18,19),(19,20),
]

# ASL overlay style
_SL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_SL_SCALE = 2.5;  _SL_THICK = 4
_SL_COLOR = (0, 240, 240);  _SL_BG = (0, 0, 0)

# Gesture panel style
_GP_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_GP_SCALE = 0.55;  _GP_THICK = 1
_GP_COLOR = (0, 255, 180);  _GP_BG = (0, 0, 0);  _GP_ALPHA = 0.55
_GP_WIDTH = 185;  _GP_LH = 22

# HUD style
_HUD_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.65;  _HUD_THICK = 2


def _lm_px(lm, w, h, transform=None):
    x, y = int(round(lm.x * w)), int(round(lm.y * h))
    return transform(x, y, w, h) if transform else (x, y)


class RendererStage:
    def __init__(self) -> None:
        self._pose_detector = PoseDetector()
        self._hand_open_flag = False   # for HUD indicator

    def process(self, ctx: FrameContext) -> FrameContext:
        h, w = ctx.frame.shape[:2]
        ct = ctx.coord_transform

        # 1. Middle-finger blur
        if ctx.config.show_blur:
            ctx.frame = ALWAYS_ON_OVERLAY(
                ctx.frame, hand_results=ctx.hand_results, coord_transform=ct
            )

        # 2. Face recognition boxes + HUD (drawn from FaceStage results)
        if ctx.config.show_face and ctx.face_results:
            for bbox, name, status in ctx.face_results:
                self._draw_face_box(ctx.frame, bbox, name, status)
            self._draw_face_hud(ctx.frame, len(ctx.face_results), ctx.capture_fps)

        # 3. Pose skeleton
        if ctx.config.show_pose:
            ctx.frame = self._pose_detector.draw(ctx.frame, ctx.pose_results, ct)

        # 4. Hand skeleton
        if ctx.config.show_hand_skeleton and ctx.hand_results and ctx.hand_results.multi_hand_landmarks:
            for idx, hand_lm in enumerate(ctx.hand_results.multi_hand_landmarks):
                lm   = hand_lm.landmark
                pts  = [_lm_px(lm[i], w, h, ct) for i in range(len(lm))]
                bone = _BONE_COLORS[idx % 2]
                col  = _HAND_COLORS[idx % 2]
                for a, b in _mp_hands.HAND_CONNECTIONS:
                    cv2.line(ctx.frame, pts[a], pts[b], bone, _BONE_THICK, cv2.LINE_AA)
                for pt in pts:
                    cv2.circle(ctx.frame, pt, _LM_RADIUS, col, cv2.FILLED)
                    cv2.circle(ctx.frame, pt, _LM_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)
                mid_col = tuple(int(c * 0.6) for c in col)
                for a, b in _MIDPOINT_CONNECTIONS:
                    mx = (pts[a][0] + pts[b][0]) // 2
                    my = (pts[a][1] + pts[b][1]) // 2
                    cv2.circle(ctx.frame, (mx, my), _MID_RADIUS, mid_col, cv2.FILLED)
                    cv2.circle(ctx.frame, (mx, my), _MID_RADIUS, (200, 200, 200), 1, cv2.LINE_AA)

        # 5. ASL letters
        if ctx.config.show_asl and ctx.asl_letters and ctx.hand_results and ctx.hand_results.multi_hand_landmarks:
            for hand_idx, letter in ctx.asl_letters:
                hand_lms_list = ctx.hand_results.multi_hand_landmarks
                if hand_idx >= len(hand_lms_list):
                    continue
                hand_lm = hand_lms_list[hand_idx]
                lm = hand_lm.landmark
                xs = [int(round(lm[i].x * w)) for i in range(len(lm))]
                ys = [int(round(lm[i].y * h)) for i in range(len(lm))]
                cx_pt = (min(xs) + max(xs)) // 2
                cy_pt = min(ys)
                if ct:
                    cx_pt, cy_pt = ct(cx_pt, cy_pt, w, h)
                    sign_y = cy_pt + 60
                else:
                    sign_y = max(cy_pt - 20, 40)
                (tw, th), base = cv2.getTextSize(letter, _SL_FONT, _SL_SCALE, _SL_THICK)
                tx = max(0, cx_pt - tw // 2)
                pad = 6
                cv2.rectangle(ctx.frame,
                               (tx - pad, sign_y - th - pad),
                               (tx + tw + pad, sign_y + base + pad),
                               _SL_BG, cv2.FILLED)
                cv2.putText(ctx.frame, letter, (tx, sign_y),
                            _SL_FONT, _SL_SCALE, _SL_COLOR, _SL_THICK, cv2.LINE_AA)

        # 6. Gesture panel
        if ctx.config.show_pose:
            self._draw_gesture_panel(ctx.frame, ctx.body_gestures)

        # 7. HUD
        self._draw_hud(ctx)

        return ctx

    def _draw_face_box(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        name: str,
        status,
    ) -> None:
        status_name = status.name if hasattr(status, "name") else str(status)
        color = _FACE_STATUS_COLORS.get(status_name, _FACE_STATUS_COLORS["UNKNOWN"])
        x1, y1, x2, y2 = (int(v) for v in bbox)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Name label background
        label = name if name not in ("?", "") else "Unknown"
        (tw, th), base = cv2.getTextSize(label, _FACE_FONT, _FACE_SCALE, _FACE_THICK)
        pad = 4
        lx, ly = x1, y1 - th - pad * 2
        if ly < 0:
            ly = y2 + pad
        cv2.rectangle(frame, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (lx + pad, ly + th + pad),
                    _FACE_FONT, _FACE_SCALE, (0, 0, 0), _FACE_THICK, cv2.LINE_AA)

    def _draw_face_hud(
        self,
        frame: np.ndarray,
        num_faces: int,
        fps: float,
    ) -> None:
        h = frame.shape[0]
        label = f"Faces: {num_faces}  {fps:.0f} FPS"
        cv2.putText(frame, label, (10, h - 14),
                    _FACE_FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def draw_enrollment_overlay(
        frame: np.ndarray,
        name: str,
        status_text: str,
    ) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"ENROLLMENT MODE — {name}", (10, 20),
                    _FACE_FONT, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, status_text, (10, 42),
                    _FACE_FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_gesture_panel(self, frame: np.ndarray, gestures: list[str]) -> None:
        h, w = frame.shape[:2]
        lines = gestures if gestures else ["No gesture"]
        ph = 28 + _GP_LH * len(lines)
        x0, y0 = w - _GP_WIDTH - 10, 10
        x1, y1 = x0 + _GP_WIDTH, y0 + ph
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _GP_BG, cv2.FILLED)
        cv2.addWeighted(overlay, _GP_ALPHA, frame, 1 - _GP_ALPHA, 0, frame)
        cv2.putText(frame, "Gestures", (x0 + 6, y0 + 14),
                    _GP_FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(frame, (x0 + 4, y0 + 17), (x1 - 4, y0 + 17), (60, 60, 60), 1)
        color = (140, 140, 140) if not gestures else _GP_COLOR
        for i, name in enumerate(lines):
            cv2.putText(frame, name, (x0 + 6, y0 + 30 + i * _GP_LH),
                        _GP_FONT, _GP_SCALE, color, _GP_THICK, cv2.LINE_AA)

    def _draw_hud(self, ctx: FrameContext) -> None:
        filter_name = ctx.active_filter.get("name", "—")
        fps_str = f"{ctx.capture_fps:.0f} FPS" if ctx.capture_fps else ""
        hud = f"Filter: {filter_name}  (open hand + pinch to cycle)  {fps_str}"
        cv2.putText(ctx.frame, hud, (10, 28),
                    _HUD_FONT, _HUD_SCALE, (255, 255, 255), _HUD_THICK, cv2.LINE_AA)
        if ctx.config.show_asl:
            cv2.putText(ctx.frame, "ASL: ON", (10, 54),
                        _HUD_FONT, 0.52, (0, 240, 240), 1, cv2.LINE_AA)

    def close(self) -> None:
        self._pose_detector.close()
