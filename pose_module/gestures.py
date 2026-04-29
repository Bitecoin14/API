"""Static and temporal body-gesture recognizer using MediaPipe Pose landmarks.

Ten gestures:
    1. Waving          — one raised wrist oscillates horizontally
    2. Clapping        — both wrists meet near the chest
    3. Arms Stretched  — T-pose: arms level with shoulders, spread wide
    4. Hands Up        — both wrists above the nose
    5. Hand Raised     — exactly one wrist above the nose
    6. Arms Crossed    — wrists swap sides across the body center-line
    7. Hands on Hips   — both wrists close to the respective hip
    8. Bowing          — nose drops to or below hip level
    9. Victory Pose    — wrists above shoulders and spread in a V shape
   10. Shrugging       — shoulders pulled close to the ears
"""

from collections import deque
import numpy as np

# ── Landmark indices ───────────────────────────────────────────────────────
NOSE           = 0
LEFT_EAR       = 7
RIGHT_EAR      = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

_WAVE_WINDOW = 20   # frames of wrist-x history used for wave detection
_VIS         = 0.45


class GestureRecognizer:
    def __init__(self):
        self._lw_hist = deque(maxlen=_WAVE_WINDOW)   # left-wrist  x positions
        self._rw_hist = deque(maxlen=_WAVE_WINDOW)   # right-wrist x positions

    # ── Public API ─────────────────────────────────────────────────────────

    def recognize(self, pose_landmarks) -> list:
        """Return a list of gesture name strings currently detected."""
        if pose_landmarks is None:
            return []

        lm = pose_landmarks.landmark

        def v(idx):
            return lm[idx].visibility > _VIS

        def p(idx):
            return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

        # Feed wrist-x history for wave detection
        if v(LEFT_WRIST):
            self._lw_hist.append(lm[LEFT_WRIST].x)
        if v(RIGHT_WRIST):
            self._rw_hist.append(lm[RIGHT_WRIST].x)

        # Shared geometry
        shoulders_visible = v(LEFT_SHOULDER) and v(RIGHT_SHOULDER)
        hips_visible      = v(LEFT_HIP) and v(RIGHT_HIP)

        if shoulders_visible:
            sh_cx = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2
            sh_cy = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
            sh_w  = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
        else:
            sh_cx = sh_cy = 0.5
            sh_w = 0.3

        hip_cy = ((lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2) if hips_visible else None

        results = []

        # Evaluate all ten gestures
        wave = self._wave(lm, v)
        if wave:
            results.append(wave)

        if self._clapping(lm, v, sh_w, sh_cy):
            results.append("Clapping")

        if self._arms_stretched(lm, v, sh_cy, sh_w):
            results.append("Arms Stretched")

        both_up = v(LEFT_WRIST) and v(RIGHT_WRIST) and v(NOSE) and \
                  lm[LEFT_WRIST].y < lm[NOSE].y and lm[RIGHT_WRIST].y < lm[NOSE].y
        one_up  = not both_up and v(NOSE) and \
                  ((v(LEFT_WRIST) and lm[LEFT_WRIST].y < lm[NOSE].y) or
                   (v(RIGHT_WRIST) and lm[RIGHT_WRIST].y < lm[NOSE].y))

        if both_up:
            results.append("Hands Up")
        elif one_up:
            results.append("Hand Raised")

        if self._arms_crossed(lm, v, sh_cx, sh_cy):
            results.append("Arms Crossed")

        if self._hands_on_hips(lm, v):
            results.append("Hands on Hips")

        if hip_cy is not None and self._bowing(lm, v, hip_cy, sh_cy):
            results.append("Bowing")

        if self._victory(lm, v, sh_w):
            results.append("Victory Pose")

        if self._shrug(lm, v, hip_cy, sh_cy):
            results.append("Shrugging")

        return results

    # ── Individual detectors ───────────────────────────────────────────────

    def _wave(self, lm, v) -> str:
        """Return 'Waving' when either wrist oscillates horizontally above its shoulder."""
        def _check(hist, wrist_idx, shoulder_idx):
            if len(hist) < 10 or not v(wrist_idx) or not v(shoulder_idx):
                return False
            if lm[wrist_idx].y >= lm[shoulder_idx].y:
                return False                       # wrist must be raised
            xs    = list(hist)[-10:]
            diffs = [xs[i+1] - xs[i] for i in range(len(xs) - 1)]
            turns = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i+1] < 0)
            amp   = max(xs) - min(xs)
            return turns >= 3 and amp > 0.05

        if _check(self._rw_hist, RIGHT_WRIST, RIGHT_SHOULDER):
            return "Waving"
        if _check(self._lw_hist, LEFT_WRIST, LEFT_SHOULDER):
            return "Waving"
        return ""

    def _clapping(self, lm, v, sh_w, sh_cy) -> bool:
        if not (v(LEFT_WRIST) and v(RIGHT_WRIST)):
            return False
        wrist_sep = abs(lm[LEFT_WRIST].x - lm[RIGHT_WRIST].x)
        wrist_y   = (lm[LEFT_WRIST].y + lm[RIGHT_WRIST].y) / 2
        # Wrists nearly touching (< 35 % of shoulder width) at chest level
        return wrist_sep < sh_w * 0.35 and wrist_y > sh_cy - 0.08

    def _arms_stretched(self, lm, v, sh_cy, sh_w) -> bool:
        """T-pose: both wrists at shoulder height and far apart."""
        if not (v(LEFT_WRIST) and v(RIGHT_WRIST) and
                v(LEFT_ELBOW) and v(RIGHT_ELBOW)):
            return False
        lw_level = abs(lm[LEFT_WRIST].y  - sh_cy) < 0.13
        rw_level = abs(lm[RIGHT_WRIST].y - sh_cy) < 0.13
        spread   = abs(lm[LEFT_WRIST].x  - lm[RIGHT_WRIST].x)
        return lw_level and rw_level and spread > sh_w * 2.4

    def _arms_crossed(self, lm, v, sh_cx, sh_cy) -> bool:
        if not (v(LEFT_WRIST) and v(RIGHT_WRIST) and
                v(LEFT_ELBOW) and v(RIGHT_ELBOW)):
            return False
        # In camera view (person facing camera):
        # Person's LEFT body side appears at larger image-x.
        # Crossed: left wrist is at smaller x, right wrist at larger x.
        lw_crossed = lm[LEFT_WRIST].x  < sh_cx
        rw_crossed = lm[RIGHT_WRIST].x > sh_cx
        # Wrists at chest/torso height
        wrist_y = (lm[LEFT_WRIST].y + lm[RIGHT_WRIST].y) / 2
        chest   = sh_cy - 0.12 < wrist_y < sh_cy + 0.30
        return lw_crossed and rw_crossed and chest

    def _hands_on_hips(self, lm, v) -> bool:
        if not (v(LEFT_WRIST) and v(RIGHT_WRIST) and
                v(LEFT_HIP)   and v(RIGHT_HIP)):
            return False
        lw_hip = (abs(lm[LEFT_WRIST].x  - lm[LEFT_HIP].x)  < 0.12 and
                  abs(lm[LEFT_WRIST].y   - lm[LEFT_HIP].y)  < 0.18)
        rw_hip = (abs(lm[RIGHT_WRIST].x - lm[RIGHT_HIP].x) < 0.12 and
                  abs(lm[RIGHT_WRIST].y  - lm[RIGHT_HIP].y) < 0.18)
        return lw_hip and rw_hip

    def _bowing(self, lm, v, hip_cy, sh_cy) -> bool:
        if not v(NOSE):
            return False
        # Nose at or below hip level, and shoulders below their usual position
        return lm[NOSE].y > hip_cy - 0.08 and sh_cy > hip_cy - 0.18

    def _victory(self, lm, v, sh_w) -> bool:
        """V arms: wrists above their shoulders AND spread wide but NOT at head level."""
        if not (v(LEFT_WRIST) and v(RIGHT_WRIST) and
                v(LEFT_SHOULDER) and v(RIGHT_SHOULDER)):
            return False
        lw_up   = lm[LEFT_WRIST].y  < lm[LEFT_SHOULDER].y  - 0.12
        rw_up   = lm[RIGHT_WRIST].y < lm[RIGHT_SHOULDER].y - 0.12
        spread  = abs(lm[LEFT_WRIST].x - lm[RIGHT_WRIST].x)
        # Not "hands up" (wrists still below nose)
        below_nose = True
        if v(NOSE):
            below_nose = (lm[LEFT_WRIST].y > lm[NOSE].y or
                          lm[RIGHT_WRIST].y > lm[NOSE].y)
        return lw_up and rw_up and spread > sh_w * 1.8 and below_nose

    def _shrug(self, lm, v, hip_cy, sh_cy) -> bool:
        if not (v(LEFT_EAR) and v(RIGHT_EAR) and
                v(LEFT_SHOULDER) and v(RIGHT_SHOULDER)):
            return False
        ear_cy    = (lm[LEFT_EAR].y + lm[RIGHT_EAR].y) / 2
        sh_to_ear = sh_cy - ear_cy           # positive when shoulder below ear
        if hip_cy is None:
            return sh_to_ear < 0.12
        torso     = hip_cy - sh_cy           # height of torso
        return torso > 0 and sh_to_ear < torso * 0.28
