# stages/asl_stage.py
from __future__ import annotations
import logging
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.context import FrameContext

log = logging.getLogger("hand_tracker.asl")

_WRIST      = 0
_MIDDLE_MCP = 9
_WINDOW     = 12    # frames of letter history per hand
_MIN_VOTES  = 7     # minimum consistent frames to accept a letter


def _normalize_landmarks(hand_landmarks) -> np.ndarray:
    lm = hand_landmarks.landmark
    pts = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
    pts -= pts[_WRIST]
    scale = float(np.linalg.norm(pts[_MIDDLE_MCP]))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()   # 63 floats


class ASLStage:
    def __init__(self, model_path: Path = Path("models/asl_classifier.pkl")) -> None:
        self._predict: Optional[Callable] = None
        self._histories: list[deque] = [
            deque(maxlen=_WINDOW), deque(maxlen=_WINDOW)
        ]
        self._load_model(model_path)

    def _load_model(self, path: Path) -> None:
        if path.exists():
            try:
                import joblib
                model = joblib.load(path)
                self._predict = lambda features: model.predict(features.reshape(1, -1))[0]
                log.info("ASL model loaded from %s", path)
                return
            except Exception as exc:
                log.warning("ASL model failed to load (%s): %s — using rule-based fallback", path, exc)

        log.warning("ASL model not found at %s — using rule-based fallback", path)
        try:
            from sign_language_module import recognize_letter

            def _rule_predict(features: np.ndarray):   # noqa — features unused by rule recognizer
                return None   # caller must use hand_landmarks directly
            # Store the raw recognizer instead
            self._rule_recognize = recognize_letter
        except ImportError:
            log.error("sign_language_module not available; ASL disabled")

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.config.show_asl:
            return ctx
        if not ctx.hand_results or not ctx.hand_results.multi_hand_landmarks:
            return ctx

        ctx.asl_letters = []
        for hand_idx, hand_lm in enumerate(ctx.hand_results.multi_hand_landmarks):
            if hand_idx >= 2:
                break

            letter: Optional[str] = None
            if self._predict is not None:
                features = _normalize_landmarks(hand_lm)
                letter = self._predict(features)
            elif hasattr(self, "_rule_recognize"):
                letter = self._rule_recognize(hand_lm)

            self._histories[hand_idx].append(letter)
            valid = [l for l in self._histories[hand_idx] if l is not None]
            if valid:
                top, count = Counter(valid).most_common(1)[0]
                if count >= _MIN_VOTES:
                    ctx.asl_letters.append((hand_idx, top))

        return ctx

    def close(self) -> None:
        pass
