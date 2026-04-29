# stages/filter_stage.py
from __future__ import annotations
import logging
import math
from typing import Optional

from core.context import FrameContext
from filters import FILTERS

log = logging.getLogger("hand_tracker.filter")

# MediaPipe hand landmark indices used for gesture → filter cycling
_THUMB_TIP  = 4;  _INDEX_TIP  = 8;  _INDEX_PIP  = 6
_MIDDLE_TIP = 12; _MIDDLE_PIP = 10; _RING_TIP   = 16
_RING_PIP   = 14; _PINKY_TIP  = 20; _PINKY_PIP  = 18
_PINCH_PX   = 30


class FilterStage:
    def __init__(self, initial_name: Optional[str] = None) -> None:
        self._index = 0
        if initial_name:
            for i, f in enumerate(FILTERS):
                if f["name"].lower() == initial_name.lower():
                    self._index = i
                    break
            else:
                log.warning("Filter '%s' not found; using '%s'", initial_name, FILTERS[0]["name"])
        self._hand_was_open = False
        self._was_pinching  = False

    def process(self, ctx: FrameContext) -> FrameContext:
        h, w = ctx.frame.shape[:2]
        pinching_now = False
        open_now     = False

        if ctx.hand_results and ctx.hand_results.multi_hand_landmarks:
            for hand_lm in ctx.hand_results.multi_hand_landmarks:
                lm = hand_lm.landmark
                # pinch check
                tx = int(lm[_THUMB_TIP].x * w);  ty = int(lm[_THUMB_TIP].y * h)
                ix = int(lm[_INDEX_TIP].x * w);  iy = int(lm[_INDEX_TIP].y * h)
                if math.hypot(tx - ix, ty - iy) < _PINCH_PX:
                    pinching_now = True
                # open-hand check
                if (lm[_INDEX_TIP].y  < lm[_INDEX_PIP].y  and
                    lm[_MIDDLE_TIP].y < lm[_MIDDLE_PIP].y and
                    lm[_RING_TIP].y   < lm[_RING_PIP].y   and
                    lm[_PINKY_TIP].y  < lm[_PINKY_PIP].y):
                    open_now = True

        if open_now and not pinching_now:
            self._hand_was_open = True
        if self._hand_was_open and pinching_now and not self._was_pinching:
            self._index = (self._index + 1) % len(FILTERS)
            self._hand_was_open = False
            log.debug("Filter changed to '%s'", FILTERS[self._index]["name"])
        self._was_pinching = pinching_now

        active = FILTERS[self._index]
        ctx.frame          = active["apply"](ctx.frame, hand_results=ctx.hand_results)
        ctx.active_filter  = active
        ctx.coord_transform = active.get("coord_transform", None)
        return ctx

    def close(self) -> None:
        pass
