# core/pipeline.py
from __future__ import annotations
import logging
import queue
import threading
from typing import Protocol, runtime_checkable

import cv2

from core.context import FrameContext

log = logging.getLogger("hand_tracker.pipeline")

_WINDOW_TITLE = "Hand Tracker  —  Q = quit  |  F = fullscreen"


@runtime_checkable
class Stage(Protocol):
    def process(self, ctx: FrameContext) -> FrameContext: ...
    def close(self) -> None: ...


@runtime_checkable
class CaptureProtocol(Protocol):
    def read_frame(self) -> "FrameContext | None": ...
    def close(self) -> None: ...


@runtime_checkable
class InferenceProtocol(Protocol):
    def process(self, ctx: "FrameContext") -> "FrameContext": ...
    def close(self) -> None: ...


class Pipeline:
    """Manages the three-thread capture → inference → render pipeline."""

    def __init__(
        self,
        capture_stage: "CaptureProtocol",
        inference_stage: "InferenceProtocol",
        render_stages: "list[Stage]",
    ) -> None:
        self._capture   = capture_stage
        self._inference = inference_stage
        self._render    = render_stages
        self._raw_q: queue.Queue     = queue.Queue(maxsize=2)
        self._result_q: queue.Queue  = queue.Queue(maxsize=2)
        self._stop = threading.Event()

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Pipeline starting")
        cap_t = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        inf_t = threading.Thread(target=self._inference_loop, name="inference", daemon=True)
        cap_t.start()
        inf_t.start()
        try:
            self._render_loop()
        finally:
            self._stop.set()
            cap_t.join(timeout=2.0)
            inf_t.join(timeout=2.0)
            self._shutdown_all()
        log.info("Pipeline stopped")

    # ── Internal threads ─────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            ctx = self._capture.read_frame()
            if ctx is None:
                break
            try:
                self._raw_q.put_nowait(ctx)
            except queue.Full:
                log.debug("capture: raw_q full — frame dropped")
        self._raw_q.put(None)   # shutdown sentinel
        log.debug("capture thread exiting")

    def _inference_loop(self) -> None:
        while True:
            ctx = self._raw_q.get()
            if ctx is None:
                break
            ctx = self._inference.process(ctx)
            try:
                self._result_q.put_nowait(ctx)
            except queue.Full:
                log.debug("inference: result_q full — frame dropped")
        self._result_q.put(None)    # shutdown sentinel
        log.debug("inference thread exiting")

    def _render_loop(self) -> None:
        cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)
        _fullscreen = False
        while True:
            ctx = self._result_q.get()
            if ctx is None:
                break
            for stage in self._render:
                ctx = stage.process(ctx)
            cv2.imshow(_WINDOW_TITLE, ctx.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._stop.set()
                try:
                    while True:
                        self._result_q.get_nowait()
                except queue.Empty:
                    pass
                break
            elif key == ord("f"):
                _fullscreen = not _fullscreen
                cv2.setWindowProperty(
                    _WINDOW_TITLE,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if _fullscreen else cv2.WINDOW_NORMAL,
                )

    def _shutdown_all(self) -> None:
        self._capture.close()
        self._inference.close()
        for stage in self._render:
            stage.close()
        cv2.destroyAllWindows()
