# tests/test_pipeline.py
"""Tests for core.pipeline — threading, queues, sentinel propagation."""
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.config import Config
from core.context import FrameContext
from core.pipeline import Pipeline, Stage


def _make_ctx(frame_id: int = 0) -> FrameContext:
    blank = np.zeros((10, 10, 3), dtype=np.uint8)
    return FrameContext(
        raw_frame=blank, frame=blank.copy(),
        timestamp=time.monotonic(), frame_id=frame_id, config=Config()
    )


class _FakeCapture:
    """Yields a fixed number of frames then returns None."""
    def __init__(self, n: int = 3):
        self._n = n
        self._count = 0

    def read_frame(self):
        if self._count >= self._n:
            return None
        self._count += 1
        # Small sleep so queues (maxsize=2) don't overflow and drop frames.
        time.sleep(0.005)
        return _make_ctx(self._count)

    def close(self):
        pass


class _PassStage:
    def process(self, ctx):
        return ctx
    def close(self):
        pass


class _RecordingStage:
    def __init__(self):
        self.received = []
    def process(self, ctx):
        self.received.append(ctx.frame_id)
        return ctx
    def close(self):
        pass


class _FakeInference:
    def process(self, ctx):
        return ctx
    def close(self):
        pass


def test_stage_protocol():
    """Stage protocol is satisfied by any class with process() and close()."""
    stage = _PassStage()
    assert isinstance(stage, Stage)


def test_pipeline_processes_all_frames():
    """All frames from capture reach the render stages."""
    recorder = _RecordingStage()
    cap = _FakeCapture(n=5)
    inf = _FakeInference()

    with patch("core.pipeline.cv2") as mock_cv2:
        mock_cv2.waitKey.return_value = 0     # never press Q
        mock_cv2.imshow.return_value = None
        mock_cv2.destroyAllWindows.return_value = None

        pipeline = Pipeline(cap, inf, [recorder])
        # Run pipeline; it will stop when capture returns None (sentinel propagates)
        pipeline.run()

    assert recorder.received == [1, 2, 3, 4, 5], f"Got {recorder.received}"


def test_pipeline_shutdown_on_q():
    """Pipeline stops when cv2.waitKey returns ord('q')."""
    recorder = _RecordingStage()
    cap = _FakeCapture(n=100)   # more frames than we'll process
    inf = _FakeInference()

    call_count = [0]
    def fake_waitKey(delay):
        call_count[0] += 1
        return ord("q") if call_count[0] >= 2 else 0

    with patch("core.pipeline.cv2") as mock_cv2:
        mock_cv2.waitKey.side_effect = fake_waitKey
        mock_cv2.imshow.return_value = None
        mock_cv2.destroyAllWindows.return_value = None
        pipeline = Pipeline(cap, inf, [recorder])
        pipeline.run()

    # Should stop early, not process all 100 frames
    assert len(recorder.received) < 100
