# tests/test_stages.py
"""Tests for render-thread stages using synthetic FrameContext."""
import time
from pathlib import Path

import numpy as np
import pytest

from core.config import Config
from core.context import FrameContext
from filters import FILTERS


def _ctx(**kwargs) -> FrameContext:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    defaults = dict(
        raw_frame=blank, frame=blank.copy(),
        timestamp=time.monotonic(), frame_id=0,
        config=Config(), active_filter=FILTERS[0]
    )
    defaults.update(kwargs)
    return FrameContext(**defaults)


# ── FilterStage ──────────────────────────────────────────────────────────────

class TestFilterStage:
    def test_default_filter_is_normal(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage()
        ctx = stage.process(_ctx())
        assert ctx.active_filter["name"] == "Normal"

    def test_initial_filter_by_name(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="Inverted")
        ctx = stage.process(_ctx())
        assert ctx.active_filter["name"] == "Inverted"

    def test_unknown_filter_falls_back_to_normal(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="DoesNotExist")
        ctx = stage.process(_ctx())
        assert ctx.active_filter["name"] == "Normal"

    def test_coord_transform_none_for_normal(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage()
        ctx = stage.process(_ctx())
        assert ctx.coord_transform is None

    def test_upside_down_sets_coord_transform(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="Upside Down")
        ctx = stage.process(_ctx())
        assert ctx.coord_transform is not None
        # Verify it transforms y as expected: y → h-1-y
        tx, ty = ctx.coord_transform(10, 20, 640, 480)
        assert ty == 480 - 1 - 20

    def test_frame_is_mutated(self):
        from stages.filter_stage import FilterStage
        stage = FilterStage(initial_name="Inverted")
        white = np.ones((480, 640, 3), dtype=np.uint8) * 200
        ctx = _ctx(frame=white.copy())
        ctx = stage.process(ctx)
        # Inverted: 200 → 55
        assert ctx.frame[0, 0, 0] == 55


# ── FaceStage ────────────────────────────────────────────────────────────────

class TestFaceStage:
    def test_skipped_when_show_face_false(self):
        from stages.face_stage import FaceStage
        stage = FaceStage()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cfg = Config(show_face=False)
        ctx = _ctx(config=cfg, frame=blank.copy())
        original = ctx.frame.copy()
        ctx = stage.process(ctx)
        np.testing.assert_array_equal(ctx.frame, original)

    def test_returns_ctx(self):
        from stages.face_stage import FaceStage
        stage = FaceStage()
        ctx = _ctx()
        result = stage.process(ctx)
        assert result is ctx


# ── ASLStage ─────────────────────────────────────────────────────────────────

class TestASLStage:
    def test_skipped_when_show_asl_false(self):
        from stages.asl_stage import ASLStage
        stage = ASLStage()
        cfg = Config(show_asl=False)
        ctx = _ctx(config=cfg)
        ctx = stage.process(ctx)
        assert ctx.asl_letters == []

    def test_no_letters_without_hand_results(self):
        from stages.asl_stage import ASLStage
        stage = ASLStage()
        cfg = Config(show_asl=True)
        ctx = _ctx(config=cfg, hand_results=None)
        ctx = stage.process(ctx)
        assert ctx.asl_letters == []

    def test_fallback_loads_when_no_model(self, tmp_path):
        from stages.asl_stage import ASLStage
        # Point to a non-existent model path → should not raise
        stage = ASLStage(model_path=tmp_path / "nonexistent.pkl")
        assert stage is not None   # initialised without error


# ── RendererStage ─────────────────────────────────────────────────────────────

class TestRendererStage:
    def test_returns_ctx(self):
        from stages.renderer import RendererStage
        stage = RendererStage()
        ctx = _ctx()
        result = stage.process(ctx)
        assert result is ctx
        stage.close()

    def test_hud_text_drawn_on_frame(self):
        from stages.renderer import RendererStage
        stage = RendererStage()
        ctx = _ctx()
        before = ctx.frame.copy()
        ctx = stage.process(ctx)
        # Frame should be modified (HUD text drawn)
        assert not np.array_equal(ctx.frame, before)
        stage.close()
