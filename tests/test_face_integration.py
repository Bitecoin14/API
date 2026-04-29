"""Integration-level tests for FaceStage and the enrollment helpers.

All heavy optional dependencies (insightface, ultralytics, onnxruntime, cv2's
camera subsystem) are either stubbed at import time or patched per-test so
that no GPU, camera, or large model file is needed.  Every test should finish
in well under 5 seconds.
"""
from __future__ import annotations

import sys
import types
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out every heavy optional dependency before the project modules load
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return sys.modules[name]


for _pkg in [
    "insightface",
    "insightface.app",
    "insightface.utils",
    "insightface.utils.face_align",
    "onnxruntime",
    "ultralytics",
]:
    _stub(_pkg)

# ---------------------------------------------------------------------------
# Imports under test (must come after stubs)
# ---------------------------------------------------------------------------

from face.gallery import FaceGallery
from face.types import RecognitionStatus
from face.enrollment import _augment, _mean_embedding, run_audit


# ===========================================================================
# Helpers
# ===========================================================================

def _blank_crop(h: int = 112, w: int = 112) -> np.ndarray:
    """Return a uint8 BGR array shaped (h, w, 3) filled with mid-grey."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _unit_emb(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _make_frame_ctx(show_face: bool = True) -> "FrameContext":  # type: ignore[name-defined]
    """Build a minimal FrameContext-like object without importing Config."""
    from core.context import FrameContext
    from core.config import Config

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cfg = Config(show_face=show_face)
    return FrameContext(
        raw_frame=frame,
        frame=frame.copy(),
        timestamp=0.0,
        frame_id=0,
        config=cfg,
    )


# ===========================================================================
# FaceStage tests
# ===========================================================================


class TestFaceStage:

    def _build_stage(self, tmp_path: Path) -> "FaceStage":  # type: ignore[name-defined]
        """Construct a FaceStage whose expensive sub-components are mocked."""
        from stages.face_stage import FaceStage

        stage = FaceStage.__new__(FaceStage)
        # Manually wire up the attributes FaceStage.process() expects
        stage._use_ensemble = True
        stage._gallery = FaceGallery(gallery_dir=tmp_path)
        stage._detector = MagicMock()
        stage._recognizer = MagicMock()
        stage._arbitrator = MagicMock()
        stage._smoother = MagicMock()
        return stage

    def test_face_stage_no_show_face(self, tmp_path: Path) -> None:
        """When config.show_face is False, process() returns immediately
        without populating ctx.face_results."""
        from stages.face_stage import FaceStage

        stage = self._build_stage(tmp_path)
        ctx = _make_frame_ctx(show_face=False)
        result = stage.process(ctx)

        # The stage must not have touched face_results
        assert result.face_results == []
        # Detector must never have been called
        stage._detector.detect.assert_not_called()

    def test_face_stage_empty_gallery_unknown(self, tmp_path: Path) -> None:
        """With an empty gallery every detected face should produce UNKNOWN."""
        stage = self._build_stage(tmp_path)

        # Simulate a single detected face with a valid crop
        import numpy as np
        from face.types import DetectedFace, FaceAttributes

        fake_face = DetectedFace(
            bbox=(10, 10, 80, 80),
            det_score=0.9,
            yaw=0.0,
            blur_var=200.0,
            quality_score=0.8,
            attributes=FaceAttributes(quality=0.8),
            crop=_blank_crop(),
        )
        stage._detector.detect.return_value = [fake_face]

        # Smoother returns nothing stabilised yet (or stable "Unknown")
        stage._smoother.update.return_value = {}
        stage._smoother.get_track_id_for_bbox.return_value = None

        ctx = _make_frame_ctx(show_face=True)
        result = stage.process(ctx)

        # The gallery is empty so every face entry should show "?" or "Unknown"
        for _bbox, name, status in result.face_results:
            assert name in ("?", "Unknown")
            assert status == RecognitionStatus.UNKNOWN

    def test_face_stage_close_no_error(self, tmp_path: Path) -> None:
        """close() must not raise even if the internal components were never
        fully initialised."""
        from stages.face_stage import FaceStage

        stage = FaceStage.__new__(FaceStage)
        stage._detector = None
        stage._recognizer = None
        # Call close() — should be a no-op and not raise
        stage.close()


# ===========================================================================
# Enrollment helper tests
# ===========================================================================


class TestEnrollmentHelpers:

    def test_enroll_augment_produces_5_variants(self) -> None:
        """_augment() must return exactly 5 variants all shaped (112, 112, 3)."""
        crop = _blank_crop()
        variants = _augment(crop)

        assert len(variants) == 5
        for v in variants:
            assert v.shape == (112, 112, 3), f"unexpected shape {v.shape}"
            assert v.dtype == np.uint8

    def test_enroll_mean_embedding_normalized(self) -> None:
        """_mean_embedding() must return L2-normalised embeddings (norm ≈ 1)."""
        crop = _blank_crop()
        variants = [crop] * 3  # 3 identical crops — result should still be unit

        # Build a mock EnsembleRecognizer that returns a fixed embedding per call
        fixed_emb = _unit_emb(seed=42)
        mock_recognizer = MagicMock()
        mock_recognizer.extract_embeddings.return_value = {"model_a": fixed_emb}

        result = _mean_embedding(variants, mock_recognizer)

        assert "model_a" in result
        emb = result["model_a"]
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"embedding not unit-normalised, norm={norm}"

    def test_run_audit_empty_gallery(self, tmp_path: Path, capsys) -> None:
        """run_audit() on an empty gallery should print 'Gallery is empty.'
        and return without error."""
        gallery = FaceGallery(gallery_dir=tmp_path)
        run_audit(gallery)

        captured = capsys.readouterr()
        assert "Gallery is empty" in captured.out
