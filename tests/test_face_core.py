"""Pure-logic tests for face recognition modules.

No GPU, no camera, no insightface/ultralytics/onnxruntime required.
All tests should complete in well under 5 seconds.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies before any face.* imports
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _pkg in [
    "insightface",
    "insightface.app",
    "insightface.utils",
    "insightface.utils.face_align",
    "onnxruntime",
    "ultralytics",
]:
    if _pkg not in sys.modules:
        _make_stub(_pkg)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from face.gallery import FaceGallery
from face.temporal import TemporalSmoother
from face.arbitration import Arbitrator
from face.types import (
    FaceAttributes,
    ModelVote,
    RecognitionResult,
    RecognitionStatus,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _unit_emb(seed: int = 0, dim: int = 512) -> np.ndarray:
    """Return a deterministic L2-normalised float32 embedding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _default_attrs(**kwargs) -> FaceAttributes:
    return FaceAttributes(quality=0.8, **kwargs)


# ===========================================================================
# FaceGallery tests
# ===========================================================================


class TestFaceGallery:

    def test_gallery_empty_on_init(self, tmp_path: Path) -> None:
        gallery = FaceGallery(gallery_dir=tmp_path)
        assert gallery.is_empty()
        assert gallery.people == []

    def test_gallery_add_and_search(self, tmp_path: Path) -> None:
        gallery = FaceGallery(gallery_dir=tmp_path)
        emb = _unit_emb(seed=1)
        gallery.add_person("Alice", {"model_a": emb}, {})

        name, score, _ru, _ru_score = gallery.search(emb, "model_a")
        assert name == "Alice"
        assert score > 0.9

    def test_gallery_search_l2_normalized(self, tmp_path: Path) -> None:
        """gallery.search() must normalise the query itself, so passing a
        non-unit vector should still return a high cosine similarity when the
        direction matches the stored embedding."""
        gallery = FaceGallery(gallery_dir=tmp_path)
        emb = _unit_emb(seed=2)
        gallery.add_person("Bob", {"model_a": emb}, {})

        # Scale the query by an arbitrary factor — direction is identical
        query_unnorm = emb * 7.3
        name, score, _ru, _ru_score = gallery.search(query_unnorm, "model_a")
        assert name == "Bob"
        assert score > 0.9

    def test_gallery_search_unknown_when_empty(self, tmp_path: Path) -> None:
        gallery = FaceGallery(gallery_dir=tmp_path)
        result = gallery.search(_unit_emb(), "model_a")
        assert result == ("Unknown", 0.0, "Unknown", 0.0)

    def test_gallery_sanitize_name(self, tmp_path: Path) -> None:
        gallery = FaceGallery(gallery_dir=tmp_path)
        # "Alice Johnson" → "alice_johnson"
        assert gallery._sanitize("Alice Johnson") == "alice_johnson"
        # Special characters stripped
        assert gallery._sanitize("Jean-Pierre!") == "jeanpierre"
        assert gallery._sanitize("O'Brien") == "obrien"

    def test_gallery_persist_and_reload(self, tmp_path: Path) -> None:
        gallery = FaceGallery(gallery_dir=tmp_path)
        emb = _unit_emb(seed=3)
        gallery.add_person("Carol", {"model_a": emb}, {})

        # save() is called by add_person; create a fresh gallery from same dir
        gallery2 = FaceGallery(gallery_dir=tmp_path)
        name, score, _ru, _ru_score = gallery2.search(emb, "model_a")
        assert name == "Carol"
        assert score > 0.9

    def test_gallery_compute_similarity_matrix(self, tmp_path: Path) -> None:
        """Two people with identical embeddings should appear as confusable
        (similarity ≈ 1.0 > 0.35 threshold)."""
        gallery = FaceGallery(gallery_dir=tmp_path)
        emb = _unit_emb(seed=4)
        gallery.add_person("Dave", {"model_a": emb}, {})
        gallery.add_person("Eve", {"model_a": emb.copy()}, {})

        pairs = gallery.compute_similarity_matrix(threshold=0.35)
        assert len(pairs) == 1
        name_a, name_b, scores = pairs[0]
        assert set([name_a, name_b]) == {"Dave", "Eve"}
        assert scores["model_a"] > 0.99

    def test_gallery_multiple_models(self, tmp_path: Path) -> None:
        """A person enrolled with both model_a and model_b can be found via
        either model independently."""
        gallery = FaceGallery(gallery_dir=tmp_path)
        emb_a = _unit_emb(seed=5)
        emb_b = _unit_emb(seed=6)
        gallery.add_person("Frank", {"model_a": emb_a, "model_b": emb_b}, {})

        name_a, score_a, _, _ = gallery.search(emb_a, "model_a")
        name_b, score_b, _, _ = gallery.search(emb_b, "model_b")
        assert name_a == "Frank" and score_a > 0.9
        assert name_b == "Frank" and score_b > 0.9

    def test_gallery_remove_person(self, tmp_path: Path) -> None:
        gallery = FaceGallery(gallery_dir=tmp_path)
        emb = _unit_emb(seed=7)
        gallery.add_person("Grace", {"model_a": emb}, {})
        assert "Grace" in gallery.people

        gallery.remove_person("Grace")
        assert "Grace" not in gallery.people
        # After removal, searching returns the empty-gallery sentinel
        result = gallery.search(emb, "model_a")
        assert result == ("Unknown", 0.0, "Unknown", 0.0)


# ===========================================================================
# TemporalSmoother tests
# ===========================================================================


class TestTemporalSmoother:

    def test_temporal_new_track_created(self) -> None:
        smoother = TemporalSmoother(init_frames=3)
        bbox = (10, 10, 50, 50)
        output = smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])
        # One track should exist
        assert len(output) == 1

    def test_temporal_track_initialized_after_n_frames(self) -> None:
        smoother = TemporalSmoother(init_frames=3, min_consensus=0.6)
        bbox = (10, 10, 50, 50)
        # Feed 3 identical decisions
        for _ in range(3):
            output = smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])
        # Track is now initialised; with 3/3 consensus the name should appear
        names = {name for name, _ in output.values()}
        assert "Alice" in names

    def test_temporal_track_not_shown_before_init(self) -> None:
        smoother = TemporalSmoother(init_frames=3, min_consensus=0.6)
        bbox = (10, 10, 50, 50)
        # Only 2 frames — still below init_frames
        for _ in range(2):
            output = smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])
        # Track not yet initialised; _stable_decision returns "?"
        names = {name for name, _ in output.values()}
        assert "Alice" not in names
        assert "?" in names

    def test_temporal_track_timeout(self) -> None:
        # Use a very short timeout
        smoother = TemporalSmoother(init_frames=1, timeout_frames=2)
        bbox = (10, 10, 50, 50)
        smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])

        # Let the track age past timeout_frames without any detection
        for _ in range(3):
            output = smoother.update([])

        assert len(output) == 0

    def test_temporal_iou_matching(self) -> None:
        """A detection at a nearby bbox should match the existing track
        instead of spawning a new one."""
        smoother = TemporalSmoother(init_frames=3, iou_threshold=0.4)
        bbox1 = (10, 10, 60, 60)
        smoother.update([(bbox1, "Alice", RecognitionStatus.CONFIRMED)])
        assert len(smoother._tracks) == 1

        # Slightly shifted bbox — IoU should be > 0.4
        bbox2 = (12, 12, 62, 62)
        smoother.update([(bbox2, "Alice", RecognitionStatus.CONFIRMED)])
        # Still only one track (matched, not created)
        assert len(smoother._tracks) == 1

    def test_temporal_reset(self) -> None:
        smoother = TemporalSmoother(init_frames=3)
        bbox = (10, 10, 50, 50)
        smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])
        assert len(smoother._tracks) > 0

        smoother.reset()
        assert len(smoother._tracks) == 0

    def test_temporal_consensus_below_threshold(self) -> None:
        """50% Alice / 50% Bob in a window means neither name reaches the
        min_consensus threshold of 0.60, so the result must be '?'."""
        smoother = TemporalSmoother(
            init_frames=2,
            window_size=4,
            min_consensus=0.60,
        )
        bbox = (10, 10, 50, 50)

        # 2 Alice decisions → track initialises (init_frames=2)
        smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])
        smoother.update([(bbox, "Alice", RecognitionStatus.CONFIRMED)])

        # 2 Bob decisions → now 50/50 split in the window
        smoother.update([(bbox, "Bob", RecognitionStatus.CONFIRMED)])
        output = smoother.update([(bbox, "Bob", RecognitionStatus.CONFIRMED)])

        names = {name for name, _ in output.values()}
        # Consensus is exactly 0.50 which is below 0.60 → should return "?"
        assert "?" in names or names == {"?"}


# ===========================================================================
# Arbitrator tests
# ===========================================================================


class TestArbitrator:

    def _vote(
        self,
        model: str,
        match: str,
        score: float,
        ru: str = "Unknown",
        ru_score: float = 0.0,
    ) -> ModelVote:
        return ModelVote(model=model, match=match, score=score,
                         runner_up=ru, ru_score=ru_score)

    def test_arbitration_unanimous_strong_confirmed(self) -> None:
        arb = Arbitrator()
        votes = [
            self._vote("model_a", "Alice", 0.80),
            self._vote("model_b", "Alice", 0.75),
            self._vote("model_c", "Alice", 0.77),
        ]
        result = arb.arbitrate(votes, _default_attrs())
        assert result.status == RecognitionStatus.CONFIRMED
        assert result.name == "Alice"

    def test_arbitration_all_below_threshold_unknown(self) -> None:
        arb = Arbitrator()
        # All scores well below their per-model thresholds
        votes = [
            self._vote("model_a", "Alice", 0.20),
            self._vote("model_b", "Alice", 0.18),
            self._vote("model_c", "Alice", 0.19),
        ]
        result = arb.arbitrate(votes, _default_attrs())
        assert result.status == RecognitionStatus.UNKNOWN
        assert result.name == "Unknown"

    def test_arbitration_majority_soft_confirmed(self) -> None:
        """2 models agree on 'Alice' (both above threshold); 1 model votes
        for 'Bob' but below threshold → SOFT_CONFIRMED for Alice."""
        arb = Arbitrator()
        votes = [
            self._vote("model_a", "Alice", 0.72),
            self._vote("model_b", "Alice", 0.65),
            # model_c below threshold, different name
            self._vote("model_c", "Bob", 0.25),
        ]
        result = arb.arbitrate(votes, _default_attrs())
        assert result.status == RecognitionStatus.SOFT_CONFIRMED
        assert result.name == "Alice"

    def test_arbitration_unanimous_weak_low_confidence(self) -> None:
        """All 3 agree on 'Alice' but scores are below their thresholds;
        margins are clearly above ambiguous_margin → LOW_CONFIDENCE."""
        arb = Arbitrator(ambiguous_margin=0.07)
        # Scores below thresholds (0.45, 0.42, 0.43) but margins wide
        votes = [
            self._vote("model_a", "Alice", 0.35, ru_score=0.10),
            self._vote("model_b", "Alice", 0.32, ru_score=0.10),
            self._vote("model_c", "Alice", 0.33, ru_score=0.10),
        ]
        result = arb.arbitrate(votes, _default_attrs())
        assert result.status == RecognitionStatus.LOW_CONFIDENCE
        assert result.name == "Alice"

    def test_arbitration_disagreement_ambiguous(self) -> None:
        """Each model votes for a different name → AMBIGUOUS."""
        arb = Arbitrator()
        votes = [
            self._vote("model_a", "Alice", 0.70),
            self._vote("model_b", "Bob",   0.68),
            self._vote("model_c", "Carol", 0.65),
        ]
        result = arb.arbitrate(votes, _default_attrs())
        assert result.status == RecognitionStatus.AMBIGUOUS
        assert result.name == "?"

    def test_arbitration_empty_votes_unknown(self) -> None:
        arb = Arbitrator()
        result = arb.arbitrate([], _default_attrs())
        assert result.status == RecognitionStatus.UNKNOWN
        assert result.name == "Unknown"
        assert result.confidence == 0.0

    def test_arbitration_single_model_vote(self) -> None:
        """With only one model available and score above threshold, the
        result must be at most SOFT_CONFIRMED (not CONFIRMED)."""
        arb = Arbitrator()
        votes = [self._vote("model_a", "Alice", 0.80)]
        result = arb.arbitrate(votes, _default_attrs())
        # Single model can never achieve CONFIRMED
        assert result.status != RecognitionStatus.CONFIRMED
        assert result.status == RecognitionStatus.SOFT_CONFIRMED
        assert result.name == "Alice"

    def test_arbitration_weight_shift_low_quality(self) -> None:
        """When face quality < 0.6, model_b weight should increase and
        model_a weight should decrease relative to the base weights."""
        arb = Arbitrator(
            base_weight_a=0.40,
            base_weight_b=0.30,
            base_weight_c=0.30,
        )
        low_quality_attrs = FaceAttributes(quality=0.4)
        high_quality_attrs = FaceAttributes(quality=0.9)

        votes = [
            self._vote("model_a", "Alice", 0.70),
            self._vote("model_b", "Alice", 0.70),
            self._vote("model_c", "Alice", 0.70),
        ]

        low_w = arb._compute_weights(votes, low_quality_attrs)
        high_w = arb._compute_weights(votes, high_quality_attrs)

        # Low-quality frame should boost model_b and reduce model_a
        assert low_w["model_b"] > high_w["model_b"]
        assert low_w["model_a"] < high_w["model_a"]
