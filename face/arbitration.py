from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from .types import FaceAttributes, ModelVote, RecognitionResult, RecognitionStatus

log = logging.getLogger("hand_tracker.face.arbitration")

# ---------------------------------------------------------------------------
# Per-model cosine-similarity thresholds
# ---------------------------------------------------------------------------
_THRESHOLDS: dict[str, float] = {
    "model_a": 0.45,
    "model_b": 0.42,
    "model_c": 0.43,
}
_DEFAULT_THRESHOLD = 0.43


def _threshold(model: str) -> float:
    return _THRESHOLDS.get(model, _DEFAULT_THRESHOLD)


# ---------------------------------------------------------------------------
# Arbitrator
# ---------------------------------------------------------------------------

class Arbitrator:
    """Resolve votes from the ensemble into a single RecognitionResult."""

    def __init__(
        self,
        threshold_a: float = 0.45,
        threshold_b: float = 0.42,
        threshold_c: float = 0.43,
        ambiguous_margin: float = 0.07,
        base_weight_a: float = 0.40,
        base_weight_b: float = 0.30,
        base_weight_c: float = 0.30,
    ) -> None:
        self._thresholds: dict[str, float] = {
            "model_a": threshold_a,
            "model_b": threshold_b,
            "model_c": threshold_c,
        }
        self._ambiguous_margin = ambiguous_margin
        self._base_weights: dict[str, float] = {
            "model_a": base_weight_a,
            "model_b": base_weight_b,
            "model_c": base_weight_c,
        }

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _get_threshold(self, model: str) -> float:
        return self._thresholds.get(model, _DEFAULT_THRESHOLD)

    def _is_strong(self, vote: ModelVote) -> bool:
        return vote.score >= self._get_threshold(vote.model)

    def _compute_weights(
        self, votes: list[ModelVote], attributes: FaceAttributes
    ) -> dict[str, float]:
        w = dict(self._base_weights)

        # Fill in zero weights for models not represented in votes
        for v in votes:
            if v.model not in w:
                w[v.model] = _DEFAULT_THRESHOLD  # safe fallback

        w_a = w.get("model_a", 0.0)
        w_b = w.get("model_b", 0.0)
        w_c = w.get("model_c", 0.0)

        # Dynamic adjustments
        if attributes.quality < 0.6:
            # AdaFace handles low-quality images better
            w_b += 0.10
            w_a -= 0.10

        if attributes.glasses_change or attributes.heavy_makeup:
            # ElasticFace handles appearance distribution shifts better
            w_c += 0.10
            w_a -= 0.10

        # Re-normalise
        total = w_a + w_b + w_c
        if total <= 0.0:
            total = 1.0
        return {
            "model_a": w_a / total,
            "model_b": w_b / total,
            "model_c": w_c / total,
        }

    def _weighted_confidence(
        self, votes: list[ModelVote], weights: dict[str, float]
    ) -> float:
        total_w = 0.0
        total_ws = 0.0
        for v in votes:
            w = weights.get(v.model, 0.0)
            total_ws += w * v.score
            total_w += w
        if total_w <= 0.0:
            return sum(v.score for v in votes) / max(len(votes), 1)
        return total_ws / total_w

    # ---------------------------------------------------------------------- #
    # Main entry point                                                         #
    # ---------------------------------------------------------------------- #

    def arbitrate(
        self,
        votes: list[ModelVote],
        attributes: FaceAttributes,
        confusable_names: set[str] | None = None,
    ) -> RecognitionResult:
        # ------------------------------------------------------------------ #
        # Guard: no votes at all                                               #
        # ------------------------------------------------------------------ #
        if not votes:
            return RecognitionResult(
                status=RecognitionStatus.UNKNOWN,
                name="Unknown",
                confidence=0.0,
            )

        weights = self._compute_weights(votes, attributes)

        # ------------------------------------------------------------------ #
        # Single-model fallback (at most SOFT_CONFIRMED)                      #
        # ------------------------------------------------------------------ #
        if len(votes) == 1:
            v = votes[0]
            if self._is_strong(v):
                return RecognitionResult(
                    status=RecognitionStatus.SOFT_CONFIRMED,
                    name=v.match,
                    confidence=v.score,
                )
            # Not strong — treat as low-confidence unanimous (trivially)
            margin_ok = (v.score - v.ru_score) > self._ambiguous_margin
            if margin_ok:
                return RecognitionResult(
                    status=RecognitionStatus.LOW_CONFIDENCE,
                    name=v.match,
                    confidence=v.score,
                )
            return RecognitionResult(
                status=RecognitionStatus.AMBIGUOUS,
                name="?",
                confidence=v.score,
                candidates=[v.match],
            )

        # ------------------------------------------------------------------ #
        # Common aggregates                                                    #
        # ------------------------------------------------------------------ #
        vote_counts: Counter[str] = Counter(v.match for v in votes)
        unique_names = set(vote_counts.keys())

        # ------------------------------------------------------------------ #
        # Rule 1 & 3: All models agree (unanimous)                            #
        # ------------------------------------------------------------------ #
        if len(unique_names) == 1:
            name = votes[0].match
            all_strong = all(self._is_strong(v) for v in votes)

            if all_strong:
                # Rule 1: CONFIRMED
                confidence = self._weighted_confidence(votes, weights)
                return RecognitionResult(
                    status=RecognitionStatus.CONFIRMED,
                    name=name,
                    confidence=confidence,
                )

            # Rule 3: unanimous but weak.
            # Only report LOW_CONFIDENCE when scores are above the noise floor
            # (60% of threshold). Scores below the floor indicate the person is
            # simply not in the gallery — fall through to Rule 5 (UNKNOWN).
            above_noise_floor = all(
                v.score >= self._get_threshold(v.model) * 0.60
                for v in votes
            )
            if above_noise_floor:
                all_margins_ok = all(
                    (v.score - v.ru_score) > self._ambiguous_margin for v in votes
                )
                if all_margins_ok:
                    confidence = sum(v.score for v in votes) / len(votes)
                    return RecognitionResult(
                        status=RecognitionStatus.LOW_CONFIDENCE,
                        name=name,
                        confidence=confidence,
                    )
                # Unanimous but margins too thin → AMBIGUOUS
                return RecognitionResult(
                    status=RecognitionStatus.AMBIGUOUS,
                    name="?",
                    confidence=max(v.score for v in votes),
                    candidates=[name],
                )
            # Scores are noise-level — fall through to Rule 5

        # ------------------------------------------------------------------ #
        # Rule 2: Majority (2 out of 3) agrees                                #
        # ------------------------------------------------------------------ #
        top_name, top_count = vote_counts.most_common(1)[0]
        if top_count >= 2:
            majority_votes = [v for v in votes if v.match == top_name]
            minority_votes = [v for v in votes if v.match != top_name]

            majority_strong = all(self._is_strong(v) for v in majority_votes)

            # A minority vote is "strong for a different person" only when
            # it is above threshold and not simply "Unknown".
            minority_strong_different = any(
                self._is_strong(v) and v.match != top_name
                for v in minority_votes
            )

            if majority_strong and not minority_strong_different:
                confidence = (
                    sum(v.score for v in majority_votes) / len(majority_votes)
                )
                return RecognitionResult(
                    status=RecognitionStatus.SOFT_CONFIRMED,
                    name=top_name,
                    confidence=confidence,
                )

            # Majority strong but minority also strong for a different name
            # → genuine confusion, fall through to AMBIGUOUS

        # ------------------------------------------------------------------ #
        # Rule 5: All below threshold + no consensus → UNKNOWN               #
        # ------------------------------------------------------------------ #
        if all(not self._is_strong(v) for v in votes):
            return RecognitionResult(
                status=RecognitionStatus.UNKNOWN,
                name="Unknown",
                confidence=0.0,
            )

        # ------------------------------------------------------------------ #
        # Rule 4: Disagreement → AMBIGUOUS                                    #
        # ------------------------------------------------------------------ #
        candidates = list({
            v.match
            for v in votes
            if self._is_strong(v)
        })
        if not candidates:
            candidates = [vote_counts.most_common(1)[0][0]]

        return RecognitionResult(
            status=RecognitionStatus.AMBIGUOUS,
            name="?",
            confidence=max(v.score for v in votes),
            candidates=candidates,
        )
