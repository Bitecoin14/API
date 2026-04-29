"""Face recognition stage — runs dual-detector + ensemble recognizer on each frame."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.context import FrameContext
from core.config import (
    FACE_MIN_SIZE, FACE_MIN_DET_SCORE, FACE_MAX_YAW, FACE_MIN_BLUR,
    MODEL_A_THRESHOLD, MODEL_B_THRESHOLD, MODEL_C_THRESHOLD,
    AMBIGUOUS_MARGIN, BASE_WEIGHT_A, BASE_WEIGHT_B, BASE_WEIGHT_C,
    TRACK_WINDOW_SIZE, TRACK_MIN_CONSENSUS, TRACK_INIT_FRAMES,
    TRACK_TIMEOUT_FRAMES, CPU_FALLBACK_TWO_MODELS,
    FACE_DETECTOR_INPUT_SIZE, YOLO_CROSS_CHECK_SIZE,
)

log = logging.getLogger("hand_tracker.face_stage")

# Lazy imports — only pull in face modules when they're available
_FACE_AVAILABLE = False
try:
    import insightface  # noqa: F401 — presence check; raises ImportError if absent
    from face.detector import DualDetector
    from face.attributes import extract_attributes
    from face.gallery import FaceGallery
    from face.models import EnsembleRecognizer
    from face.arbitration import Arbitrator
    from face.temporal import TemporalSmoother
    from face.types import RecognitionStatus, RecognitionResult
    _FACE_AVAILABLE = True
except ImportError as _e:
    log.warning("Advanced face recognition unavailable (%s). "
                "Falling back to legacy LBPH recognizer.", _e)


class FaceStage:
    """Pipeline stage that overlays face recognition results onto the frame."""

    def __init__(
        self,
        gallery_dir: str | Path = "gallery",
        models_dir: str | Path = "models",
        known_faces_dir: Optional[Path] = None,
    ) -> None:
        self._use_ensemble = _FACE_AVAILABLE
        self._detector: Optional[DualDetector] = None
        self._recognizer: Optional[EnsembleRecognizer] = None
        self._gallery: Optional[FaceGallery] = None
        self._arbitrator: Optional[Arbitrator] = None
        self._smoother: Optional[TemporalSmoother] = None

        if self._use_ensemble:
            self._init_ensemble(gallery_dir, models_dir)
        else:
            self._init_legacy(known_faces_dir)

    # ──────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────

    def _init_ensemble(self, gallery_dir: str | Path, models_dir: str | Path) -> None:
        try:
            self._detector = DualDetector(
                det_size=FACE_DETECTOR_INPUT_SIZE,
                cross_check_size=YOLO_CROSS_CHECK_SIZE,
                min_face_px=FACE_MIN_SIZE,
                min_det_score=FACE_MIN_DET_SCORE,
                max_yaw=FACE_MAX_YAW,
                min_blur=FACE_MIN_BLUR,
            )
            self._recognizer = EnsembleRecognizer(
                models_dir=models_dir,
                cpu_fallback_two_models=CPU_FALLBACK_TWO_MODELS,
            )
            self._gallery = FaceGallery(gallery_dir=gallery_dir)
            self._arbitrator = Arbitrator(
                threshold_a=MODEL_A_THRESHOLD,
                threshold_b=MODEL_B_THRESHOLD,
                threshold_c=MODEL_C_THRESHOLD,
                ambiguous_margin=AMBIGUOUS_MARGIN,
                base_weight_a=BASE_WEIGHT_A,
                base_weight_b=BASE_WEIGHT_B,
                base_weight_c=BASE_WEIGHT_C,
            )
            self._smoother = TemporalSmoother(
                window_size=TRACK_WINDOW_SIZE,
                min_consensus=TRACK_MIN_CONSENSUS,
                init_frames=TRACK_INIT_FRAMES,
                timeout_frames=TRACK_TIMEOUT_FRAMES,
            )
            log.info("Ensemble face recognition initialised (gallery: %s)", gallery_dir)
        except Exception as exc:
            log.error("Failed to initialise ensemble face recognition: %s", exc)
            self._use_ensemble = False
            self._init_legacy(None)

    def _init_legacy(self, known_faces_dir: Optional[Path]) -> None:
        self._legacy_recognizer = None
        self._legacy_label_map: dict = {}
        try:
            from face_recognition_module import load_known_faces
            self._legacy_recognizer, self._legacy_label_map = load_known_faces(
                *([known_faces_dir] if known_faces_dir else [])
            )
        except Exception as exc:
            log.warning("Legacy face recognizer also failed: %s", exc)

    # ──────────────────────────────────────────────
    # Pipeline stage entry point
    # ──────────────────────────────────────────────

    def process(self, ctx: FrameContext) -> FrameContext:
        if not ctx.config.show_face:
            return ctx

        if self._use_ensemble:
            ctx = self._process_ensemble(ctx)
        else:
            ctx = self._process_legacy(ctx)

        return ctx

    def _process_ensemble(self, ctx: FrameContext) -> FrameContext:
        assert self._detector is not None
        assert self._gallery is not None

        try:
            faces = self._detector.detect(ctx.frame)
        except Exception as exc:
            log.debug("Detection error: %s", exc)
            return ctx

        if not faces:
            return ctx

        # Get confusable names from gallery metadata for stricter arbitration
        confusable_names = self._get_confusable_names()

        detection_decisions: list[tuple[tuple, str, "RecognitionStatus"]] = []
        raw_results: dict[int, tuple[str, "RecognitionStatus"]] = {}

        for i, face in enumerate(faces):
            result = self._recognize_face(face, ctx.frame, confusable_names)
            detection_decisions.append((face.bbox, result.name, result.status))
            raw_results[i] = (result.name, result.status)

        # Update temporal smoother and get stable decisions
        stable = self._smoother.update(detection_decisions) if self._smoother else {}

        # Store results in context for renderer
        face_results = []
        for i, face in enumerate(faces):
            track_id = self._smoother.get_track_id_for_bbox(face.bbox) if self._smoother else None
            if track_id is not None and track_id in stable:
                stable_name, stable_status = stable[track_id]
                if stable_name != "?":
                    face_results.append((face.bbox, stable_name, stable_status))
                else:
                    # Smoother not yet stable — show raw per-frame result
                    raw_name, raw_status = raw_results[i]
                    face_results.append((face.bbox, raw_name, raw_status))
            else:
                # No track yet — show raw per-frame result immediately
                raw_name, raw_status = raw_results[i]
                face_results.append((face.bbox, raw_name, raw_status))

        ctx.face_results = face_results
        return ctx

    def _recognize_face(self, face, frame: np.ndarray,
                        confusable_names: set[str]) -> "RecognitionResult":
        """Run recognition pipeline for a single detected face."""
        from face.types import RecognitionResult, RecognitionStatus, ModelVote

        if self._gallery is None or self._gallery.is_empty():
            return RecognitionResult(RecognitionStatus.UNKNOWN, "Unknown", 0.0)

        if face.crop is None:
            return RecognitionResult(RecognitionStatus.UNKNOWN, "Unknown", 0.0)

        # Extract attributes (glasses, makeup) for dynamic weight adjustment
        attrs = extract_attributes(face, frame, insight_face_obj=None)

        # Extract embeddings from all available models
        embeddings = self._recognizer.extract_embeddings(face.crop) if self._recognizer else {}

        if not embeddings:
            return RecognitionResult(RecognitionStatus.UNKNOWN, "Unknown", 0.0)

        # Query each model against gallery
        votes: list[ModelVote] = []
        for model_name, embedding in embeddings.items():
            best_name, best_score, ru_name, ru_score = self._gallery.search(
                embedding, model_name
            )
            votes.append(ModelVote(
                model=model_name,
                match=best_name,
                score=best_score,
                runner_up=ru_name,
                ru_score=ru_score,
            ))

        # Arbitrate
        result = self._arbitrator.arbitrate(votes, attrs, confusable_names)
        return result

    def _get_confusable_names(self) -> set[str]:
        """Return set of names flagged as confusable in gallery metadata."""
        if self._gallery is None:
            return set()
        names = set()
        for person_key in self._gallery.people:
            meta = self._gallery.get_metadata(person_key)
            if "confusable_with" in meta:
                names.add(meta.get("display_name", person_key))
        return names

    def _process_legacy(self, ctx: FrameContext) -> FrameContext:
        """Fallback: OpenCV LBPH face recognition."""
        if self._legacy_recognizer is None:
            return ctx
        try:
            from face_recognition_module import recognize_and_draw
            ctx.frame = recognize_and_draw(
                ctx.frame, self._legacy_recognizer, self._legacy_label_map
            )
        except Exception as exc:
            log.debug("Legacy face recognition error: %s", exc)
        return ctx

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        if self._recognizer is not None:
            self._recognizer.close()
