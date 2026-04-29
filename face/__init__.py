"""Face recognition subsystem — dual-detector + 3-model ensemble."""
from face.types import (
    RecognitionStatus,
    FaceAttributes,
    DetectedFace,
    ModelVote,
    RecognitionResult,
)
from face.detector import DualDetector
from face.attributes import extract_attributes, compute_quality_score
from face.gallery import FaceGallery
from face.models import EnsembleRecognizer
from face.arbitration import Arbitrator
from face.temporal import TemporalSmoother

__all__ = [
    "RecognitionStatus",
    "FaceAttributes",
    "DetectedFace",
    "ModelVote",
    "RecognitionResult",
    "DualDetector",
    "extract_attributes",
    "compute_quality_score",
    "FaceGallery",
    "EnsembleRecognizer",
    "Arbitrator",
    "TemporalSmoother",
]
