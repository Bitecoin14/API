"""Shared dataclasses and enums used across all face recognition modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class RecognitionStatus(Enum):
    CONFIRMED = "CONFIRMED"            # All models agree, all scores strong
    SOFT_CONFIRMED = "SOFT_CONFIRMED"  # 2/3 models agree, both scores strong
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # All agree but margins thin
    AMBIGUOUS = "AMBIGUOUS"            # Models disagree
    UNKNOWN = "UNKNOWN"                # All scores below threshold


@dataclass
class FaceAttributes:
    quality: float = 0.0
    glasses_detected: bool = False
    glasses_change: bool = False        # True if differs from enrollment
    heavy_makeup: bool = False
    yaw_degrees: float = 0.0


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]    # x1, y1, x2, y2 (pixel coords)
    det_score: float = 0.0
    yaw: float = 0.0
    blur_var: float = 0.0
    quality_score: float = 0.0
    attributes: FaceAttributes = field(default_factory=FaceAttributes)
    landmarks: Optional[np.ndarray] = None    # shape (5, 2) float32
    crop: Optional[np.ndarray] = None         # aligned 112×112 BGR crop


@dataclass
class ModelVote:
    model: str
    match: str
    score: float
    runner_up: str
    ru_score: float


@dataclass
class RecognitionResult:
    status: RecognitionStatus
    name: str
    confidence: float
    candidates: list[str] = field(default_factory=list)
    track_id: Optional[int] = None
