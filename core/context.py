# core/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from core.config import Config


@dataclass
class FrameContext:
    raw_frame:  np.ndarray      # original BGR from VideoCapture
    frame:      np.ndarray      # working copy — stages mutate this
    timestamp:  float           # time.monotonic() at capture
    frame_id:   int             # monotonically increasing, 0-based
    config:     Config

    # Set by InferenceStage
    hand_results:  Any = None   # mediapipe.solutions.hands result object
    pose_results:  Any = None   # mediapipe.solutions.pose result object
    body_gestures: list[str] = field(default_factory=list)

    # Set by FilterStage
    active_filter:   dict = field(default_factory=dict)
    coord_transform: Any = None  # callable(x,y,w,h)->tuple | None

    # Set by ASLStage — (hand_index, letter) pairs
    asl_letters: list[tuple[int, str]] = field(default_factory=list)

    # Set by CaptureStage — rolling FPS read by RendererStage
    capture_fps: float = 0.0

    # Set by FaceStage — list of (bbox, name, RecognitionStatus) tuples
    face_results: list = field(default_factory=list)
