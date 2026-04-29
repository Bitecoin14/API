# core/config.py
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Camera
    camera: int = 0
    resolution: tuple[int, int] = (640, 480)
    model_complexity: int = 1

    # Feature flags — all off by default; mode logic in from_args sets implied defaults
    show_hand_skeleton: bool = False
    show_pose: bool = False
    show_face: bool = False
    show_blur: bool = False
    show_asl: bool = False

    # Mode: "hand" | "face" | "enroll" | "audit"
    mode: str = "hand"

    # Enroll-from-folder path (only used in enroll mode)
    enroll_folder: Optional[str] = None
    auto_enroll: bool = False     # skip per-photo prompts in enroll mode

    # MediaPipe confidence thresholds
    hand_detection_conf: float = 0.7
    hand_tracking_conf: float = 0.7
    pose_detection_conf: float = 0.6
    pose_tracking_conf: float = 0.6

    # Paths
    asl_model_path: Path = field(default_factory=lambda: Path("models/asl_classifier.pkl"))
    known_faces_dir: Path = field(default_factory=lambda: Path("known_faces"))
    gallery_dir: Path = field(default_factory=lambda: Path("gallery"))
    models_dir: Path = field(default_factory=lambda: Path("models"))

    # Display
    filter_name: Optional[str] = None
    log_level: str = "WARNING"

    @classmethod
    def from_args(cls, ns: argparse.Namespace) -> "Config":
        res = (640, 480)
        if ns.resolution:
            w, h = ns.resolution.split("x")
            res = (int(w), int(h))
        # Path fields (known_faces_dir, gallery_dir, models_dir, asl_model_path) intentionally
        # use their dataclass defaults; they are not exposed as CLI arguments.
        return cls(
            camera=ns.camera,
            resolution=res,
            model_complexity=ns.model_complexity,
            show_hand_skeleton=(ns.mode == "hand"),  # defining feature of hand mode; no CLI flag needed
            show_face=ns.face or (ns.mode == "face"),
            show_pose=ns.pose,
            show_blur=ns.blur,
            show_asl=ns.asl,
            mode=ns.mode,
            enroll_folder=getattr(ns, "from_folder", None),
            auto_enroll=getattr(ns, "auto", False),
            filter_name=ns.filter,
            log_level=ns.log_level,
        )


# ──────────────────────────────────────────────
# Face recognition constants (module-level, not per-run)
# ──────────────────────────────────────────────

# Detection quality gate
FACE_MIN_SIZE: int = 80
FACE_MIN_DET_SCORE: float = 0.65
FACE_MAX_YAW: float = 35.0
FACE_MIN_BLUR: float = 50.0

# Per-model cosine similarity thresholds
MODEL_A_THRESHOLD: float = 0.45
MODEL_B_THRESHOLD: float = 0.42
MODEL_C_THRESHOLD: float = 0.43
AMBIGUOUS_MARGIN: float = 0.07

# Arbitration base weights
BASE_WEIGHT_A: float = 0.40
BASE_WEIGHT_B: float = 0.30
BASE_WEIGHT_C: float = 0.30

# Temporal smoothing
TRACK_WINDOW_SIZE: int = 15
TRACK_MIN_CONSENSUS: float = 0.60
TRACK_INIT_FRAMES: int = 3
TRACK_TIMEOUT_FRAMES: int = 30

# Enrollment
CONFUSABLE_THRESHOLD: float = 0.35
ENROLLMENT_AUGMENT: bool = True

# Performance
FACE_DETECTOR_INPUT_SIZE: int = 640
YOLO_CROSS_CHECK_SIZE: int = 320
CPU_FALLBACK_TWO_MODELS: bool = True

# Photo enrollment — much more lenient than live-video detection.
# Still photos often have smaller faces, more blur, or slight angles
# compared to a person sitting in front of a webcam.
ENROLL_MIN_SIZE: int = 40          # vs FACE_MIN_SIZE=80 in live mode
ENROLL_MIN_DET_SCORE: float = 0.40  # vs FACE_MIN_DET_SCORE=0.65
ENROLL_MAX_YAW: float = 45.0        # vs FACE_MAX_YAW=35.0
ENROLL_MIN_BLUR: float = 15.0       # vs FACE_MIN_BLUR=50.0
ENROLL_MIN_QUALITY: float = 0.08    # quality_score floor for photo enrollment


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hand_tracker.py",
        description="Real-time hand + body tracker with face recognition, filters, and overlays.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python hand_tracker.py                        # hand skeleton only",
            "  python hand_tracker.py --face                 # hand skeleton + face recognition",
            "  python hand_tracker.py --face --pose --blur   # hand + face + pose + blur",
            "  python hand_tracker.py --mode face            # face recognition only",
            "  python hand_tracker.py --mode enroll          # enroll new person",
            "  python hand_tracker.py --mode enroll --from-folder ./photos/",
            "  python hand_tracker.py --mode audit           # audit gallery for confusable pairs",
            "  python hand_tracker.py --filter Mosaic --resolution 320x240",
            "  python hand_tracker.py --list-filters",
        ]),
    )
    ap.add_argument("-h", "-help", "--help", action="help",
                    default=argparse.SUPPRESS, help="Show this help message and exit.")
    ap.add_argument("--mode", default="hand",
                    choices=["hand", "face", "enroll", "audit"],
                    help="Operating mode (default: hand).")
    ap.add_argument("--from-folder", default=None, metavar="PATH",
                    help="Batch-enroll from a folder of images (enroll mode only).")
    ap.add_argument("--camera", "-c", type=int, default=0, metavar="N",
                    help="Camera device index (default: 0).")
    ap.add_argument("--filter", "-f", default=None, metavar="NAME",
                    help="Start with this filter active (case-insensitive).")
    ap.add_argument("--list-filters", action="store_true", default=False,
                    help="Print available filter names and exit.")
    ap.add_argument("--resolution", default=None, metavar="WxH",
                    help="Inference resolution e.g. 640x480 (default: 640x480).")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                    metavar="N", help="MediaPipe model quality 0/1/2 (default: 1).")
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity (default: WARNING).")
    ap.add_argument("--face", action="store_true", default=False,
                    help="Enable face recognition overlay (colored square + name label). Also implied by --mode face.")
    ap.add_argument("--pose", action="store_true", default=False,
                    help="Enable body pose detection + gesture control.")
    ap.add_argument("--blur", action="store_true", default=False,
                    help="Enable middle-finger blur effect.")
    ap.add_argument("--asl", action="store_true", default=False,
                    help="Enable ASL fingerspelling recognition.")
    ap.add_argument("--auto", action="store_true", default=False,
                    help="Auto-enroll all photos without per-photo prompts (enroll mode only).")
    return ap
