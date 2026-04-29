# hand_tracker.py
from __future__ import annotations

import sys
import logging
from pathlib import Path

from core.config import Config, build_parser
from core.logging_setup import configure_logging
from core.pipeline import Pipeline
from filters import FILTERS
from stages.capture import CaptureStage
from stages.inference import InferenceStage
from stages.filter_stage import FilterStage
from stages.face_stage import FaceStage
from stages.asl_stage import ASLStage
from stages.renderer import RendererStage

log = logging.getLogger("hand_tracker")


def _run_hand_mode(config: Config) -> None:
    """Standard hand-tracking pipeline. Always runs FilterStage → RendererStage;
    FaceStage and ASLStage are added only when their feature flag is enabled."""
    render_stages = [FilterStage(initial_name=config.filter_name)]
    if config.show_face:
        render_stages.append(FaceStage(
            gallery_dir=config.gallery_dir,
            models_dir=config.models_dir,
            known_faces_dir=config.known_faces_dir,
        ))
    if config.show_asl:
        render_stages.append(ASLStage(model_path=config.asl_model_path))
    render_stages.append(RendererStage())

    pipeline = Pipeline(
        capture_stage=CaptureStage(config),
        inference_stage=InferenceStage(config),
        render_stages=render_stages,
    )
    pipeline.run()


def _run_face_mode(config: Config) -> None:
    """Face-recognition-primary pipeline (no hand/ASL processing)."""
    assert config.show_face, "face mode requires show_face=True; use Config.from_args() to ensure this"
    pipeline = Pipeline(
        capture_stage   = CaptureStage(config),
        inference_stage = InferenceStage(config),
        render_stages   = [
            FaceStage(
                gallery_dir=config.gallery_dir,
                models_dir=config.models_dir,
                known_faces_dir=config.known_faces_dir,
            ),
            RendererStage(),
        ],
    )
    pipeline.run()


def _resolve_enroll_folder(config: Config) -> str | None:
    """Return the folder to enroll from, or None to use the webcam.

    Priority:
      1. --from-folder CLI argument (explicit)
      2. photos/ in the project directory (drag-and-drop default)
      3. None → fall through to webcam capture
    """
    if config.enroll_folder:
        return config.enroll_folder

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    default = Path("photos")
    if default.exists() and any(
        f.suffix.lower() in _IMAGE_EXTS for f in default.iterdir()
    ):
        print(f"Found photos in '{default}/' — using folder enrollment.")
        print("(Drag photos named after each person into that folder.)\n")
        return str(default)

    return None


def _run_enroll_mode(config: Config) -> None:
    """Enrollment mode: register people into the face gallery.

    Folder detection order:
      1. --from-folder path (explicit CLI argument)
      2. photos/ folder in the project directory (drag photos there)
      3. Live webcam capture (fallback when photos/ is empty)

    Uses the 3-model ensemble when InsightFace is installed; falls back to
    the OpenCV/LBPH pipeline (known_faces/ folder structure) otherwise.
    """
    enroll_folder = _resolve_enroll_folder(config)

    try:
        import insightface  # noqa: F401 — presence check; raises ImportError if absent
        from face.detector import DualDetector
        from face.models import EnsembleRecognizer
        from face.gallery import FaceGallery
        from face.enrollment import enroll_from_capture, enroll_from_folder
        from core.config import (
            ENROLL_MIN_SIZE, ENROLL_MIN_DET_SCORE, ENROLL_MAX_YAW, ENROLL_MIN_BLUR,
            FACE_DETECTOR_INPUT_SIZE, YOLO_CROSS_CHECK_SIZE, CPU_FALLBACK_TWO_MODELS,
        )
        _ensemble_ok = True
    except ImportError as e:
        log.warning("Ensemble enrollment unavailable (%s). Falling back to LBPH.", e)
        _ensemble_ok = False

    if not _ensemble_ok:
        _run_legacy_enroll(config, enroll_folder)
        return

    # Use lenient thresholds for still-photo enrollment
    detector = DualDetector(
        det_size=FACE_DETECTOR_INPUT_SIZE,
        cross_check_size=YOLO_CROSS_CHECK_SIZE,
        min_face_px=ENROLL_MIN_SIZE,
        min_det_score=ENROLL_MIN_DET_SCORE,
        max_yaw=ENROLL_MAX_YAW,
        min_blur=ENROLL_MIN_BLUR,
    )
    recognizer = EnsembleRecognizer(
        models_dir=config.models_dir,
        cpu_fallback_two_models=CPU_FALLBACK_TWO_MODELS,
    )
    gallery = FaceGallery(gallery_dir=config.gallery_dir)

    try:
        if enroll_folder:
            enroll_from_folder(
                folder=enroll_folder,
                detector=detector,
                recognizer=recognizer,
                gallery=gallery,
                auto=config.auto_enroll,
            )
        else:
            enroll_from_capture(
                detector=detector,
                recognizer=recognizer,
                gallery=gallery,
                camera_id=config.camera,
            )
    finally:
        detector.close()
        recognizer.close()


# ---------------------------------------------------------------------------
# LBPH fallback enrollment (no InsightFace required)
# ---------------------------------------------------------------------------

def _read_image_correct_rotation(img_path: Path):
    """Read an image and correct EXIF rotation (phones often store portrait
    photos as landscape with a rotation tag that OpenCV ignores)."""
    import cv2
    try:
        from PIL import Image, ImageOps
        import numpy as np
        pil_img = Image.open(str(img_path))
        pil_img = ImageOps.exif_transpose(pil_img)   # applies EXIF rotation
        pil_img = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        return cv2.imread(str(img_path))


def _detect_faces_mediapipe(img_bgr):
    """Detect faces with MediaPipe (more robust than Haar cascade).
    Returns list of (x, y, w, h) pixel tuples sorted largest-first."""
    import cv2
    import mediapipe as mp

    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_fd = mp.solutions.face_detection

    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.3) as det:
        res = det.process(rgb)

    if not res.detections:
        return []

    boxes = []
    for d in res.detections:
        bb = d.location_data.relative_bounding_box
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        bw = min(int(bb.width * w), w - x)
        bh = min(int(bb.height * h), h - y)
        if bw > 0 and bh > 0:
            boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)  # largest first
    return boxes


def _run_legacy_enroll(config: Config, enroll_folder: str | None = None) -> None:
    """Enroll using MediaPipe face detection + LBPH when InsightFace is absent.

    Folder mode: copies photos into known_faces/<name>/ so the LBPH loader
    picks them up on next launch.
    Camera mode: live capture into the same structure.
    """
    known_faces_dir = Path(config.known_faces_dir)
    known_faces_dir.mkdir(parents=True, exist_ok=True)

    if enroll_folder:
        _legacy_enroll_from_folder(Path(enroll_folder), known_faces_dir)
    else:
        _legacy_enroll_from_capture(config.camera, known_faces_dir)


def _legacy_enroll_from_folder(folder: Path, known_faces_dir: Path) -> None:
    """Copy photos from a flat folder into known_faces/<name>/ for LBPH training.

    Accepts any filename: ``rui.jpg`` -> person ``Rui``,
    ``alice_johnson.jpg`` -> person ``Alice Johnson``.
    Uses MediaPipe for face detection (handles EXIF rotation, various angles).
    """
    import shutil

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in image_extensions)

    if not images:
        print(f"No images found in {folder}")
        return

    print(f"\nFound {len(images)} images. Enrolling into LBPH gallery...\n")
    print("(Install insightface + onnxruntime for the higher-accuracy ensemble system.)\n")

    ok = skipped = 0

    for img_path in images:
        stem = img_path.stem.replace("_", " ").replace("-", " ")
        name = " ".join(w.capitalize() for w in stem.split())

        img = _read_image_correct_rotation(img_path)
        if img is None:
            print(f"  X Cannot read: {img_path.name}")
            skipped += 1
            continue

        faces = _detect_faces_mediapipe(img)

        if not faces:
            print(f"  X No face detected: {img_path.name}")
            skipped += 1
            continue

        if len(faces) > 1:
            print(f"  ! {len(faces)} faces in {img_path.name} — using largest")

        # Save the EXIF-corrected image so LBPH loader can re-detect from it
        import cv2
        person_dir = known_faces_dir / name
        person_dir.mkdir(exist_ok=True)
        dest = person_dir / img_path.name
        cv2.imwrite(str(dest), img)
        print(f"  OK {name:<25}  <- {img_path.name}")
        ok += 1

    print(f"\nDone: {ok} enrolled, {skipped} skipped.")
    print(f"Gallery saved to: {known_faces_dir}")
    print("\nRun 'python hand_tracker.py --mode face' to start recognition.")


def _legacy_enroll_from_capture(camera_id: int, known_faces_dir: Path) -> None:
    """Interactive LBPH enrollment from webcam into known_faces/."""
    import cv2
    import mediapipe as mp

    mp_fd = mp.solutions.face_detection
    face_detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        face_detector.close()
        return

    print("\n╔══════════════════════════════╗")
    print("║   ENROLLMENT MODE (LBPH)     ║")
    print("╠══════════════════════════════╣")
    print("║  SPACE  — capture face       ║")
    print("║  Q/ESC  — quit enrollment    ║")
    print("╚══════════════════════════════╝")
    print("(Install insightface + onnxruntime for the higher-accuracy ensemble system.)\n")

    def _detect_live(frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = face_detector.process(rgb)
        if not res.detections:
            return []
        boxes = []
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width * w), w - x)
            bh = min(int(bb.height * h), h - y)
            if bw > 0 and bh > 0:
                boxes.append((x, y, bw, bh))
        return boxes

    try:
        while True:
            name = input("Enter person's name (or 'q' to quit): ").strip()
            if name.lower() == "q":
                break
            if not name:
                continue

            print(f"Frame the face of '{name}' and press SPACE to capture.")
            captured = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                display = frame.copy()
                faces = _detect_live(frame)

                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                label = f"Enrolling: {name}  |  SPACE = capture" if faces else "No face detected"
                color = (0, 255, 0) if faces else (0, 0, 255)
                cv2.putText(display, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                cv2.imshow("Enrollment", display)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    captured = None
                    break
                elif key == ord(" "):
                    if not faces:
                        print("  No face detected — reframe and try again.")
                        continue
                    x, y, w, h = faces[0]
                    captured = frame[y:y + h, x:x + w].copy()
                    print("  Face captured!")
                    break

            if captured is None:
                print(f"  Skipping '{name}'.\n")
                continue

            preview = cv2.resize(captured, (200, 200))
            cv2.imshow("Captured — press any key", preview)
            cv2.waitKey(0)
            cv2.destroyWindow("Captured — press any key")

            while True:
                ans = input(f"Save as '{name}'? (y/n/r to retake): ").strip().lower()
                if ans == "y":
                    person_dir = known_faces_dir / name
                    person_dir.mkdir(exist_ok=True)
                    count = len(list(person_dir.glob("*.jpg")))
                    cv2.imwrite(str(person_dir / f"{count:04d}.jpg"), captured)
                    print(f"  OK '{name}' saved.\n")
                    break
                elif ans == "n":
                    print("  Discarded.\n")
                    break
                elif ans == "r":
                    captured = None
                    break
                else:
                    print("  Please enter y, n, or r.")

    finally:
        face_detector.close()
        cap.release()
        cv2.destroyAllWindows()


def _run_audit_mode(config: Config) -> None:
    """Audit mode: print confusable-pairs report for the enrolled gallery."""
    try:
        from face.gallery import FaceGallery
        from face.enrollment import run_audit
        from core.config import CONFUSABLE_THRESHOLD
    except ImportError as e:
        log.error("Face recognition modules not available: %s", e)
        sys.exit(1)

    gallery = FaceGallery(gallery_dir=config.gallery_dir)
    run_audit(gallery, threshold=CONFUSABLE_THRESHOLD)


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()

    if ns.list_filters:
        print("Available filters:")
        for f in FILTERS:
            print(f"  {f['name']}")
        sys.exit(0)

    configure_logging(ns.log_level)
    config = Config.from_args(ns)

    mode = config.mode
    log.info("Starting in mode: %s", mode)

    if mode == "hand":
        _run_hand_mode(config)
    elif mode == "face":
        _run_face_mode(config)
    elif mode == "enroll":
        _run_enroll_mode(config)
    elif mode == "audit":
        _run_audit_mode(config)
    else:
        log.error("Unknown mode: %s", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
