"""Enrollment pipeline: register new people into the face gallery."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from face.gallery import FaceGallery
    from face.models import EnsembleRecognizer
    from face.detector import DualDetector

from face.types import DetectedFace

log = logging.getLogger("hand_tracker.face.enrollment")

# Augmentation: 5 variants from one capture
_AUGMENTATIONS = ["original", "flip", "bright_up", "bright_down", "blur"]


def _augment(crop: np.ndarray) -> list[np.ndarray]:
    """Generate 5 variants of a 112×112 face crop for robust enrollment."""
    variants = [crop.copy()]

    # Horizontal flip
    variants.append(cv2.flip(crop, 1))

    # Brightness +15%
    bright_up = np.clip(crop.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    variants.append(bright_up)

    # Brightness -15%
    bright_down = np.clip(crop.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    variants.append(bright_down)

    # Mild Gaussian blur σ=0.5
    variants.append(cv2.GaussianBlur(crop, (3, 3), 0.5))

    return variants


def _mean_embedding(crops: list[np.ndarray], recognizer: "EnsembleRecognizer",
                    ) -> dict[str, np.ndarray]:
    """Extract embeddings for all augmented crops, return L2-normalized mean per model."""
    per_model: dict[str, list[np.ndarray]] = {}
    for crop in crops:
        embs = recognizer.extract_embeddings(crop)
        for model_name, emb in embs.items():
            per_model.setdefault(model_name, []).append(emb)

    result: dict[str, np.ndarray] = {}
    for model_name, emb_list in per_model.items():
        mean_emb = np.mean(emb_list, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        result[model_name] = mean_emb.astype(np.float32)
    return result


def enroll_from_capture(
    detector: "DualDetector",
    recognizer: "EnsembleRecognizer",
    gallery: "FaceGallery",
    camera_id: int = 0,
) -> None:
    """Interactive enrollment mode: capture from webcam, confirm, save."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        log.error("Cannot open camera %d for enrollment", camera_id)
        return

    print("\n╔══════════════════════════════╗")
    print("║      ENROLLMENT MODE         ║")
    print("╠══════════════════════════════╣")
    print("║  SPACE  — capture face        ║")
    print("║  Q/ESC  — quit enrollment     ║")
    print("╚══════════════════════════════╝\n")

    while True:
        name = input("Enter person's name (or 'q' to quit): ").strip()
        if name.lower() == "q":
            break
        if not name:
            continue

        print(f"Frame the face of '{name}' and press SPACE to capture. Press 'q' to skip.")
        captured_face: DetectedFace | None = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect(frame)
            display = frame.copy()

            if faces:
                for face in faces:
                    x1, y1, x2, y2 = face.bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Enrolling: {name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display, "SPACE = capture  |  Q = skip", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(display, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                captured_face = None
                break
            elif key == ord(" "):
                if not faces:
                    print("No face detected. Please reframe and try again.")
                    continue
                if len(faces) > 1:
                    print(f"Multiple faces detected ({len(faces)}). Please ensure only one face is visible.")
                    continue
                face = faces[0]
                if face.quality_score < 0.4:
                    print(f"Quality too low ({face.quality_score:.2f}). "
                          "Ensure good lighting and face the camera.")
                    continue
                captured_face = face
                print(f"Face captured! Quality: {face.quality_score:.2f}")
                break

        if captured_face is None or captured_face.crop is None:
            print(f"Skipping '{name}'.\n")
            continue

        # Show captured crop and ask for confirmation
        crop_display = cv2.resize(captured_face.crop, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Captured face — press any key", crop_display)
        cv2.waitKey(0)
        cv2.destroyWindow("Captured face — press any key")

        while True:
            answer = input(f"Enrolled '{name}'. Is this correct? (y/n/r): ").strip().lower()
            if answer == "y":
                _commit_enrollment(name, captured_face, recognizer, gallery)
                break
            elif answer == "n":
                print("Discarding. Please re-enter a name or skip.\n")
                break
            elif answer == "r":
                print("Retaking photo. Press SPACE to capture.\n")
                # Re-enter the capture loop for the same name
                captured_face = None
                break
            else:
                print("Please enter y, n, or r.")

    cap.release()
    cv2.destroyAllWindows()


def enroll_from_folder(
    folder: str | Path,
    detector: "DualDetector",
    recognizer: "EnsembleRecognizer",
    gallery: "FaceGallery",
    auto: bool = False,
) -> None:
    """Batch-enroll from a folder of images.

    Filenames become display names:
      ``rui.jpg``          → ``Rui``
      ``alice_johnson.jpg`` → ``Alice Johnson``
      ``bob-smith.png``    → ``Bob Smith``

    When *auto* is True every detected face is enrolled without prompting —
    useful for large unattended batches.
    """
    from core.config import ENROLL_MIN_QUALITY

    folder = Path(folder)
    if not folder.exists():
        log.error("Folder not found: %s", folder)
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in image_extensions)

    if not images:
        print(f"No images found in {folder}")
        return

    print(f"\nFound {len(images)} images in {folder}")
    if auto:
        print("Auto mode: enrolling all detected faces without prompting.\n")
    else:
        print()

    ok = skipped = 0

    for img_path in images:
        # Derive display name: "alice_johnson.jpg" → "Alice Johnson", "rui.jpg" → "Rui"
        stem = img_path.stem.replace("_", " ").replace("-", " ")
        name = " ".join(word.capitalize() for word in stem.split())

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ✗ Cannot read: {img_path.name}")
            skipped += 1
            continue

        faces = detector.detect(img)
        if not faces:
            print(f"  ✗ No face detected: {img_path.name}")
            skipped += 1
            continue
        if len(faces) > 1:
            print(f"  ! {len(faces)} faces in {img_path.name} — using highest quality")

        face = max(faces, key=lambda f: f.quality_score)

        if face.quality_score < ENROLL_MIN_QUALITY:
            print(f"  ✗ Quality too low ({face.quality_score:.2f}): {img_path.name}")
            skipped += 1
            continue

        if auto:
            _commit_enrollment(name, face, recognizer, gallery)
            ok += 1
            continue

        # Interactive: show preview and confirm
        if face.crop is not None:
            crop_display = cv2.resize(face.crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(f"Preview: {name}", crop_display)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

        while True:
            answer = input(f"Enroll '{name}' from {img_path.name}? (y/n/r/a): ").strip().lower()
            if answer == "y":
                _commit_enrollment(name, face, recognizer, gallery)
                ok += 1
                break
            elif answer == "n":
                print(f"  Skipped {img_path.name}")
                skipped += 1
                break
            elif answer == "r":
                custom = input(f"  Enter correct name for {img_path.name}: ").strip()
                if custom:
                    name = custom
                    _commit_enrollment(name, face, recognizer, gallery)
                    ok += 1
                break
            elif answer == "a":
                print("  Auto mode on — enrolling remaining images without prompting.")
                auto = True
                _commit_enrollment(name, face, recognizer, gallery)
                ok += 1
                break
            else:
                print("  Please enter y, n, r (rename), or a (auto-enroll all remaining).")

    print(f"\nBatch enrollment complete: {ok} enrolled, {skipped} skipped.")


def _commit_enrollment(
    name: str,
    face: DetectedFace,
    recognizer: "EnsembleRecognizer",
    gallery: "FaceGallery",
) -> None:
    """Extract embeddings, augment, save to gallery, check for confusables."""
    assert face.crop is not None, "Face crop must be set before enrollment"

    print(f"  Extracting embeddings for '{name}'...")
    variants = _augment(face.crop)
    embeddings = _mean_embedding(variants, recognizer)

    if not embeddings:
        print(f"  ✗ No models available for embedding. Cannot enroll '{name}'.")
        return

    from face.attributes import extract_attributes
    attrs = extract_attributes(face, np.zeros((480, 640, 3), np.uint8), insight_face_obj=None)

    metadata = {
        "enrollment_quality": round(face.quality_score, 4),
        "has_glasses": attrs.glasses_detected,
        "yaw_at_enrollment": round(face.yaw, 2),
    }

    gallery.add_person(name, embeddings, metadata)

    # Confusable-pair warning
    from core.config import CONFUSABLE_THRESHOLD
    confusables = gallery.compute_similarity_matrix(threshold=CONFUSABLE_THRESHOLD)
    sanitized_name = gallery._sanitize(name)
    for name_a, name_b, scores in confusables:
        key_a = gallery._sanitize(name_a)
        key_b = gallery._sanitize(name_b)
        if sanitized_name in (key_a, key_b):
            other = name_b if key_a == sanitized_name else name_a
            score_str = "  ".join(
                f"{m}: {s:.2f}" for m, s in scores.items()
            )
            print(f"\n  ⚠ WARNING: '{name}' looks similar to '{other}'")
            print(f"    Scores → {score_str}")
            print("    Consider enrolling a second photo for both.\n")

    print(f"  ✓ '{name}' enrolled successfully "
          f"({len(embeddings)} model(s), {len(variants)} augmentations)")


def run_audit(gallery: "FaceGallery", threshold: float = 0.35) -> None:
    """Print N×N confusable pairs report and flag them in metadata."""
    if gallery.is_empty():
        print("Gallery is empty.")
        return

    confusables = gallery.compute_similarity_matrix(threshold)
    print("\n═══════════════════════════════════")
    print("      CONFUSABLE PAIRS REPORT      ")
    print("═══════════════════════════════════")

    if not confusables:
        print(f"\nAll pairs: max similarity < {threshold:.2f} ✓")
    else:
        for name_a, name_b, scores in confusables:
            max_score = max(scores.values())
            level = "HIGH RISK" if max_score > 0.45 else "WARNING"
            print(f"\n{level}: {name_a} ↔ {name_b}")
            for model, score in scores.items():
                print(f"   {model}: {score:.2f}")
            if max_score > 0.45:
                print("   Recommendation: require unanimous consensus for these identities.")
            else:
                print("   Recommendation: consider enrolling a second photo for both.")

            # Flag in metadata so arbitration uses stricter rules
            key_a = gallery._sanitize(name_a)
            key_b = gallery._sanitize(name_b)
            gallery.update_metadata(key_a, {"confusable_with": name_b})
            gallery.update_metadata(key_b, {"confusable_with": name_a})

    print("\n═══════════════════════════════════\n")
    gallery.save()
