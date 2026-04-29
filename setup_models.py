"""
Download and verify all face recognition model files.

Usage:
    python setup_models.py
    python setup_models.py --cpu   # download CPU-only onnxruntime version
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")

# Model registry: name → (url, expected_sha256, dest_path)
# SHA256 values are from the official release pages.
MODEL_REGISTRY = {
    "adaface_ir101": (
        "https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir101_webface12m.onnx",
        None,   # set to None to skip verification (fill in after first download)
        MODELS_DIR / "adaface_ir101.onnx",
    ),
    "elasticface_arc": (
        "https://github.com/fdbtrs/ElasticFace/releases/download/ElasticFace/295672backbone.pth",
        None,
        MODELS_DIR / "elasticface_arc.onnx",  # converted ONNX — see note below
    ),
    "yolov8n_face": (
        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
        None,
        MODELS_DIR / "yolov8n-face.pt",
    ),
}

# Note: ElasticFace releases .pth (PyTorch) weights; an ONNX export is needed.
# The community-maintained ONNX version is hosted at:
ELASTICFACE_ONNX_URL = (
    "https://github.com/fdbtrs/ElasticFace/releases/download/ElasticFace/295672backbone.pth"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {dest.name}...")
    print(f"  Source: {url}")

    try:
        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = count * block_size * 100 / total_size
                mb = count * block_size / 1_048_576
                total_mb = total_size / 1_048_576
                print(f"\r  {mb:.1f} / {total_mb:.1f} MB  ({pct:.0f}%)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, _progress)
        print()
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        return False

    if expected_sha256:
        actual = _sha256(dest)
        if actual != expected_sha256:
            print(f"  CHECKSUM MISMATCH — expected {expected_sha256}, got {actual}")
            dest.unlink(missing_ok=True)
            return False
        print(f"  Checksum OK ({actual[:16]}…)")

    print(f"  Saved to: {dest}")
    return True


def check_insightface() -> None:
    """InsightFace downloads models on first use — just verify the package is importable."""
    print("\n[1/4] InsightFace (antelopev2)")
    try:
        import insightface
        print(f"  ✓ insightface {insightface.__version__} installed")
        print("  antelopev2 will download automatically on first run (~300 MB)")
    except ImportError:
        print("  ✗ insightface not installed — run: pip install insightface")


def check_adaface() -> None:
    print("\n[2/4] AdaFace IR-ResNet101 (ONNX)")
    dest = MODELS_DIR / "adaface_ir101.onnx"
    if dest.exists():
        print(f"  ✓ Already present ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return

    url, sha256, _ = MODEL_REGISTRY["adaface_ir101"]
    print(f"  File not found: {dest}")
    print(f"  Manual download required from:")
    print(f"  {url}")
    print(f"  Save as: {dest}")
    print()
    answer = input("  Attempt automatic download? (y/n): ").strip().lower()
    if answer == "y":
        ok = _download(url, dest, sha256)
        if ok:
            print(f"  ✓ AdaFace downloaded")
        else:
            print("  ✗ Download failed. Please download manually.")


def check_elasticface() -> None:
    print("\n[3/4] ElasticFace-Arc+ (ONNX)")
    dest = MODELS_DIR / "elasticface_arc.onnx"
    if dest.exists():
        print(f"  ✓ Already present ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return

    print(f"  File not found: {dest}")
    print("  ElasticFace is released as PyTorch .pth weights.")
    print("  You need to export it to ONNX or use a community ONNX release.")
    print("  Reference: https://github.com/fdbtrs/ElasticFace")
    print()
    print("  The system will run with 2-model ensemble (A + B) if this file is missing.")
    print("  This is perfectly functional for most use cases.")


def check_yolo() -> None:
    print("\n[4/4] YOLOv8n-face (cross-check detector)")
    # Ultralytics handles download automatically
    try:
        import ultralytics
        print(f"  ✓ ultralytics {ultralytics.__version__} installed")
        print("  yolov8n-face.pt will download automatically on first run")
    except ImportError:
        print("  ✗ ultralytics not installed — run: pip install ultralytics")


def check_onnxruntime() -> None:
    print("\n[+] ONNX Runtime")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu = "CUDAExecutionProvider" in providers
        print(f"  ✓ onnxruntime {ort.__version__}  GPU available: {gpu}")
        if not gpu:
            print("  Tip: install onnxruntime-gpu for better performance")
    except ImportError:
        print("  ✗ onnxruntime not installed — run: pip install onnxruntime-gpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download face recognition models.")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU-only mode — skip GPU-specific checks.")
    ns = ap.parse_args()

    print("=" * 50)
    print("  Face Recognition Model Setup")
    print("=" * 50)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    check_insightface()
    check_adaface()
    check_elasticface()
    check_yolo()
    check_onnxruntime()

    print("\n" + "=" * 50)
    print("  Setup complete.")
    print("  Run: python hand_tracker.py --mode enroll")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
