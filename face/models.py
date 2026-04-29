from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("hand_tracker.face.models")

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import insightface
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_OK = True
except ImportError:
    insightface = None          # type: ignore[assignment]
    _FaceAnalysis = None        # type: ignore[assignment]
    _INSIGHTFACE_OK = False
    log.warning(
        "insightface is not installed — Model A unavailable. "
        "Install with: pip install insightface onnxruntime"
    )

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    ort = None                  # type: ignore[assignment]
    _ORT_OK = False
    log.warning(
        "onnxruntime is not installed — Models B and C unavailable. "
        "Install with: pip install onnxruntime (or onnxruntime-gpu)"
    )

try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None                  # type: ignore[assignment]
    _CV2_OK = False
    log.warning("opencv-python is not installed — embedding extraction will fail.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return emb
    return emb / norm


def _ort_providers(use_gpu: bool) -> list[str]:
    if not _ORT_OK:
        return ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _has_cuda(use_gpu: bool) -> bool:
    if not _ORT_OK:
        return False
    return use_gpu and "CUDAExecutionProvider" in ort.get_available_providers()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_arcface(crop: np.ndarray) -> np.ndarray:
    """ArcFace / ElasticFace normalisation: (pixel / 127.5) - 1, NCHW RGB."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required for preprocessing")
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))     # HWC → CHW
    img = np.expand_dims(img, 0)            # (1, 3, 112, 112)
    return img


# ---------------------------------------------------------------------------
# EnsembleRecognizer
# ---------------------------------------------------------------------------

class EnsembleRecognizer:
    """Loads up to 3 independent face-embedding models and fuses them."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        use_gpu: bool = True,
        cpu_fallback_two_models: bool = True,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._use_gpu = use_gpu

        gpu_available = _has_cuda(use_gpu)
        ctx_id = 0 if gpu_available else -1

        # ------------------------------------------------------------------ #
        # Model A: InsightFace antelopev2 (ArcFace / Glint360K)              #
        # ------------------------------------------------------------------ #
        self._model_a: Optional[object] = None
        if _INSIGHTFACE_OK:
            try:
                app = _FaceAnalysis(
                    name="antelopev2",
                    root=str(self._models_dir / "insightface"),
                    allowed_modules=["recognition"],
                )
                app.prepare(ctx_id=ctx_id)
                self._model_a = app.models.get("recognition")
                if self._model_a is None:
                    log.warning(
                        "antelopev2 recognition module not found after prepare(); "
                        "Model A unavailable."
                    )
                else:
                    log.info("Model A (antelopev2 ArcFace) ready (ctx_id=%d).", ctx_id)
            except Exception:
                log.exception("Failed to initialise Model A (antelopev2).")
                self._model_a = None

        providers = _ort_providers(use_gpu)

        # ------------------------------------------------------------------ #
        # Model B: AdaFace IR-ResNet101                                       #
        # ------------------------------------------------------------------ #
        self._model_b: Optional[object] = None
        if _ORT_OK:
            path_b = self._models_dir / "adaface_ir101.onnx"
            if path_b.exists():
                try:
                    self._model_b = ort.InferenceSession(str(path_b), providers=providers)
                    log.info("Model B (AdaFace IR101) loaded from %s.", path_b)
                except Exception:
                    log.exception("Failed to load Model B from %s.", path_b)
                    self._model_b = None
            else:
                log.warning(
                    "Model B (AdaFace IR101) not found at %s — skipping.", path_b
                )

        # ------------------------------------------------------------------ #
        # Model C: ElasticFace-Arc+                                           #
        # ------------------------------------------------------------------ #
        self._model_c: Optional[object] = None
        if _ORT_OK:
            skip_c = (not gpu_available) and cpu_fallback_two_models
            if skip_c:
                log.info(
                    "CPU mode detected: using 2-model ensemble (A + B)"
                )
            else:
                path_c = self._models_dir / "elasticface_arc.onnx"
                if path_c.exists():
                    try:
                        self._model_c = ort.InferenceSession(
                            str(path_c), providers=providers
                        )
                        log.info(
                            "Model C (ElasticFace-Arc+) loaded from %s.", path_c
                        )
                    except Exception:
                        log.exception("Failed to load Model C from %s.", path_c)
                        self._model_c = None
                else:
                    log.warning(
                        "Model C (ElasticFace-Arc+) not found at %s — skipping.", path_c
                    )

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def extract_embeddings(self, crop: np.ndarray) -> dict[str, np.ndarray]:
        """Return L2-normalised 512-d embeddings for all loaded models.

        Args:
            crop: 112×112 BGR uint8 numpy array (aligned face crop).

        Returns:
            Dict mapping model key → (512,) float32 embedding.
            Only contains keys for models that are loaded and functional.
        """
        result: dict[str, np.ndarray] = {}

        # --- Model A ---
        if self._model_a is not None:
            try:
                raw = self._model_a.get_feat(crop)  # (512,) or (1, 512)
                emb = np.array(raw, dtype=np.float32).flatten()
                result["model_a"] = _l2_normalize(emb)
            except Exception:
                log.warning("Model A inference failed.", exc_info=True)

        # --- Model B ---
        if self._model_b is not None:
            try:
                inp = _preprocess_arcface(crop)
                input_name = self._model_b.get_inputs()[0].name
                raw = self._model_b.run(None, {input_name: inp})[0]  # (1, 512)
                emb = np.array(raw, dtype=np.float32).flatten()
                result["model_b"] = _l2_normalize(emb)
            except Exception:
                log.warning("Model B inference failed.", exc_info=True)

        # --- Model C ---
        if self._model_c is not None:
            try:
                inp = _preprocess_arcface(crop)
                input_name = self._model_c.get_inputs()[0].name
                raw = self._model_c.run(None, {input_name: inp})[0]  # (1, 512)
                emb = np.array(raw, dtype=np.float32).flatten()
                result["model_c"] = _l2_normalize(emb)
            except Exception:
                log.warning("Model C inference failed.", exc_info=True)

        return result

    @property
    def available_models(self) -> list[str]:
        models: list[str] = []
        if self._model_a is not None:
            models.append("model_a")
        if self._model_b is not None:
            models.append("model_b")
        if self._model_c is not None:
            models.append("model_c")
        return models

    def close(self) -> None:
        self._model_a = None
        self._model_b = None
        self._model_c = None
        log.debug("EnsembleRecognizer closed.")
