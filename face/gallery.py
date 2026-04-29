from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

log = logging.getLogger("hand_tracker.face.gallery")


class FaceGallery:
    def __init__(self, gallery_dir: str | Path = "gallery") -> None:
        self._gallery_dir = Path(gallery_dir)
        self._embeddings_dir = self._gallery_dir / "embeddings"
        self._metadata_path = self._gallery_dir / "metadata.json"

        self._gallery_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings: dict[str, dict[str, np.ndarray]] = {}
        self._metadata: dict[str, dict] = {}

        self.load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self, embedding: np.ndarray, model_name: str
    ) -> tuple[str, float, str, float]:
        query = embedding / (np.linalg.norm(embedding) + 1e-8)

        names: list[str] = []
        matrix: list[np.ndarray] = []
        for key, embs in self._embeddings.items():
            if model_name in embs:
                names.append(self._metadata[key].get("display_name", key))
                matrix.append(embs[model_name])

        if not names:
            return ("Unknown", 0.0, "Unknown", 0.0)

        mat = np.array(matrix, dtype=np.float32)  # (N, 512)
        scores = mat @ query.astype(np.float32)    # cosine similarity

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_name = names[best_idx]

        scores[best_idx] = -1.0
        ru_idx = int(np.argmax(scores))
        ru_score = float(scores[ru_idx])
        ru_name = names[ru_idx] if len(names) > 1 else "Unknown"

        return (best_name, best_score, ru_name, ru_score)

    def add_person(
        self,
        name: str,
        embeddings: dict[str, np.ndarray],
        metadata: dict,
    ) -> None:
        key = self._sanitize(name)

        for model_name, emb in embeddings.items():
            path = self._embeddings_dir / f"{key}_{model_name}.npy"
            np.save(str(path), emb.astype(np.float32))
            log.debug("Saved embedding %s", path.name)

        self._embeddings[key] = {k: v.astype(np.float32) for k, v in embeddings.items()}

        meta: dict = {"display_name": name, **metadata}
        meta.setdefault("enrolled_at", datetime.utcnow().isoformat() + "Z")
        self._metadata[key] = meta

        self.save()
        log.info("Added person '%s' (key=%s) with models: %s", name, key, list(embeddings))

    def remove_person(self, name: str) -> None:
        key = self._sanitize(name)
        if key not in self._metadata:
            log.warning("remove_person: key '%s' not found in gallery", key)
            return

        # Delete .npy files
        for model_name in list(self._embeddings.get(key, {})):
            path = self._embeddings_dir / f"{key}_{model_name}.npy"
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not delete %s: %s", path, exc)

        self._embeddings.pop(key, None)
        self._metadata.pop(key, None)
        self.save()
        log.info("Removed person '%s' (key=%s)", name, key)

    def save(self) -> None:
        try:
            with open(self._metadata_path, "w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2)
        except OSError as exc:
            log.error("Failed to write metadata.json: %s", exc)

    def load(self) -> None:
        self._metadata = {}
        self._embeddings = {}

        if not self._metadata_path.exists():
            log.debug("No metadata.json found; gallery is empty")
            return

        try:
            with open(self._metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Failed to load metadata.json: %s", exc)
            self._metadata = {}
            return

        for key in list(self._metadata):
            self._embeddings[key] = {}
            # Discover model names from filenames matching <key>_<model>.npy
            prefix = f"{key}_"
            for npy_path in self._embeddings_dir.glob(f"{prefix}*.npy"):
                model_name = npy_path.stem[len(prefix):]
                try:
                    arr = np.load(str(npy_path)).astype(np.float32)
                    self._embeddings[key][model_name] = arr
                    log.debug("Loaded embedding %s", npy_path.name)
                except (OSError, ValueError) as exc:
                    log.warning("Skipping corrupted file %s: %s", npy_path.name, exc)

        log.info(
            "Gallery loaded: %d people, models per person: %s",
            len(self._metadata),
            {k: list(v) for k, v in self._embeddings.items()},
        )

    def compute_similarity_matrix(
        self, threshold: float = 0.35
    ) -> list[tuple[str, str, dict[str, float]]]:
        keys = list(self._embeddings)
        results: list[tuple[str, str, dict[str, float]]] = []

        for key_a, key_b in combinations(keys, 2):
            embs_a = self._embeddings[key_a]
            embs_b = self._embeddings[key_b]
            shared_models = set(embs_a) & set(embs_b)
            if not shared_models:
                continue

            model_scores: dict[str, float] = {}
            for model in shared_models:
                a = embs_a[model] / (np.linalg.norm(embs_a[model]) + 1e-8)
                b = embs_b[model] / (np.linalg.norm(embs_b[model]) + 1e-8)
                model_scores[model] = float(np.dot(a, b))

            if any(s > threshold for s in model_scores.values()):
                name_a = self._metadata[key_a].get("display_name", key_a)
                name_b = self._metadata[key_b].get("display_name", key_b)
                results.append((name_a, name_b, model_scores))

        return results

    @property
    def people(self) -> list[str]:
        return [self._metadata[k].get("display_name", k) for k in self._metadata]

    def get_metadata(self, name: str) -> dict:
        key = self._sanitize(name)
        return dict(self._metadata.get(key, {}))

    def update_metadata(self, name: str, updates: dict) -> None:
        key = self._sanitize(name)
        if key not in self._metadata:
            log.warning("update_metadata: key '%s' not found", key)
            return
        self._metadata[key].update(updates)
        self.save()

    def is_empty(self) -> bool:
        return len(self._metadata) == 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(name: str) -> str:
        key = name.lower().replace(" ", "_")
        key = re.sub(r"[^a-z0-9_]", "", key)
        return key
