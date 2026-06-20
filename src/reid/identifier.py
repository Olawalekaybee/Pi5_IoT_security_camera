"""
Person re-identification: extracts an embedding for each detected
person crop (via a ResNet Re-ID model on the Hailo NPU, or a CPU mock),
and matches it against a gallery of known embeddings using cosine
similarity to decide "known" vs "unknown".
"""

from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

from src.detection.hailo_engine import HailoInferenceEngine

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class PersonIdentifier:
    """Maintains a gallery of known person embeddings and matches new crops."""

    def __init__(self, model_path: str, known_faces_dir: str, threshold: float = 0.75):
        self.engine = HailoInferenceEngine(model_path, input_shape=(128, 256))
        self.threshold = threshold
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.gallery: Dict[str, np.ndarray] = {}
        self._load_gallery()

    def _gallery_index_path(self) -> Path:
        return self.known_faces_dir / "embeddings.json"

    def _load_gallery(self) -> None:
        index_path = self._gallery_index_path()
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            self.gallery = {name: np.array(vec) for name, vec in data.items()}
            logger.info(f"Loaded {len(self.gallery)} known person embeddings")
        else:
            logger.info("No known-faces gallery found yet — all detections will be 'unknown'")

    def _save_gallery(self) -> None:
        data = {name: vec.tolist() for name, vec in self.gallery.items()}
        with open(self._gallery_index_path(), "w") as f:
            json.dump(data, f)

    def _embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Run the Re-ID model on a person crop to get a feature embedding.
        The real osnet_x1_0 HEF outputs a 512-dim UINT8 vector (vstream
        'fc49'). Cosine similarity is scale-invariant, so the raw 0-255
        UINT8 range doesn't need rescaling before normalizing to unit
        length below.
        """
        EMBED_DIM = 512

        if crop is None or crop.size == 0:
            return np.zeros(EMBED_DIM)

        if self.engine.mock_mode:
            # In mock mode, derive a deterministic embedding from pixel
            # statistics rather than calling the detection-style mock
            # inference (which is randomized and not meant for Re-ID).
            # This keeps identical crops mapping to identical embeddings,
            # matching how a real Re-ID model behaves.
            flat = crop.astype(np.float64).flatten()
            stats = np.array([
                flat.mean(), flat.std(),
                np.percentile(flat, 25), np.percentile(flat, 75),
                crop.shape[0], crop.shape[1],
            ])
            rng = np.random.default_rng(seed=int(abs(stats.sum())) % (2**32))
            noise = rng.standard_normal(EMBED_DIM - len(stats)) * 0.0
            vec = np.concatenate([stats, noise])
            vec = np.resize(vec, EMBED_DIM)
        else:
            raw = self.engine.infer(crop)
            vec = np.asarray(raw, dtype=np.float64).flatten()
            if vec.size != EMBED_DIM:
                logger.warning(
                    f"Re-ID embedding size {vec.size} != expected {EMBED_DIM} "
                    "— check the HEF output vstream shape"
                )
                vec = np.resize(vec, EMBED_DIM)

        return vec / (np.linalg.norm(vec) + 1e-12)

    def enroll(self, name: str, crop: np.ndarray) -> None:
        """Add a new known person to the gallery."""
        embedding = self._embed(crop)
        self.gallery[name] = embedding
        self._save_gallery()
        logger.info(f"Enrolled '{name}' into known-persons gallery")

    def identify(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Returns (person_id, confidence). person_id is None if no
        gallery match clears the similarity threshold.
        """
        if not self.gallery:
            return None, 0.0

        embedding = self._embed(crop)
        best_name, best_score = None, -1.0
        for name, ref_vec in self.gallery.items():
            score = cosine_similarity(embedding, ref_vec)
            if score > best_score:
                best_name, best_score = name, score

        if best_score >= self.threshold:
            return best_name, best_score
        return None, best_score

    def close(self) -> None:
        self.engine.close()