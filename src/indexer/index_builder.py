"""FAISS index construction and persistence utilities.

Provides both a brute-force flat index (exact search) and an IVF index
(approximate search for scalability demonstrations).  All indices use
inner-product similarity on L2-normalised vectors, which is equivalent to
cosine similarity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FashionIndexBuilder:
    """Build, save, and load FAISS indices and their associated metadata."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        """Initialise the builder.

        Args:
            embedding_dim: Dimensionality of the embedding vectors.
        """
        self.embedding_dim = embedding_dim

    # ── Index construction ───────────────────────────────────────────────
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a flat inner-product index (exact nearest-neighbour).

        Args:
            embeddings: Array of shape ``(N, D)`` - **must** already be
                        L2-normalised so that IP == cosine similarity.

        Returns:
            A populated :class:`faiss.IndexFlatIP`.
        """
        embeddings = self._ensure_float32(embeddings)
        faiss.normalize_L2(embeddings)  # in-place, idempotent on unit vectors

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        logger.info(
            "Built IndexFlatIP - %d vectors, dim=%d",
            index.ntotal,
            self.embedding_dim,
        )
        return index

    def build_scalable_index(
        self,
        embeddings: np.ndarray,
        nlist: int = 100,
    ) -> faiss.Index:
        """Create an IVF index for approximate (scalable) search.

        Falls back to a flat index if there are fewer vectors than
        *nlist* (FAISS requires ``n >= nlist`` for training).

        Args:
            embeddings: Array of shape ``(N, D)``, L2-normalised.
            nlist:      Number of Voronoi cells (inverted lists).

        Returns:
            A trained and populated :class:`faiss.IndexIVFFlat`.
        """
        embeddings = self._ensure_float32(embeddings)
        faiss.normalize_L2(embeddings)

        n_vectors = embeddings.shape[0]

        if n_vectors < nlist:
            logger.warning(
                "Only %d vectors but nlist=%d - falling back to flat index.",
                n_vectors,
                nlist,
            )
            return self.build_index(embeddings)

        quantiser = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFFlat(
            quantiser,
            self.embedding_dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(embeddings)
        index.add(embeddings)
        # Search more cells for better recall at query time
        index.nprobe = min(10, nlist)

        logger.info(
            "Built IndexIVFFlat - %d vectors, nlist=%d, nprobe=%d",
            index.ntotal,
            nlist,
            index.nprobe,
        )
        return index

    # ── Persistence - indices ────────────────────────────────────────────
    @staticmethod
    def save_index(index: faiss.Index, path: Path) -> None:
        """Write a FAISS index to disk.

        Args:
            index: The FAISS index to persist.
            path:  Destination file path (e.g. ``index_dir / "flat.index"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        logger.info("Saved FAISS index → %s (%d vectors)", path, index.ntotal)

    @staticmethod
    def load_index(path: Path) -> faiss.Index:
        """Load a FAISS index from disk.

        Args:
            path: Path to the ``.index`` file.

        Returns:
            The deserialised :class:`faiss.Index`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        index = faiss.read_index(str(path))
        logger.info("Loaded FAISS index ← %s (%d vectors)", path, index.ntotal)
        return index

    # ── Persistence - metadata ───────────────────────────────────────────
    @staticmethod
    def save_metadata(metadata: list[dict[str, Any]], path: Path) -> None:
        """Serialise a list of metadata dicts to JSON.

        Args:
            metadata: One dict per indexed image (tags, file path, etc.).
            path:     Destination ``.json`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
        logger.info("Saved metadata → %s (%d records)", path, len(metadata))

    @staticmethod
    def load_metadata(path: Path) -> list[dict[str, Any]]:
        """Load metadata from a JSON file.

        Args:
            path: Path to the ``.json`` file.

        Returns:
            A list of metadata dicts.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data: list[dict[str, Any]] = json.load(fh)
        logger.info("Loaded metadata ← %s (%d records)", path, len(data))
        return data

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_float32(arr: np.ndarray) -> np.ndarray:
        """Cast to float32 if needed (FAISS requirement)."""
        if arr.dtype != np.float32:
            return arr.astype(np.float32)
        return arr
