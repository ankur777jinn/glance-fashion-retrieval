"""
Vector search module using FAISS for fast similarity retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorSearcher:
    """FAISS-based vector similarity search engine."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        image_paths_path: Optional[Path] = None,
    ):
        """
        Initialize the vector searcher.

        Args:
            index_path: Path to the FAISS index file.
            metadata_path: Path to the metadata JSON file.
            image_paths_path: Path to the image paths JSON file.
        """
        from src.config import INDEX_DIR

        self.index_path = index_path or INDEX_DIR / "flat.index"
        self.metadata_path = metadata_path or INDEX_DIR / "metadata.json"
        self.image_paths_path = image_paths_path or INDEX_DIR / "image_paths.json"

        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self.image_paths: list[str] = []
        self.embeddings: Optional[np.ndarray] = None

        self._load()

    def _load(self) -> None:
        """Load FAISS index, metadata, and image paths from disk."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            logger.warning(f"Index not found at {self.index_path}")

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} images")

        if self.image_paths_path.exists():
            with open(self.image_paths_path, "r", encoding="utf-8") as f:
                self.image_paths = json.load(f)

        # Load embeddings for reranker sub-phrase scoring
        emb_path = self.index_path.parent / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(str(emb_path))
            logger.info(f"Loaded embeddings: {self.embeddings.shape}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 50
    ) -> list[tuple[int, float]]:
        """
        Search for the most similar images.

        Args:
            query_embedding: 1-D or 2-D query vector (will be normalized).
            top_k: Number of results to return.

        Returns:
            List of (image_index, cosine_similarity) tuples.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded.")

        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)

        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append((int(idx), float(dist)))
        return results

    def search_with_metadata(
        self, query_embedding: np.ndarray, top_k: int = 50
    ) -> list[dict]:
        """
        Search and return results enriched with metadata.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: index, similarity, metadata, image_path.
        """
        raw_results = self.search(query_embedding, top_k)
        enriched = []
        for idx, sim in raw_results:
            result = {
                "index": idx,
                "similarity": sim,
                "metadata": self.metadata[idx] if idx < len(self.metadata) else {},
                "image_path": self.image_paths[idx] if idx < len(self.image_paths) else "",
            }
            enriched.append(result)
        return enriched
