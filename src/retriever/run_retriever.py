"""
Main retrieval pipeline - ties together query decomposition,
vector search, and hybrid re-ranking.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    INDEX_DIR,
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
    DEVICE,
    MODEL_NAME,
)
from src.retriever.query_decomposer import QueryDecomposer
from src.retriever.vector_search import VectorSearcher
from src.retriever.reranker import HybridReranker

logger = logging.getLogger(__name__)


class FashionRetriever:
    """
    End-to-end fashion image retrieval system.

    Pipeline:
        query → decompose → encode → FAISS search → hybrid rerank → top-k
    """

    def __init__(self, index_dir: Optional[Path] = None):
        """
        Load all components.

        Args:
            index_dir: Override for index directory path.
        """
        self._index_dir = index_dir or INDEX_DIR

        logger.info("Initializing FashionRetriever...")

        # 1. Query decomposer (no model needed - rule-based)
        self.decomposer = QueryDecomposer()

        # 2. Vector searcher (loads FAISS index + metadata)
        self.searcher = VectorSearcher(
            index_path=self._index_dir / "flat.index",
            metadata_path=self._index_dir / "metadata.json",
            image_paths_path=self._index_dir / "image_paths.json",
        )

        # 3. Hybrid re-ranker
        self.reranker = HybridReranker(alpha=0.4, beta=0.3, gamma=0.3)

        # 4. Load the embedding model (shared for text encoding + reranking)
        self._model = None
        self._tokenizer = None
        self._preprocess = None
        self._load_model()

        logger.info("FashionRetriever ready.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Marqo-FashionSigLIP model for text encoding."""
        import open_clip
        import torch

        logger.info(f"Loading model: {MODEL_NAME}")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME
        )
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        self._model = self._model.to(DEVICE)
        if DEVICE == "cuda":
            self._model = self._model.half()
        self._model.eval()
        logger.info(f"Model loaded on {DEVICE}")

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string into a normalized embedding."""
        import torch

        tokens = self._tokenizer([text])
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(DEVICE)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().float().numpy().squeeze(0)

    # ------------------------------------------------------------------
    # Full pipeline (our method)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
        """
        Full hybrid retrieval pipeline.

        Steps:
            1. Decompose query into structured attributes + sub-phrases
            2. Encode full query via FashionSigLIP
            3. FAISS search for top-N candidates
            4. Hybrid re-rank using attribute matching + sub-phrase bottleneck
            5. Return top-k

        Args:
            query: Natural language search query.
            top_k: Number of final results.

        Returns:
            List of result dicts sorted by final_score.
        """
        # Step 1: Decompose
        decomposition = self.decomposer.decompose(query)
        sub_phrases = decomposition.get("sub_phrases", [query])
        logger.info(f"Query: '{query}' → sub-phrases: {sub_phrases}")

        # Step 2: Encode full query
        query_embedding = self._encode_text(query)

        # Step 3: FAISS search
        candidates = self.searcher.search_with_metadata(
            query_embedding, top_k=TOP_K_RETRIEVAL
        )

        if not candidates:
            logger.warning("No candidates found in FAISS search.")
            return []

        # Step 4: Hybrid re-rank
        all_embeddings = self.searcher.embeddings
        if all_embeddings is None:
            # Fall back: skip sub-phrase scoring
            logger.warning("Embeddings not loaded; skipping sub-phrase reranking.")
            for c in candidates:
                c["final_score"] = c["similarity"]
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
        else:
            candidates = self.reranker.rerank(
                candidates=candidates,
                query_attrs=decomposition,
                all_embeddings=all_embeddings,
                model=self._model,
                tokenizer=self._tokenizer,
                sub_phrases=sub_phrases,
            )

        # Step 5: Return top-k
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Vanilla CLIP baseline (for comparison)
    # ------------------------------------------------------------------

    def search_vanilla_clip(
        self, query: str, top_k: int = TOP_K_FINAL
    ) -> list[dict]:
        """
        Baseline: encode full query → FAISS → return top-k.
        No decomposition, no re-ranking.

        Args:
            query: Natural language search query.
            top_k: Number of results.

        Returns:
            List of result dicts sorted by cosine similarity.
        """
        query_embedding = self._encode_text(query)
        results = self.searcher.search_with_metadata(query_embedding, top_k=top_k)
        for r in results:
            r["final_score"] = r["similarity"]
        return results


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    """Run a demo search from the command line."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Fashion Image Retrieval")
    parser.add_argument(
        "--query",
        type=str,
        default="A person in a bright yellow raincoat",
        help="Natural language search query",
    )
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = FashionRetriever()

    print(f"\n{'='*60}")
    print(f"  Query: {args.query}")
    print(f"{'='*60}\n")

    print("--- Our Hybrid Pipeline ---")
    results = retriever.search(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(
            f"  {i}. score={r['final_score']:.4f}  "
            f"| {meta.get('clothing_type','?')} "
            f"| {meta.get('clothing_color','?')} "
            f"| {meta.get('environment','?')} "
            f"| {meta.get('style','?')} "
            f"| {Path(r.get('image_path','')).name}"
        )

    print("\n--- Vanilla CLIP Baseline ---")
    baseline = retriever.search_vanilla_clip(args.query, top_k=args.top_k)
    for i, r in enumerate(baseline, 1):
        meta = r.get("metadata", {})
        print(
            f"  {i}. score={r['similarity']:.4f}  "
            f"| {meta.get('clothing_type','?')} "
            f"| {meta.get('clothing_color','?')} "
            f"| {meta.get('environment','?')} "
            f"| {meta.get('style','?')} "
            f"| {Path(r.get('image_path','')).name}"
        )


if __name__ == "__main__":
    main()
