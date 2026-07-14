"""
Hybrid re-ranker that combines vector similarity, attribute matching,
and sub-phrase bottleneck scoring to solve compositionality.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HybridReranker:
    """
    Re-ranks FAISS candidates using three scoring signals:
      1. vector_score  - original cosine similarity from FAISS
      2. attribute_score - structured metadata matching
      3. subphrase_score - per-sub-phrase cosine sim (min = bottleneck)
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
    ):
        """
        Args:
            alpha: Weight for vector similarity score.
            beta:  Weight for attribute match score.
            gamma: Weight for sub-phrase bottleneck score.
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidates: list[dict],
        query_attrs: dict,
        all_embeddings: np.ndarray,
        model,
        tokenizer,
        sub_phrases: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Re-rank candidates using hybrid scoring.

        Args:
            candidates: List of dicts from VectorSearcher.search_with_metadata.
            query_attrs: Decomposed query attributes from QueryDecomposer.
            all_embeddings: Full embedding matrix (N, D) for all indexed images.
            model: open_clip model (for encoding sub-phrases).
            tokenizer: open_clip tokenizer.
            sub_phrases: List of sub-phrase strings for bottleneck scoring.

        Returns:
            Re-ranked list of candidate dicts with added 'final_score'.
        """
        if not candidates:
            return []

        # Pre-compute sub-phrase embeddings
        subphrase_embeddings = None
        if sub_phrases and len(sub_phrases) > 0:
            subphrase_embeddings = self._encode_sub_phrases(
                sub_phrases, model, tokenizer
            )

        for candidate in candidates:
            idx = candidate["index"]
            vector_score = candidate.get("similarity", 0.0)

            # Signal 2: attribute match
            attr_score = self.compute_attribute_match_score(
                candidate.get("metadata", {}), query_attrs
            )

            # Signal 3: sub-phrase bottleneck
            sp_score = 0.0
            if subphrase_embeddings is not None and idx < len(all_embeddings):
                sp_score = self._compute_subphrase_score(
                    all_embeddings[idx], subphrase_embeddings
                )

            # Weighted fusion
            final = (
                self.alpha * vector_score
                + self.beta * attr_score
                + self.gamma * sp_score
            )
            candidate["final_score"] = float(final)
            candidate["score_breakdown"] = {
                "vector": float(vector_score),
                "attribute": float(attr_score),
                "subphrase": float(sp_score),
            }

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Attribute matching
    # ------------------------------------------------------------------

    @staticmethod
    def compute_attribute_match_score(
        image_meta: dict, query_attrs: dict
    ) -> float:
        """
        Score how well image metadata matches decomposed query attributes.

        Checks clothing_terms, color_terms, environment_terms, style_terms
        against the image's pre-extracted tags.

        Returns:
            Float in [0, 1].
        """
        if not image_meta or not query_attrs:
            return 0.0

        matches = 0
        total = 0

        # Clothing match
        clothing_terms = query_attrs.get("clothing_terms", [])
        if clothing_terms:
            total += 1
            img_clothing = image_meta.get("clothing_type", "").lower()
            if any(t.lower() in img_clothing for t in clothing_terms):
                matches += 1

        # Color match
        color_terms = query_attrs.get("color_terms", [])
        if color_terms:
            total += 1
            img_color = image_meta.get("clothing_color", "").lower()
            if any(c.lower() in img_color for c in color_terms):
                matches += 1

        # Environment match
        env_terms = query_attrs.get("environment_terms", [])
        if env_terms:
            total += 1
            img_env = image_meta.get("environment", "").lower()
            if any(e.lower() in img_env for e in env_terms):
                matches += 1

        # Style match
        style_terms = query_attrs.get("style_terms", [])
        if style_terms:
            total += 1
            img_style = image_meta.get("style", "").lower()
            if any(s.lower() in img_style for s in style_terms):
                matches += 1

        return matches / max(total, 1)

    # ------------------------------------------------------------------
    # Sub-phrase bottleneck scoring
    # ------------------------------------------------------------------

    def _encode_sub_phrases(
        self, sub_phrases: list[str], model, tokenizer
    ) -> np.ndarray:
        """Encode sub-phrases into normalized embeddings."""
        tokens = tokenizer(sub_phrases)
        device = next(model.parameters()).device
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(device)
        else:
            tokens = {k: v.to(device) for k, v in tokens.items()} if isinstance(tokens, dict) else tokens.to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().float().numpy()

    @staticmethod
    def _compute_subphrase_score(
        image_embedding: np.ndarray,
        subphrase_embeddings: np.ndarray,
    ) -> float:
        """
        Compute sub-phrase bottleneck score.

        For each sub-phrase, compute cosine similarity with the image.
        Return the MINIMUM - ensuring ALL attributes must be satisfied.
        Blend with the mean for robustness.

        Returns:
            Float - blended min/mean similarity.
        """
        img = image_embedding.astype(np.float32)
        img_norm = img / (np.linalg.norm(img) + 1e-8)

        sims = subphrase_embeddings @ img_norm  # (num_phrases,)
        min_sim = float(sims.min())
        avg_sim = float(sims.mean())

        # Blend: 60% bottleneck + 40% average
        return 0.6 * min_sim + 0.4 * avg_sim
