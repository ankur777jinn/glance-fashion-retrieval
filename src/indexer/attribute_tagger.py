"""Zero-shot attribute tagging via fashionSigLIP.

Each image is classified along four axes - clothing type, colour,
environment, and style - by computing cosine similarity between
the image embedding and a set of pre-encoded text prompt embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.config import (
    CLOTHING_CATEGORIES,
    COLOR_CATEGORIES,
    ENVIRONMENT_CATEGORIES,
    STYLE_CATEGORIES,
)
from src.indexer.feature_extractor import FashionFeatureExtractor

logger = logging.getLogger(__name__)

# ── Prompt templates (one per axis) ──────────────────────────────────────────
_PROMPT_TEMPLATES: dict[str, str] = {
    "clothing_type": "a photo of a person wearing a {}",
    "clothing_color": "a photo in {} color",
    "environment": "a photo taken in a {}",
    "style": "a {} style outfit",
}

# Map axis name → category list
_AXIS_CATEGORIES: dict[str, list[str]] = {
    "clothing_type": CLOTHING_CATEGORIES,
    "clothing_color": COLOR_CATEGORIES,
    "environment": ENVIRONMENT_CATEGORIES,
    "style": STYLE_CATEGORIES,
}


class AttributeTagger:
    """Assigns clothing-type, colour, environment, and style tags to images.

    The tagger reuses a :class:`FashionFeatureExtractor` for both image and
    text encoding.  Text prompt embeddings for every category label are
    computed once at initialisation and cached for the lifetime of the
    object.
    """

    def __init__(self, extractor: FashionFeatureExtractor) -> None:
        """Build the tagger and pre-encode all category label embeddings.

        Args:
            extractor: A pre-loaded :class:`FashionFeatureExtractor` instance.
        """
        self.extractor = extractor

        # Pre-encode text prompts for each axis and cache them
        # {axis_name: np.ndarray of shape (n_labels, embed_dim)}
        self._label_embeddings: dict[str, np.ndarray] = {}
        self._label_names: dict[str, list[str]] = {}

        logger.info("Pre-encoding category label embeddings …")
        for axis, categories in _AXIS_CATEGORIES.items():
            template = _PROMPT_TEMPLATES[axis]
            prompts = [template.format(cat) for cat in categories]
            embeddings = self.extractor.extract_text_embeddings(prompts)
            self._label_embeddings[axis] = embeddings  # (n_labels, D)
            self._label_names[axis] = list(categories)
            logger.info(
                "  %s: %d labels encoded", axis, len(categories)
            )

    # ── Single-image tagging ─────────────────────────────────────────────
    def tag_image(self, image_path: Path) -> dict[str, Any]:
        """Tag a single image across all four axes.

        Args:
            image_path: Path to the image file.

        Returns:
            A dict with keys ``clothing_type``, ``clothing_color``,
            ``environment``, ``style``, and ``confidence_scores`` (a
            nested dict mapping each axis to its winning softmax score).
        """
        image_embedding = self.extractor.extract_image_embeddings(
            [image_path]
        )  # (1, D)

        result: dict[str, Any] = {}
        confidence_scores: dict[str, float] = {}

        for axis in _AXIS_CATEGORIES:
            label_embs = self._label_embeddings[axis]  # (n_labels, D)
            similarities = (image_embedding @ label_embs.T).squeeze(0)  # (n_labels,)

            # Softmax to get a proper probability distribution
            exp_sim = np.exp(similarities - similarities.max())
            probs = exp_sim / exp_sim.sum()

            best_idx = int(np.argmax(probs))
            result[axis] = self._label_names[axis][best_idx]
            confidence_scores[axis] = float(probs[best_idx])

        result["confidence_scores"] = confidence_scores
        return result

    # ── Batch tagging ────────────────────────────────────────────────────
    def tag_images_batch(
        self,
        image_paths: list[Path],
    ) -> list[dict[str, Any]]:
        """Tag a batch of images (with tqdm progress).

        This method first extracts all image embeddings in one vectorised
        pass and then classifies each image against the cached label
        embeddings - much faster than calling :meth:`tag_image` in a loop.

        Args:
            image_paths: Paths to the image files.

        Returns:
            A list of tag dicts, one per image, in the same order as
            *image_paths*.
        """
        logger.info("Tagging %d images …", len(image_paths))

        # 1. Extract all image embeddings at once
        image_embeddings = self.extractor.extract_image_embeddings(
            image_paths
        )  # (N, D)

        # 2. Classify each image against cached label embeddings
        results: list[dict[str, Any]] = []

        for idx in tqdm(
            range(len(image_paths)),
            desc="Assigning attribute tags",
            unit="img",
        ):
            emb = image_embeddings[idx : idx + 1]  # (1, D)
            record: dict[str, Any] = {}
            confidence_scores: dict[str, float] = {}

            for axis in _AXIS_CATEGORIES:
                label_embs = self._label_embeddings[axis]
                similarities = (emb @ label_embs.T).squeeze(0)

                exp_sim = np.exp(similarities - similarities.max())
                probs = exp_sim / exp_sim.sum()

                best_idx = int(np.argmax(probs))
                record[axis] = self._label_names[axis][best_idx]
                confidence_scores[axis] = float(probs[best_idx])

            record["confidence_scores"] = confidence_scores
            results.append(record)

        logger.info("Tagging complete for %d images.", len(results))
        return results
