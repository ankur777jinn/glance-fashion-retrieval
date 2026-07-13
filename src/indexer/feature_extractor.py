"""Feature extraction using Marqo/marqo-fashionSigLIP via open_clip.

This module wraps the SigLIP model behind a clean interface that handles
batching, device placement, and FP16 inference to fit within 4 GB VRAM.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from src.config import BATCH_SIZE, DEVICE, EMBEDDING_DIM, MODEL_NAME

logger = logging.getLogger(__name__)


class FashionFeatureExtractor:
    """Load fashionSigLIP once and expose helpers for image / text encoding."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """Initialise the model, tokenizer, and image transform.

        Args:
            model_name: An ``open_clip`` model identifier
                        (e.g. ``"hf-hub:Marqo/marqo-fashionSigLIP"``).
            device:     ``"cuda"`` or ``"cpu"``.
            batch_size: Number of images per forward pass.
        """
        self.device = device
        self.batch_size = batch_size

        logger.info("Loading model %s on %s …", model_name, device)
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(model_name)
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Move to device & use float16 on CUDA for memory efficiency
        self.model = self.model.to(device)
        if device == "cuda":
            self.model = self.model.half()
        self.model.eval()

        logger.info(
            "Model loaded - embedding dim = %d, dtype = %s",
            EMBEDDING_DIM,
            next(self.model.parameters()).dtype,
        )

    # ── Image encoding ───────────────────────────────────────────────────
    def _load_and_preprocess(self, image_path: Path) -> torch.Tensor:
        """Open an image and apply the validation pre-processing pipeline.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            A pre-processed tensor ready for the model.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
            ValueError:        If the file cannot be decoded as an image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot decode image {path}: {exc}") from exc
        return self.preprocess_val(image)  # type: ignore[operator]

    @torch.no_grad()
    def extract_image_embeddings(
        self,
        image_paths: list[Path],
    ) -> np.ndarray:
        """Batch-encode images and return L2-normalised embeddings.

        Args:
            image_paths: List of paths to image files.

        Returns:
            ``np.ndarray`` of shape ``(N, EMBEDDING_DIM)`` with unit-norm
            rows (float32).
        """
        all_embeddings: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="Extracting image embeddings",
            unit="batch",
        ):
            batch_paths = image_paths[start : start + self.batch_size]
            tensors: list[torch.Tensor] = []
            for p in batch_paths:
                try:
                    tensors.append(self._load_and_preprocess(p))
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning("Skipping %s - %s", p, exc)
                    # Insert a zero tensor so indices stay aligned
                    tensors.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(tensors).to(self.device)
            if self.device == "cuda":
                batch_tensor = batch_tensor.half()

            features = self.model.encode_image(batch_tensor)
            features = torch.nn.functional.normalize(features, dim=-1)
            all_embeddings.append(features.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(
            "Extracted embeddings for %d images - shape %s",
            len(image_paths),
            embeddings.shape,
        )
        return embeddings

    # ── Text encoding ────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """Encode a single text query into a normalised embedding.

        Args:
            text: The query string.

        Returns:
            ``np.ndarray`` of shape ``(EMBEDDING_DIM,)`` (float32).
        """
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.float().cpu().numpy().squeeze(0)

    @torch.no_grad()
    def extract_text_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode multiple text strings into normalised embeddings.

        Args:
            texts: List of query / label strings.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), EMBEDDING_DIM)`` (float32).
        """
        all_embeddings: list[np.ndarray] = []

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Extracting text embeddings",
            unit="batch",
            disable=len(texts) <= self.batch_size,
        ):
            batch_texts = texts[start : start + self.batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            features = self.model.encode_text(tokens)
            features = torch.nn.functional.normalize(features, dim=-1)
            all_embeddings.append(features.float().cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
