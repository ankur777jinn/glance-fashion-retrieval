"""End-to-end indexing pipeline for the Glance Fashion Retrieval project.

Discovers images → extracts embeddings → tags attributes → builds FAISS
indices → persists everything to ``INDEX_DIR``.

Run from the project root:

    python -m src.indexer.run_indexer
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.config import IMAGE_DIR, INDEX_DIR
from src.indexer.attribute_tagger import AttributeTagger
from src.indexer.feature_extractor import FashionFeatureExtractor
from src.indexer.index_builder import FashionIndexBuilder

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def discover_images(image_dir: Path) -> list[Path]:
    """Recursively find all image files under *image_dir*.

    Args:
        image_dir: Root directory containing fashion images.

    Returns:
        A sorted list of absolute image paths.

    Raises:
        FileNotFoundError: If *image_dir* does not exist.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}. "
            "Please download the dataset first."
        )

    images = sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    logger.info("Discovered %d images in %s", len(images), image_dir)
    return images


def run_pipeline(
    image_dir: Path = IMAGE_DIR,
    index_dir: Path = INDEX_DIR,
) -> None:
    """Execute the full indexing pipeline.

    1. Discover images.
    2. Extract image embeddings with fashionSigLIP.
    3. Tag every image (clothing type, colour, environment, style).
    4. Build flat & IVF FAISS indices.
    5. Persist embeddings, metadata, and indices.

    Args:
        image_dir: Directory containing the fashion images.
        index_dir: Output directory for all artefacts.
    """
    t_start = time.perf_counter()

    # ── 0. Prepare output directory ──────────────────────────────────────
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover images ───────────────────────────────────────────────
    image_paths = discover_images(image_dir)
    if not image_paths:
        logger.error("No images found - aborting.")
        sys.exit(1)

    # ── 2. Extract embeddings ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 / 4 - Extracting image embeddings")
    logger.info("=" * 60)
    extractor = FashionFeatureExtractor()
    embeddings = extractor.extract_image_embeddings(image_paths)

    embeddings_path = index_dir / "embeddings.npy"
    np.save(str(embeddings_path), embeddings)
    logger.info("Saved embeddings → %s", embeddings_path)

    # ── 3. Attribute tagging ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 / 4 - Tagging image attributes")
    logger.info("=" * 60)
    tagger = AttributeTagger(extractor)
    tags = tagger.tag_images_batch(image_paths)

    # Build metadata: one record per image
    metadata: list[dict] = []
    for path, tag_dict in zip(image_paths, tags):
        record = {
            "image_path": str(path),
            "filename": path.name,
            **tag_dict,
        }
        metadata.append(record)

    # ── 4. Build FAISS indices ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 / 4 - Building FAISS indices")
    logger.info("=" * 60)
    builder = FashionIndexBuilder()

    flat_index = builder.build_index(embeddings.copy())
    builder.save_index(flat_index, index_dir / "flat.index")

    ivf_index = builder.build_scalable_index(embeddings.copy(), nlist=100)
    builder.save_index(ivf_index, index_dir / "ivf.index")

    # ── 5. Persist metadata ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 / 4 - Saving metadata")
    logger.info("=" * 60)
    builder.save_metadata(metadata, index_dir / "metadata.json")

    # Save image paths list (for retriever index→path mapping)
    import json
    image_paths_list = [str(p) for p in image_paths]
    with open(index_dir / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(image_paths_list, f, indent=2)
    logger.info("Saved image paths → %s", index_dir / "image_paths.json")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("-" * 60)
    logger.info("  Images indexed   : %d", len(image_paths))
    logger.info("  Embedding shape  : %s", embeddings.shape)
    logger.info("  Flat index size  : %d vectors", flat_index.ntotal)
    logger.info("  IVF  index size  : %d vectors", ivf_index.ntotal)
    logger.info("  Output directory : %s", index_dir)
    logger.info("  Total time       : %.1f s", elapsed)
    logger.info("=" * 60)


# ── Entry-point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
