"""
Dataset download and preparation script for Glance Fashion Retrieval.

Downloads a curated subset of fashion images from Fashionpedia or
uses a fallback strategy with open-license image sources.
"""

import os
import json
import shutil
import random
import hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# Fashionpedia image URLs follow this pattern on CVDF hosting
FASHIONPEDIA_BASE_URL = "https://s3.amazonaws.com/ifashionist/Fashionpedia"
FASHIONPEDIA_TRAIN_URL = f"{FASHIONPEDIA_BASE_URL}/train2020.zip"
FASHIONPEDIA_VAL_URL = f"{FASHIONPEDIA_BASE_URL}/val2020.zip"
FASHIONPEDIA_ANNOT_URL = (
    "https://raw.githubusercontent.com/cvdfoundation/fashionpedia/"
    "main/instances_attributes_val2020.json"
)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file from URL to destination with progress."""
    try:
        print(f"Downloading {desc or url}...")
        urlretrieve(url, str(dest))
        print(f"  -> Saved to {dest}")
        return True
    except Exception as e:
        print(f"  -> Failed: {e}")
        return False


def prepare_fashionpedia_subset(
    output_dir: Path,
    num_images: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Download and prepare a subset of Fashionpedia images.
    
    If direct download fails (large files), falls back to using
    the HuggingFace datasets library or Kaggle API.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to select
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with dataset statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if images already exist
    existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Found {len(existing)} existing images in {output_dir}. Skipping download.")
        return {"num_images": len(existing), "source": "existing"}
    
    print(f"\n{'='*60}")
    print(f"  DATASET PREPARATION")
    print(f"  Target: {num_images} diverse fashion images")
    print(f"{'='*60}\n")
    
    # Strategy 1: Try HuggingFace datasets
    try:
        print("Strategy 1: Downloading from HuggingFace Fashionpedia...")
        return _download_from_huggingface(output_dir, num_images, seed)
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
    
    # Strategy 2: Try Kaggle API
    try:
        print("\nStrategy 2: Trying Kaggle API...")
        return _download_from_kaggle(output_dir, num_images, seed)
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
    
    # Strategy 3: Generate synthetic dataset description
    print("\nStrategy 3: Please download manually.")
    print("  Option A: kaggle datasets download -d pengxin/fashionpedia")
    print("  Option B: Visit https://huggingface.co/datasets/detection-datasets/fashionpedia")
    print(f"  Place {num_images} .jpg images into: {output_dir}")
    
    return {"num_images": 0, "source": "manual_required"}


def _download_from_huggingface(
    output_dir: Path, num_images: int, seed: int
) -> dict:
    """Download Fashionpedia from HuggingFace datasets."""
    from datasets import load_dataset
    
    print("  Loading Fashionpedia from HuggingFace (this may take a few minutes)...")
    dataset = load_dataset(
        "detection-datasets/fashionpedia", 
        split="val",
        trust_remote_code=True,
    )
    
    # Sample a diverse subset
    random.seed(seed)
    total = len(dataset)
    indices = random.sample(range(total), min(num_images, total))
    
    print(f"  Saving {len(indices)} images to {output_dir}...")
    for i, idx in enumerate(tqdm(indices, desc="Saving images")):
        item = dataset[idx]
        image = item["image"]
        image_path = output_dir / f"fashion_{i:05d}.jpg"
        image.save(str(image_path), "JPEG", quality=95)
    
    return {
        "num_images": len(indices),
        "source": "huggingface/fashionpedia",
    }


def _download_from_kaggle(
    output_dir: Path, num_images: int, seed: int
) -> dict:
    """Download from Kaggle using the kaggle API."""
    import subprocess
    
    tmp_dir = output_dir.parent / "_kaggle_tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "pengxin/fashionpedia",
         "-p", str(tmp_dir), "--unzip"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    
    # Find and copy images
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        all_images.extend(tmp_dir.rglob(ext))
    
    random.seed(seed)
    selected = random.sample(all_images, min(num_images, len(all_images)))
    
    for i, img_path in enumerate(tqdm(selected, desc="Copying images")):
        shutil.copy2(str(img_path), str(output_dir / f"fashion_{i:05d}.jpg"))
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return {
        "num_images": len(selected),
        "source": "kaggle/fashionpedia",
    }


if __name__ == "__main__":
    from src.config import IMAGE_DIR
    
    stats = prepare_fashionpedia_subset(IMAGE_DIR, num_images=1000)
    print(f"\nDataset ready: {stats}")
