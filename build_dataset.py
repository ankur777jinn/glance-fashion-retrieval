"""
Fast streaming dataset download - downloads images one at a time
without needing to cache the entire 3-5 GB dataset first.
"""
import os
import sys
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = Path(__file__).parent / "data" / "images"
TARGET = 1000


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(IMAGE_DIR.glob("*.jpg"))
    if len(existing) >= TARGET:
        print(f"Already have {len(existing)} images. Done!")
        return

    from datasets import load_dataset

    print("=" * 60)
    print("  STREAMING DOWNLOAD - Fashionpedia (1000 images)")
    print("  No full dataset cache needed. Downloads image by image.")
    print("=" * 60)
    print()

    # Use streaming=True so we don't need to download the whole thing
    ds = load_dataset(
        "detection-datasets/fashionpedia",
        split="val",
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    skipped = 0

    for item in tqdm(ds, desc="Downloading", total=TARGET):
        if saved >= TARGET:
            break
        try:
            img = item["image"]
            if img.size[0] < 200 or img.size[1] < 200:
                skipped += 1
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize for efficiency
            mx = max(img.size)
            if mx > 800:
                s = 800 / mx
                img = img.resize((int(img.size[0]*s), int(img.size[1]*s)), Image.LANCZOS)

            img.save(str(IMAGE_DIR / f"fashion_{saved:05d}.jpg"), "JPEG", quality=90)
            saved += 1
        except Exception:
            skipped += 1

    # If val split wasn't enough, grab from train
    if saved < TARGET:
        print(f"\nVal had {saved} images. Grabbing more from train split...")
        ds_train = load_dataset(
            "detection-datasets/fashionpedia",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for item in tqdm(ds_train, desc="Downloading (train)", total=TARGET - saved):
            if saved >= TARGET:
                break
            try:
                img = item["image"]
                if img.size[0] < 200 or img.size[1] < 200:
                    continue
                if img.mode != "RGB":
                    img = img.convert("RGB")
                mx = max(img.size)
                if mx > 800:
                    s = 800 / mx
                    img = img.resize((int(img.size[0]*s), int(img.size[1]*s)), Image.LANCZOS)
                img.save(str(IMAGE_DIR / f"fashion_{saved:05d}.jpg"), "JPEG", quality=90)
                saved += 1
            except Exception:
                continue

    total_mb = sum(f.stat().st_size for f in IMAGE_DIR.glob("*.jpg")) / (1024*1024)
    print(f"\n{'='*60}")
    print(f"  DONE! Saved {saved} images ({total_mb:.1f} MB)")
    print(f"  Location: {IMAGE_DIR}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*60}")
    print(f"\nNext step: python -m src.indexer.run_indexer")


if __name__ == "__main__":
    main()
