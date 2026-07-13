"""Central configuration for the Glance Fashion Retrieval project.

All hyperparameters, directory paths, model identifiers, and predefined
category lists live here so that every other module imports from a single
source of truth.
"""

from __future__ import annotations

from pathlib import Path

import torch

# ── Project layout ───────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
IMAGE_DIR: Path = DATA_DIR / "images"
INDEX_DIR: Path = DATA_DIR / "index"

FAISS_INDEX_PATH: Path = INDEX_DIR / "fashion.index"
METADATA_PATH: Path = INDEX_DIR / "metadata.json"
IMAGE_FEATURES_PATH: Path = INDEX_DIR / "image_features.npy"

RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME: str = "hf-hub:Marqo/marqo-fashionSigLIP"
EMBEDDING_DIM: int = 768

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Retrieval hyper-parameters ───────────────────────────────────────────────
TOP_K_RETRIEVAL: int = 50   # number of candidates returned by FAISS
TOP_K_FINAL: int = 10       # number of results after reranking

# ── Reranker weights (alpha + beta + gamma == 1.0) ──────────────────────────
RERANKER_ALPHA: float = 0.4   # weight for vector (FAISS) similarity
RERANKER_BETA: float = 0.3    # weight for attribute string match
RERANKER_GAMMA: float = 0.3   # weight for sub-phrase CLIP bottleneck

# ── Batch processing ────────────────────────────────────────────────────────
BATCH_SIZE: int = 32        # images per forward pass (fits comfortably in 4 GB VRAM)

# ── Zero-shot classification categories ──────────────────────────────────────
CLOTHING_CATEGORIES: list[str] = [
    # Tops
    "shirt", "t-shirt", "tee", "blouse", "top", "tank top", "crop top",
    "polo", "henley", "tunic", "camisole", "bodysuit",
    # Outerwear
    "jacket", "blazer", "coat", "overcoat", "trench coat", "parka",
    "windbreaker", "raincoat", "bomber jacket", "leather jacket",
    "denim jacket", "cardigan", "hoodie", "sweater", "pullover",
    "sweatshirt", "vest", "gilet", "poncho", "cape", "shrug",
    # Bottoms
    "pants", "trousers", "jeans", "shorts", "skirt", "leggings",
    "joggers", "chinos", "cargo pants", "culottes", "palazzo pants",
    "mini skirt", "maxi skirt", "midi skirt",
    # Dresses & Jumpsuits
    "dress", "gown", "maxi dress", "midi dress", "mini dress",
    "sundress", "cocktail dress", "evening gown", "jumpsuit", "romper",
    "overalls", "dungarees",
    # Suits & Formalwear
    "suit", "tuxedo", "waistcoat",
    # Swimwear & Activewear
    "swimsuit", "bikini", "swim trunks", "wetsuit",
    "sports bra", "athletic wear", "tracksuit",
    # Accessories
    "scarf", "tie", "bow tie", "belt", "hat", "cap", "beanie",
    "gloves", "socks", "stockings", "tights",
    # Footwear
    "shoes", "sneakers", "boots", "sandals", "heels", "loafers",
    "flats", "oxfords", "mules", "espadrilles", "slippers",
    # Bags
    "bag", "handbag", "backpack", "tote", "clutch", "crossbody bag",
    "satchel", "purse",
]

COLOR_CATEGORIES: list[str] = [
    # Primary & Basics
    "red", "blue", "green", "yellow", "orange", "purple", "violet",
    "pink", "black", "white", "grey", "gray", "brown", "beige",
    "cream", "ivory", "tan", "khaki",
    # Extended Palette
    "navy", "navy blue", "royal blue", "sky blue", "baby blue",
    "cobalt", "teal", "turquoise", "cyan", "aqua",
    "emerald", "olive", "sage", "mint", "lime", "forest green",
    "burgundy", "maroon", "wine", "coral", "salmon", "peach",
    "rose", "magenta", "fuchsia", "lavender", "lilac", "plum",
    "gold", "silver", "bronze", "copper", "champagne",
    "charcoal", "slate",
    # Modifiers (matched as compound: 'bright red', 'dark blue')
    "bright", "dark", "light", "pale", "deep", "neon", "pastel",
    "muted", "vivid",
]

ENVIRONMENT_CATEGORIES: list[str] = [
    # Professional
    "office", "workplace", "boardroom", "corporate", "business meeting",
    # Outdoor
    "park", "garden", "beach", "mountain", "hiking trail", "forest",
    "city street", "street", "urban", "rooftop", "poolside", "lakeside",
    # Social
    "party", "wedding", "dinner", "restaurant", "bar", "club",
    "red carpet", "gala", "concert", "festival",
    # Casual
    "home", "café", "cafe", "coffee shop", "mall", "grocery store",
    "gym", "yoga studio", "brunch", "studio",
    # Weather / Season
    "rainy day", "sunny day", "winter", "summer", "spring", "autumn",
    "fall", "cold weather", "warm weather", "snowy",
    # Travel
    "airport", "vacation", "tropical", "resort",
    # Abstract settings
    "formal setting", "casual setting", "outdoor setting",
    "indoor setting", "professional setting",
]

STYLE_CATEGORIES: list[str] = [
    "casual", "formal", "business", "smart casual", "streetwear",
    "athleisure", "sporty", "vintage", "retro", "classic", "modern",
    "minimalist", "bohemian", "boho", "preppy", "grunge", "punk",
    "chic", "elegant", "glamorous", "edgy", "trendy", "luxury",
    "sustainable", "eco-friendly", "western", "ethnic", "traditional",
    "workwear", "loungewear", "sleepwear", "resort wear", "festival",
]

# ── Evaluation queries (from the assignment brief) ───────────────────────────
EVALUATION_QUERIES: list[dict[str, str]] = [
    {
        "id": "Q1",
        "query": "A bright yellow raincoat for a rainy day",
        "description": "Color + clothing + weather context",
    },
    {
        "id": "Q2",
        "query": "A red tie and a white shirt in a formal setting",
        "description": "Multi-item compositional with environment",
    },
    {
        "id": "Q3",
        "query": "Casual streetwear outfit with sneakers and a hoodie",
        "description": "Style + multiple clothing items",
    },
    {
        "id": "Q4",
        "query": "Elegant black evening gown for a gala",
        "description": "Style + color + clothing + event context",
    },
    {
        "id": "Q5",
        "query": "Sporty athleisure leggings and sports bra for the gym",
        "description": "Style + multiple clothing + environment",
    },
]
