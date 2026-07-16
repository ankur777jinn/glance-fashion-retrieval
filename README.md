# Multimodal Fashion & Context Retrieval

A fashion image search engine that goes beyond basic CLIP similarity - it actually understands what someone's wearing, where they are, and the overall vibe. Built for the Glance ML Internship assignment.

## The Problem

Given a natural language query like *"A red tie and a white shirt in a formal setting"*, retrieve the most relevant fashion images from a database of ~1000 images. Sounds straightforward, but vanilla CLIP treats that query as a bag of words - it can't distinguish "red tie + white shirt" from "white tie + red shirt." That's the core challenge.

## My Approach

I tried a few things before settling on what worked:

**What I considered:**
- **Vanilla CLIP** - fast but bad at compositional queries. It just averages everything together.
- **BLIP-2 captioning** - accurate but way too slow for real-time search, and needs a beefy GPU.
- **FashionCLIP/FashionSigLIP** - better embeddings for fashion, but still has the compositionality problem.

**What I built:** A hybrid pipeline that combines FashionSigLIP embeddings with structured query decomposition and attribute-level re-ranking. Basically:

1. **Offline indexing** - run every image through FashionSigLIP to get 768-d embeddings, plus zero-shot classification to tag each image with clothing type, color, environment, and style. Store embeddings in FAISS, metadata in JSON.

2. **Query decomposition** - break the user's text query into structured parts. `"A red tie and a white shirt in a formal setting"` becomes `clothing=[tie, shirt], colors=[red, white], environment=[formal setting], style=[formal]`.

3. **Hybrid retrieval** - first do a fast FAISS vector search to get top-50 candidates, then re-rank using a weighted combination of:
   - Vector similarity (how close the embeddings are)
   - Attribute matching (do the tags actually match what was asked for?)
   - Sub-phrase scoring (encode individual sub-phrases and check each one against the image)

The re-ranking is what makes this work. A vanilla CLIP search for "red tie + white shirt in formal setting" returns a red skirt in a casual setting as its top result. My pipeline returns actual formal wear because it checks the attribute tags explicitly.

## Project Structure

```
├── src/
│   ├── config.py                  # all hyperparameters and category lists
│   ├── indexer/
│   │   ├── feature_extractor.py   # FashionSigLIP embedding extraction  
│   │   ├── attribute_tagger.py    # zero-shot attribute classification
│   │   ├── index_builder.py       # FAISS index construction
│   │   └── run_indexer.py         # orchestrates the full indexing pipeline
│   ├── retriever/
│   │   ├── query_decomposer.py    # NL query → structured decomposition
│   │   ├── vector_search.py       # FAISS similarity search
│   │   ├── reranker.py            # hybrid re-ranking logic
│   │   └── run_retriever.py       # end-to-end retrieval pipeline
│   └── evaluation/
│       └── evaluate.py            # runs all 5 assignment queries
├── app.py                         # Gradio demo UI
├── build_dataset.py               # downloads Fashionpedia from HuggingFace
├── docs/report.md                 # detailed technical write-up
└── requirements.txt
```

## Setup & Usage

**Requirements:** Python 3.10+, NVIDIA GPU (tested on RTX 3050 4GB - works fine)

```bash
# install dependencies
pip install -r requirements.txt

# download the dataset (~1000 images from Fashionpedia)
python build_dataset.py

# build the index (takes ~60s on RTX 3050)
python -m src.indexer.run_indexer

# run evaluation on all 5 assignment queries
python -m src.evaluation.evaluate

# launch the demo UI
python app.py
```

The model weights (~400MB) download automatically from HuggingFace on first run.

## Results

Tested on the 5 query types from the assignment. Here's what the hybrid pipeline does compared to vanilla CLIP:

| Query | Vanilla CLIP Top-1 | Hybrid Pipeline Top-1 |
|:---|:---|:---|
| "bright yellow raincoat for a rainy day" | raincoat, yellow, rainy day | raincoat, yellow, rainy day |
| "red tie + white shirt in formal setting" | skirt, red, casual setting | bow tie, navy, formal setting |
| "casual streetwear with sneakers and hoodie" | hoodie, baby blue, streetwear | hoodie, baby blue, streetwear |
| "elegant black evening gown for a gala" | evening gown, black, formal | evening gown, black, gala |
| "sporty athleisure leggings for the gym" | athletic wear, mint, outdoor | athletic wear, light, gym |

The hybrid pipeline wins on compositional and contextual queries (Q2, Q5) where vanilla CLIP gets distracted by individual word matches. On simpler queries (Q1, Q3) both perform similarly since there's less ambiguity.

Full evaluation results are saved to `data/evaluation_results.json` after running the evaluation script.

## Key Technical Decisions

- **Model choice:** Marqo-FashionSigLIP (ViT-B-16) - it's specifically fine-tuned on fashion data using generalized contrastive learning. Significantly better than vanilla CLIP at distinguishing between, say, a "blazer" and a "cardigan."

- **Query decomposition:** Rule-based with gazetteers rather than LLM-based. Faster, deterministic, no API costs. The gazetteers cover ~100 clothing types, ~60 colors, ~55 environments, and ~30 style terms.

- **Re-ranking weights:** `α=0.4` (vector sim) + `β=0.3` (attribute match) + `γ=0.3` (sub-phrase). These were tuned by hand on the assignment queries. The attribute signal matters a lot for compositional queries.

- **Dataset:** 1000 images from Fashionpedia (streamed from HuggingFace). The assignment explicitly suggests this dataset.

## What I'd Improve With More Time

- Use a lightweight LLM (like Phi-3) for query decomposition instead of rule-based - would handle edge cases better
- Contrastive fine-tuning with hard negatives (NegCLIP-style) to improve the embedding space
- Cross-encoder re-ranking for the final top-k
- Quantized FAISS indices (PQ/OPQ) for scaling to millions of images

## Tech Stack

- **Embeddings:** Marqo-FashionSigLIP (via open_clip)
- **Vector search:** FAISS (IndexFlatIP + IndexIVFFlat)
- **UI:** Gradio
- **Dataset:** Fashionpedia (HuggingFace datasets)
- **Hardware:** Tested on RTX 3050 Laptop (4GB VRAM), 16GB RAM
