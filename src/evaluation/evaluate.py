"""
Evaluation script - runs all 5 assignment queries through both
our hybrid pipeline and the vanilla CLIP baseline, then saves
a comparison report.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVALUATION_QUERIES, INDEX_DIR
from src.retriever.run_retriever import FashionRetriever

logger = logging.getLogger(__name__)

QUERY_TYPES = [
    "Attribute Specific",
    "Contextual/Place",
    "Complex Semantic",
    "Style Inference",
    "Compositional",
]


def run_evaluation(top_k: int = 5) -> dict:
    """
    Run all 5 evaluation queries through both pipelines.

    Returns:
        Dictionary containing full evaluation results.
    """
    print("\n" + "=" * 70)
    print("  GLANCE ML INTERNSHIP - EVALUATION")
    print("  Running all 5 assignment queries")
    print("=" * 70 + "\n")

    retriever = FashionRetriever()
    report = {
        "timestamp": datetime.now().isoformat(),
        "top_k": top_k,
        "queries": [],
    }

    for i, (query, qtype) in enumerate(zip(EVALUATION_QUERIES, QUERY_TYPES), 1):
        print(f"\n{'─'*60}")
        print(f"  Query {i} [{qtype}]")
        print(f"  \"{query['query']}\"")
        print(f"{'─'*60}")

        # --- Our hybrid pipeline ---
        our_results = retriever.search(query['query'], top_k=top_k)

        print(f"\n  ✅ Our Hybrid Pipeline:")
        our_entries = []
        for j, r in enumerate(our_results, 1):
            meta = r.get("metadata", {})
            breakdown = r.get("score_breakdown", {})
            entry = {
                "rank": j,
                "final_score": round(r.get("final_score", 0), 4),
                "vector_score": round(breakdown.get("vector", 0), 4),
                "attribute_score": round(breakdown.get("attribute", 0), 4),
                "subphrase_score": round(breakdown.get("subphrase", 0), 4),
                "image": Path(r.get("image_path", "")).name,
                "clothing_type": meta.get("clothing_type", "?"),
                "clothing_color": meta.get("clothing_color", "?"),
                "environment": meta.get("environment", "?"),
                "style": meta.get("style", "?"),
            }
            our_entries.append(entry)
            print(
                f"     {j}. score={entry['final_score']:.4f}  "
                f"(vec={entry['vector_score']:.3f} "
                f"attr={entry['attribute_score']:.3f} "
                f"sp={entry['subphrase_score']:.3f})  "
                f"| {entry['clothing_type']} | {entry['clothing_color']} "
                f"| {entry['environment']} | {entry['style']}"
            )

        # --- Vanilla CLIP baseline ---
        baseline_results = retriever.search_vanilla_clip(query['query'], top_k=top_k)

        print(f"\n  📊 Vanilla CLIP Baseline:")
        baseline_entries = []
        for j, r in enumerate(baseline_results, 1):
            meta = r.get("metadata", {})
            entry = {
                "rank": j,
                "similarity": round(r.get("similarity", 0), 4),
                "image": Path(r.get("image_path", "")).name,
                "clothing_type": meta.get("clothing_type", "?"),
                "clothing_color": meta.get("clothing_color", "?"),
                "environment": meta.get("environment", "?"),
                "style": meta.get("style", "?"),
            }
            baseline_entries.append(entry)
            print(
                f"     {j}. sim={entry['similarity']:.4f}  "
                f"| {entry['clothing_type']} | {entry['clothing_color']} "
                f"| {entry['environment']} | {entry['style']}"
            )

        # Decomposition analysis
        decomp = retriever.decomposer.decompose(query['query'])
        print(f"\n  🔍 Query Decomposition:")
        print(f"     clothing:    {decomp.get('clothing_terms', [])}")
        print(f"     colors:      {decomp.get('color_terms', [])}")
        print(f"     environment: {decomp.get('environment_terms', [])}")
        print(f"     style:       {decomp.get('style_terms', [])}")
        print(f"     sub-phrases: {decomp.get('sub_phrases', [])}")

        report["queries"].append({
            "id": i,
            "type": qtype,
            "query": query['query'],
            "decomposition": decomp,
            "our_results": our_entries,
            "baseline_results": baseline_entries,
        })

    # Save report
    report_path = INDEX_DIR.parent / "evaluation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n\n📄 Full report saved to: {report_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Queries evaluated: {len(EVALUATION_QUERIES)}")
    print(f"  Top-K per query:   {top_k}")
    print(f"  Methods compared:  Our Hybrid Pipeline vs Vanilla CLIP")
    print(f"  Report saved to:   {report_path}")
    print()

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_evaluation(top_k=5)
