"""
Gradio-based interactive demo for the Multimodal Fashion Retrieval system.

Launch with: python app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from PIL import Image


def create_demo():
    """Create and return the Gradio demo interface."""
    
    # Lazy import to avoid loading model at module level
    from src.retriever.run_retriever import FashionRetriever
    
    print("Loading Fashion Retrieval System...")
    retriever = FashionRetriever()
    print("System ready!")
    
    def search_fashion(query: str, top_k: int, method: str):
        """
        Search for fashion images matching the query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            method: 'Our Pipeline (Hybrid)' or 'Vanilla CLIP (Baseline)'
            
        Returns:
            Tuple of (gallery_images, results_text)
        """
        if not query.strip():
            return [], "Please enter a search query."
        
        try:
            if method == "Vanilla CLIP (Baseline)":
                results = retriever.search_vanilla_clip(query, top_k=top_k)
            else:
                results = retriever.search(query, top_k=top_k)
            
            if not results:
                return [], "No results found."
            
            # Prepare gallery images with captions
            gallery_items = []
            results_lines = [f"**Query:** {query}\n", f"**Method:** {method}\n"]
            
            for i, result in enumerate(results):
                img_path = result.get("image_path", "")
                score = result.get("final_score", result.get("similarity", 0))
                metadata = result.get("metadata", {})
                
                if img_path and Path(img_path).exists():
                    img = Image.open(img_path)
                    caption = (
                        f"#{i+1} | Score: {score:.3f}\n"
                        f"{metadata.get('clothing_type', 'N/A')} | "
                        f"{metadata.get('clothing_color', 'N/A')} | "
                        f"{metadata.get('environment', 'N/A')} | "
                        f"{metadata.get('style', 'N/A')}"
                    )
                    gallery_items.append((img, caption))
                    
                    results_lines.append(
                        f"**Result {i+1}:** Score={score:.4f} | "
                        f"Clothing={metadata.get('clothing_type', '?')}, "
                        f"Color={metadata.get('clothing_color', '?')}, "
                        f"Env={metadata.get('environment', '?')}, "
                        f"Style={metadata.get('style', '?')}"
                    )
            
            return gallery_items, "\n".join(results_lines)
            
        except Exception as e:
            return [], f"Error: {str(e)}"
    
    def compare_methods(query: str, top_k: int):
        """Run both methods and return side-by-side results."""
        if not query.strip():
            return [], [], "Please enter a search query."
        
        try:
            # Our pipeline
            our_results = retriever.search(query, top_k=top_k)
            our_gallery = []
            for i, r in enumerate(our_results):
                img_path = r.get("image_path", "")
                if img_path and Path(img_path).exists():
                    score = r.get("final_score", 0)
                    meta = r.get("metadata", {})
                    caption = (
                        f"#{i+1} | {score:.3f} | "
                        f"{meta.get('clothing_type', '')} "
                        f"{meta.get('clothing_color', '')}"
                    )
                    our_gallery.append((Image.open(img_path), caption))
            
            # Vanilla CLIP baseline
            baseline_results = retriever.search_vanilla_clip(query, top_k=top_k)
            baseline_gallery = []
            for i, r in enumerate(baseline_results):
                img_path = r.get("image_path", "")
                if img_path and Path(img_path).exists():
                    score = r.get("similarity", 0)
                    meta = r.get("metadata", {})
                    caption = (
                        f"#{i+1} | {score:.3f} | "
                        f"{meta.get('clothing_type', '')} "
                        f"{meta.get('clothing_color', '')}"
                    )
                    baseline_gallery.append((Image.open(img_path), caption))
            
            # Decomposition info
            decomp = retriever.decomposer.decompose(query)
            decomp_text = (
                f"**Query Decomposition:**\n"
                f"- Clothing: {decomp.get('clothing_terms', [])}\n"
                f"- Colors: {decomp.get('color_terms', [])}\n"
                f"- Environment: {decomp.get('environment_terms', [])}\n"
                f"- Style: {decomp.get('style_terms', [])}\n"
                f"- Sub-phrases: {decomp.get('sub_phrases', [])}"
            )
            
            return our_gallery, baseline_gallery, decomp_text
            
        except Exception as e:
            return [], [], f"Error: {str(e)}"
    
    # Build the Gradio UI
    with gr.Blocks(
        title="Multimodal Fashion Retrieval",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="pink",
        ),
        css="""
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: 800;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1.1em;
            margin-top: 0;
        }
        """
    ) as demo:
        
        gr.HTML("""
            <h1 class="main-title">🔍 Multimodal Fashion Retrieval</h1>
            <p class="subtitle">
                Search fashion images by describing clothing, color, environment, and style
            </p>
        """)
        
        with gr.Tabs():
            # Tab 1: Single Search
            with gr.TabItem("🔎 Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Natural Language Query",
                            placeholder='Try: "A person in a bright yellow raincoat" or "Casual weekend outfit for a city walk"',
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        top_k_slider = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Number of Results (k)"
                        )
                        method_dropdown = gr.Dropdown(
                            choices=[
                                "Our Pipeline (Hybrid)",
                                "Vanilla CLIP (Baseline)",
                            ],
                            value="Our Pipeline (Hybrid)",
                            label="Retrieval Method",
                        )
                        search_btn = gr.Button("🔍 Search", variant="primary")
                
                gallery_output = gr.Gallery(
                    label="Retrieved Images",
                    columns=5,
                    height="auto",
                    object_fit="cover",
                )
                details_output = gr.Markdown(label="Result Details")
                
                search_btn.click(
                    fn=search_fashion,
                    inputs=[query_input, top_k_slider, method_dropdown],
                    outputs=[gallery_output, details_output],
                )
            
            # Tab 2: Comparison Mode
            with gr.TabItem("⚔️ Compare Methods"):
                gr.Markdown(
                    "### Our Hybrid Pipeline vs. Vanilla CLIP Baseline\n"
                    "See how structured query decomposition + attribute re-ranking "
                    "improves over naive CLIP similarity."
                )
                with gr.Row():
                    compare_query = gr.Textbox(
                        label="Query",
                        placeholder='Try compositional: "A red tie and a white shirt in a formal setting"',
                        lines=2,
                        scale=3,
                    )
                    compare_k = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Top-K", scale=1,
                    )
                    compare_btn = gr.Button("⚔️ Compare", variant="primary", scale=1)
                
                decomp_output = gr.Markdown(label="Query Analysis")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ✅ Our Pipeline (Hybrid Retrieval)")
                        our_gallery = gr.Gallery(
                            label="Our Results",
                            columns=3,
                            height="auto",
                            object_fit="cover",
                        )
                    with gr.Column():
                        gr.Markdown("### 📊 Vanilla CLIP (Baseline)")
                        baseline_gallery = gr.Gallery(
                            label="Baseline Results",
                            columns=3,
                            height="auto",
                            object_fit="cover",
                        )
                
                compare_btn.click(
                    fn=compare_methods,
                    inputs=[compare_query, compare_k],
                    outputs=[our_gallery, baseline_gallery, decomp_output],
                )
            
            # Tab 3: Evaluation Queries
            with gr.TabItem("📋 Evaluation"):
                gr.Markdown(
                    "### Run All 5 Evaluation Queries\n"
                    "These are the exact queries from the assignment specification."
                )
                eval_btn = gr.Button("🚀 Run All Evaluation Queries", variant="primary")
                eval_output = gr.Markdown()
                
                def run_all_evaluations():
                    from src.config import EVALUATION_QUERIES
                    lines = ["# Evaluation Results\n"]
                    for i, query in enumerate(EVALUATION_QUERIES, 1):
                        results = retriever.search(query, top_k=5)
                        baseline = retriever.search_vanilla_clip(query, top_k=5)
                        
                        lines.append(f"## Query {i}: \"{query}\"\n")
                        lines.append("**Our Pipeline:**")
                        for j, r in enumerate(results):
                            meta = r.get("metadata", {})
                            score = r.get("final_score", 0)
                            lines.append(
                                f"  {j+1}. Score={score:.4f} | "
                                f"{meta.get('clothing_type','?')}, "
                                f"{meta.get('clothing_color','?')}, "
                                f"{meta.get('environment','?')}, "
                                f"{meta.get('style','?')}"
                            )
                        
                        lines.append("\n**Vanilla CLIP Baseline:**")
                        for j, r in enumerate(baseline):
                            meta = r.get("metadata", {})
                            score = r.get("similarity", 0)
                            lines.append(
                                f"  {j+1}. Score={score:.4f} | "
                                f"{meta.get('clothing_type','?')}, "
                                f"{meta.get('clothing_color','?')}, "
                                f"{meta.get('environment','?')}, "
                                f"{meta.get('style','?')}"
                            )
                        lines.append("\n---\n")
                    
                    return "\n".join(lines)
                
                eval_btn.click(fn=run_all_evaluations, outputs=eval_output)
        
        gr.Markdown(
            "---\n"
            "*Built for the Glance ML Internship Assignment - "
            "Multimodal Fashion & Context Retrieval*\n\n"
            "**Architecture:** Marqo-FashionSigLIP + FAISS + "
            "Structured Query Decomposition + Hybrid Re-ranking*"
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
