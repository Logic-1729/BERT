from __future__ import annotations

import argparse
import os

import shap
from transformers import pipeline

from src.utils.common import ensure_dir, load_yaml
from src.utils.model_loader import resolve_model_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--text", required=True, help="Text to explain")
    ap.add_argument("--out_dir", default="assets/shap", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    _ = load_yaml(args.config)

    ensure_dir(args.out_dir)

    model_path = resolve_model_path(args.ckpt)

    # Use Hugging Face pipeline for SHAP integration
    clf = pipeline(
        task="text-classification",
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device=-1,
    )

    # Create SHAP explainer for the pipeline
    explainer = shap.Explainer(clf)
    shap_values = explainer([args.text])

    # Save SHAP visualization
    html_path = os.path.join(args.out_dir, "shap_explanation.html")
    html = shap.plots.text(shap_values, display=False)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved SHAP explanation to: {html_path}")
    print("Tip: Open the HTML in a browser to see red/blue token highlights.")
    print("Take a screenshot to include in your report for model interpretability analysis.")


if __name__ == "__main__":
    main()
