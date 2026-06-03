from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import pipeline

from src.utils.common import ensure_dir, load_yaml
from src.utils.model_loader import resolve_model_path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--text", required=True, help="Text to explain")
    ap.add_argument("--out_dir", default="assets/lime", help="Output directory")
    ap.add_argument("--num_features", type=int, default=10, help="Number of features to show")
    ap.add_argument("--num_samples", type=int, default=1000, help="LIME perturbation samples")
    return ap.parse_args()


def main():
    args = parse_args()
    _ = load_yaml(args.config)

    ensure_dir(args.out_dir)

    model_path = resolve_model_path(args.ckpt)

    clf = pipeline(
        task="text-classification",
        model=model_path,
        tokenizer=model_path,
        return_all_scores=True,
        device=-1,
    )

    # LIME requires a predict_proba-like function: List[str] -> np.ndarray (n, num_classes)
    def predict_proba(texts: List[str]) -> np.ndarray:
        outputs = clf(texts)
        # outputs: List[List[{label, score}, ...]]
        probs = []
        for per_text in outputs:
            probs.append([x["score"] for x in per_text])
        return np.array(probs, dtype=np.float64)

    explainer = LimeTextExplainer(class_names=None, split_expression=r"\s+")

    exp = explainer.explain_instance(
        args.text,
        predict_proba,
        num_features=args.num_features,
        num_samples=args.num_samples,
    )

    html_path = os.path.join(args.out_dir, "lime_explanation.html")
    exp.save_to_file(html_path)

    print(f"Saved LIME explanation to: {html_path}")
    print("Tip: Open the HTML in a browser to see feature importance for tokens.")
    print("Take a screenshot to include in your report for model interpretability comparison with SHAP.")


if __name__ == "__main__":
    main()
