from __future__ import annotations

import argparse
import json
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


def load_label_names(config_path: str) -> list[str]:
    """Load human-readable label names from label2id.json."""
    data_dir = load_yaml(config_path)["data"]["data_dir"]
    label2id_path = os.path.join(data_dir, "label2id.json")
    if os.path.exists(label2id_path):
        with open(label2id_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        return [id2label.get(i, f"Class {i}") for i in range(len(id2label))]
    return []


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    ensure_dir(args.out_dir)

    model_path = resolve_model_path(args.ckpt)
    label_names = load_label_names(args.config)

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

    if label_names and len(label_names) == len(shap_values.output_names):
        shap_values.output_names = label_names

    # Save SHAP visualization
    html_path = os.path.join(args.out_dir, "shap_explanation.html")
    body = shap.plots.text(shap_values, display=False)

    body = body.replace(
        'style="color: rgb(120,120,120); font-size: 12px; margin-top: -15px;">inputs<',
        'style="color: rgb(120,120,120); font-size: 12px; margin-top: 8px;">inputs<',
    )
    body = body.replace(
        'overflow="visible" width="30">',
        'overflow="visible" width="60">',
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SHAP Explanation</title>
<style>
  /* Prevent output label text from clipping into bar chart */
  div[id$="_output_name"] {{
    min-width: 50px;
    margin-right: 6px;
  }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved SHAP explanation to: {html_path}")
    print("Tip: Open the HTML in a browser to see red/blue token highlights.")
    print("Take a screenshot to include in your report for model interpretability analysis.")


if __name__ == "__main__":
    main()
