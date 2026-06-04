from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.common import ensure_dir, load_yaml
from src.utils.model_loader import load_model, load_tokenizer

STOP_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
PUNCT_PATTERN = re.compile(r"^[^\w\u4e00-\u9fff]+$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--text", required=True, help="Text to extract keywords from")
    ap.add_argument("--top_k", type=int, default=5, help="Number of keywords to extract")
    ap.add_argument("--out_dir", default="assets/attention_keywords", help="Output directory")
    return ap.parse_args()


def extract_attention_keywords(token_list, cls_attention, top_k=5):
    scores = []
    for i, (tok, score) in enumerate(zip(token_list, cls_attention)):
        if tok in STOP_TOKENS:
            continue
        if PUNCT_PATTERN.match(tok):
            continue
        scores.append((i, tok, float(score)))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def extract_tfidf_keywords(text, top_k=5):
    char_ngrams = []
    for n in [1, 2, 3]:
        for i in range(len(text) - n + 1):
            char_ngrams.append(text[i:i + n])
    if not char_ngrams:
        return []
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        max_features=50,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
    except ValueError:
        return []
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    ranked = sorted(
        [(f, float(s)) for f, s in zip(feature_names, tfidf_scores) if s > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]


def plot_keyword_comparison(attn_keywords, tfidf_keywords, tokens, cls_attn, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n = len(tokens)
    colors = ["steelblue"] * n
    for attn_kw in attn_keywords:
        idx = attn_kw[0]
        if idx < n:
            colors[idx] = "darkorange"

    axes[0].bar(range(n), cls_attn[:n], color=colors)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("[CLS] Attention (Last Layer)")
    axes[0].set_title("Attention-Based Keywords")
    for kw in attn_keywords:
        if kw[0] < n:
            axes[0].annotate(kw[1], (kw[0], cls_attn[kw[0]]),
                             textcoords="offset points", xytext=(0, 10),
                             ha="center", fontsize=9, color="darkorange", fontweight="bold")

    if tfidf_keywords:
        labels = [kw[0] for kw in tfidf_keywords]
        values = [kw[1] for kw in tfidf_keywords]
        axes[1].barh(range(len(labels)), values, color="steelblue")
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels, fontsize=10)
        axes[1].set_xlabel("TF-IDF Score")
        axes[1].set_title("TF-IDF Keywords (Baseline)")
        axes[1].invert_yaxis()

    plt.suptitle("Keyword Extraction: Attention vs TF-IDF", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    _ = load_yaml(args.config)
    ensure_dir(args.out_dir)

    tokenizer = load_tokenizer(args.ckpt, use_fast=True)
    model = load_model(args.ckpt, attn_implementation="eager")
    model.eval()

    text = args.text
    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = torch.stack(outputs.attentions).squeeze(1).cpu().numpy()
    last_layer_avg = attentions[-1].mean(axis=0)
    cls_attention = last_layer_avg[0]

    attn_keywords = extract_attention_keywords(token_list, cls_attention, args.top_k)
    tfidf_keywords = extract_tfidf_keywords(text, args.top_k)

    print(f"\nInput: {text}")
    print(f"Predicted class: {outputs.logits.argmax(dim=-1).item()}")
    print(f"\n{'='*50}")
    print(f"  Attention-Based Keywords (Top {args.top_k})")
    print(f"{'='*50}")
    for idx, tok, score in attn_keywords:
        print(f"  [{idx:2d}] {tok:8s}  score={score:.4f}")

    print(f"\n{'='*50}")
    print(f"  TF-IDF Keywords (Top {args.top_k})")
    print(f"{'='*50}")
    for kw, score in tfidf_keywords:
        print(f"  {kw:12s}  score={score:.4f}")

    attn_set = {kw[1] for kw in attn_keywords}
    tfidf_set = {kw[0] for kw in tfidf_keywords}
    overlap = attn_set & tfidf_set
    if overlap:
        print(f"\nOverlap: {overlap}")

    plot_keyword_comparison(
        attn_keywords, tfidf_keywords, token_list, cls_attention,
        os.path.join(args.out_dir, "attention_keywords_comparison.png"),
    )

    result = {
        "text": text,
        "predicted_class": int(outputs.logits.argmax(dim=-1).item()),
        "attention_keywords": [{"token": kw[1], "score": kw[2]} for kw in attn_keywords],
        "tfidf_keywords": [{"ngram": kw[0], "score": kw[1]} for kw in tfidf_keywords],
    }
    json_path = os.path.join(args.out_dir, "attention_keywords_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
