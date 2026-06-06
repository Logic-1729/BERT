from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.utils.common import ensure_dir, load_yaml
from src.utils.model_loader import load_model, load_tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--text", required=True, help="Text to analyze")
    ap.add_argument("--out_dir", default="assets/attention_rollout", help="Output directory")
    return ap.parse_args()


def compute_rollout(attentions):
    """Compute attention rollout across all layers.

    Formula from Abnar & Zuidema (2020):
        R = (0.5·A₁ + 0.5·I) × (0.5·A₂ + 0.5·I) × ... × (0.5·A_L + 0.5·I)

    where Aᵢ is the head-averaged attention matrix at layer i.
    """
    n_layers, _, n_heads, seq_len, _ = attentions.shape
    rollout = torch.eye(seq_len)

    for layer_idx in range(n_layers):
        attn_avg = attentions[layer_idx, 0].mean(dim=0)
        attn_res = 0.5 * attn_avg + 0.5 * torch.eye(seq_len)
        rollout = attn_res @ rollout

    return rollout.cpu().numpy()


def plot_rollout_comparison(raw_attn, rollout_attn, tokens, out_path):
    n = len(tokens)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    vmax = max(raw_attn.max(), rollout_attn.max())

    for ax, data, title in [
        (axes[0], raw_attn, "Raw Attention (last layer, head-averaged)"),
        (axes[1], rollout_attn, "Attention Rollout (full layer propagation)"),
    ]:
        sns.heatmap(
            data[:n, :n], ax=ax, cmap="Reds", square=True,
            xticklabels=tokens, yticklabels=tokens,
            cbar=True, vmin=0, vmax=vmax,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)

    plt.suptitle("Attention Rollout: Raw vs Global Information Flow", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cls_comparison(raw_cls, rollout_cls, tokens, out_path):
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 5))
    x = np.arange(n)
    width = 0.35

    ax.bar(x - width / 2, raw_cls, width, label="Raw Attention (L12)", color="steelblue", alpha=0.7)
    ax.bar(x + width / 2, rollout_cls, width, label="Attention Rollout", color="darkorange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Attention Weight")
    ax.set_title("[CLS] Token Attention: Raw vs Rollout")
    ax.legend()
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

    pred_cls = outputs.logits.argmax(dim=-1).item()
    print(f"Input: {text}")
    print(f"Predicted class: {pred_cls}")

    attentions = torch.stack(outputs.attentions)
    rollout = compute_rollout(attentions)

    last_layer_raw = attentions[-1, 0].mean(dim=0).cpu().numpy()

    n = len(token_list)

    print(f"\n{'='*60}")
    print(f"  [CLS] Token Attention Comparison")
    print(f"{'='*60}")
    print(f"  {'Token':12s} {'Raw (L12)':>10s} {'Rollout':>10s} {'Ratio':>10s}")
    print(f"  {'-'*42}")
    raw_cls = last_layer_raw[0, :n]
    rollout_cls = rollout[0, :n]

    top_by_rollout = sorted(
        [(i, token_list[i], rollout_cls[i]) for i in range(n)],
        key=lambda x: x[2], reverse=True,
    )
    for i, tok, s in top_by_rollout[:8]:
        raw_val = raw_cls[i]
        ratio = s / (raw_val + 1e-8)
        print(f"  {tok:12s} {raw_val:10.4f} {s:10.4f} {ratio:10.1f}x")

    plot_rollout_comparison(
        last_layer_raw, rollout, token_list,
        os.path.join(args.out_dir, "attention_rollout_comparison.png"),
    )
    plot_cls_comparison(
        raw_cls, rollout_cls, token_list,
        os.path.join(args.out_dir, "attention_rollout_cls_shift.png"),
    )

    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
