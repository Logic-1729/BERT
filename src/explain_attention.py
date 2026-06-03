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
    ap.add_argument("--text", required=True, help="Text to explain")
    ap.add_argument("--out_dir", default="assets/attention", help="Output directory")
    ap.add_argument("--layer", type=int, default=None, help="Specific layer to plot (-1 for last). If unset, plots all layers in a grid.")
    return ap.parse_args()


def _plot_attention_heatmap(attn_matrix, tokens, title, out_path, figsize=(12, 10)):
    n = len(tokens)
    attn = attn_matrix[:n, :n]

    plt.figure(figsize=figsize)
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="Reds", square=True, linewidths=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _plot_layer_grid(attentions, tokens, out_path, ncols=4):
    """Plot every layer's avg-over-heads attention in a grid."""
    num_layers = attentions.shape[0]
    nrows = int(np.ceil(num_layers / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten() if num_layers > 1 else [axes]

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_avg = attentions[layer_idx].mean(axis=0)  # avg over heads
        n = len(tokens)
        sns.heatmap(
            layer_avg[:n, :n],
            ax=ax, cmap="Reds", square=True,
            xticklabels=tokens, yticklabels=tokens,
            cbar=(layer_idx == 0),
        )
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)

    for idx in range(num_layers, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Attention Patterns (averaged over all heads per layer)", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    args = parse_args()
    _ = load_yaml(args.config)
    ensure_dir(args.out_dir)

    tokenizer = load_tokenizer(args.ckpt, use_fast=True)
    model = load_model(args.ckpt)
    model.eval()

    inputs = tokenizer(args.text, truncation=True, max_length=128, return_tensors="pt")
    token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer, shape: (num_layers, B, H, S, S)
    attentions = torch.stack(outputs.attentions).squeeze(1)  # (num_layers, num_heads, seq_len, seq_len)
    attentions = attentions.cpu().numpy()

    pred = outputs.logits.argmax(dim=-1).item()
    print(f"Predicted class: {pred}")

    # 1. Global average attention (over all layers and heads)
    global_avg = attentions.mean(axis=(0, 1))
    _plot_attention_heatmap(
        global_avg, token_list,
        title="Average Attention (all layers & heads)",
        out_path=os.path.join(args.out_dir, "attention_global_avg.png"),
    )

    # 2. Per-layer grid (avg over heads)
    _plot_layer_grid(
        attentions, token_list,
        out_path=os.path.join(args.out_dir, "attention_per_layer.png"),
    )

    # 3. Specific layer if requested (with per-head detail)
    if args.layer is not None:
        layer_idx = args.layer if args.layer >= 0 else attentions.shape[0] + args.layer
        layer_attn = attentions[layer_idx]  # (num_heads, seq_len, seq_len)
        num_heads = layer_attn.shape[0]
        ncols = min(4, num_heads)
        nrows = int(np.ceil(num_heads / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        for h in range(num_heads):
            sns.heatmap(
                layer_attn[h][:len(token_list), :len(token_list)],
                ax=axes[h], cmap="Reds", square=True,
                xticklabels=token_list, yticklabels=token_list,
                cbar=(h == 0),
            )
            axes[h].set_title(f"Head {h + 1}", fontsize=9)
            axes[h].set_xticklabels(axes[h].get_xticklabels(), fontsize=5, rotation=45, ha="right")
            axes[h].set_yticklabels(axes[h].get_yticklabels(), fontsize=5, rotation=0)
        for h in range(num_heads, len(axes)):
            axes[h].axis("off")
        plt.suptitle(f"Layer {layer_idx + 1} — Per-head Attention", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"attention_layer_{layer_idx + 1}_heads.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # 4. [CLS] token attention across layers
    cls_attn_per_layer = []
    for l in range(attentions.shape[0]):
        cls_attn = attentions[l].mean(axis=0)[0, :len(token_list)]  # avg heads, [CLS] -> all tokens
        cls_attn_per_layer.append(cls_attn)
    cls_attn_matrix = np.stack(cls_attn_per_layer)  # (num_layers, seq_len)

    plt.figure(figsize=(max(8, len(token_list) * 0.5), 6))
    sns.heatmap(
        cls_attn_matrix,
        xticklabels=token_list, cmap="Reds",
        yticklabels=[f"L{i + 1}" for i in range(cls_attn_matrix.shape[0])],
        linewidths=0.3,
    )
    plt.title("[CLS] Token Attention to Input Tokens (per layer, avg over heads)", fontsize=13)
    plt.xlabel("Input Token")
    plt.ylabel("Layer")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "attention_cls_per_layer.png"), dpi=200)
    plt.close()

    print(f"Saved attention visualizations to: {args.out_dir}")
    print("Tip: In your report, discuss how attention patterns reveal which tokens the model focuses on.")
    print("Compare with SHAP/LIME results — attention is model-internal, while SHAP/LIME are post-hoc.")


if __name__ == "__main__":
    main()
