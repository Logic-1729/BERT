from __future__ import annotations

import argparse
import os

import jieba.posseg as pseg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from src.utils.common import ensure_dir, load_yaml
from src.utils.model_loader import load_model, load_tokenizer

CONTENT_POS = {"n", "nr", "ns", "nt", "nz", "v", "vd", "vn", "a", "ad", "an"}
FUNCTION_POS = {"u", "p", "c", "d", "r", "w", "x", "m", "q", "f", "t", "e", "y", "o", "h", "k"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--text", required=True, help="Text to explain")
    ap.add_argument("--out_dir", default="assets/attention_prune", help="Output directory")
    return ap.parse_args()


def map_pos_to_tokens(text: str, token_list: list[str]) -> list[str]:
    """Align jieba POS tags to BERT subword tokens (character-level for Chinese)."""
    pos_per_char = {}
    for word, flag in pseg.cut(text):
        for ch in word:
            pos_per_char.setdefault(ch, flag)

    tags = []
    char_idx = 0
    for tok in token_list:
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]") or tok.startswith("##"):
            tags.append("special")
        elif char_idx < len(text):
            ch = text[char_idx]
            tags.append(pos_per_char.get(ch, "x"))
            char_idx += 1
        else:
            tags.append("x")
    return tags


def boosted_attention_mask(original_mask: torch.Tensor, tags: list[str], boost: float = 1.0, suppress: float = -0.5) -> torch.Tensor:
    """Create a 4D attention bias mask based on POS tags.

    In transformers v5.x, the 2D attention_mask is internally cast to ``torch.bool``
    inside ``_preprocess_mask_arguments``, so scaled float values (2.0, 0.3) are
    indistinguishable from the original 1.0 — the pruning has **no effect**.

    A 4D mask of shape ``[1, 1, seq_len, seq_len]`` bypasses this conversion and
    is added directly to the attention logits (Q·Kᵀ / √dₖ) before softmax:
        - positive bias → boosted attention
        - negative bias → suppressed attention
        - 0.0 → no change

    Returns:
        4D float tensor ``[1, 1, seq_len, seq_len]`` with additive biases.
    """
    seq_len = original_mask.shape[1]
    bias = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
    finfo_min = torch.finfo(torch.float32).min

    for i, tag in enumerate(tags):
        if original_mask[0, i].item() == 0:
            bias[0, 0, :, i] = finfo_min
            continue
        if tag == "special":
            continue
        if tag in CONTENT_POS:
            bias[0, 0, :, i] += boost
        elif tag in FUNCTION_POS:
            bias[0, 0, :, i] += suppress

    return bias


def plot_attention_comparison(original_attn, pruned_attn, tokens, tags, out_path):
    n = len(tokens)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, data, title in [
        (axes[0], original_attn, "Original Attention (all layers & heads avg)"),
        (axes[1], pruned_attn, "POS-Guided Attention (content words boosted)"),
    ]:
        sns.heatmap(
            data[:n, :n], ax=ax, cmap="Reds", square=True,
            xticklabels=tokens, yticklabels=tokens,
            cbar=True,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, rotation=0)

        for i in range(n):
            if tags[i] in CONTENT_POS:
                ax.get_xticklabels()[i].set_color("blue")
                ax.get_xticklabels()[i].set_fontweight("bold")

    plt.suptitle("POS-Guided Attention Mask Pruning", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_cls_attention_shift(original_cls, pruned_cls, tokens, tags, out_path):
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 5))
    x = np.arange(n)
    width = 0.35
    bars1 = ax.bar(x - width / 2, original_cls, width, label="Original", color="steelblue", alpha=0.7)
    bars2 = ax.bar(x + width / 2, pruned_cls, width, label="POS-Guided", color="darkorange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("[CLS] Attention Weight")
    ax.set_title("[CLS] Token Attention Shift After POS-Guided Pruning")
    ax.legend()
    for i, tag in enumerate(tags):
        if tag in CONTENT_POS:
            ax.get_xticklabels()[i].set_color("blue")
            ax.get_xticklabels()[i].set_fontweight("bold")
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
    tags = map_pos_to_tokens(text, token_list)

    print("Token POS mapping:")
    for tok, tag in zip(token_list, tags):
        display_tag = f"[{tag}]" if tag in CONTENT_POS else f"({tag})" if tag in FUNCTION_POS else tag
        print(f"  {tok:12s} → {display_tag}")

    with torch.no_grad():
        out_orig = model(**inputs, output_attentions=True)

    boosted_mask = boosted_attention_mask(inputs["attention_mask"], tags)
    with torch.no_grad():
        out_pruned = model(input_ids=inputs["input_ids"], attention_mask=boosted_mask, output_attentions=True)

    pred_orig = out_orig.logits.argmax(dim=-1).item()
    pred_pruned = out_pruned.logits.argmax(dim=-1).item()
    print(f"\nPredicted class — Original: {pred_orig} | POS-Guided: {pred_pruned}")

    attn_orig = torch.stack(out_orig.attentions).squeeze(1).cpu().numpy()
    attn_pruned = torch.stack(out_pruned.attentions).squeeze(1).cpu().numpy()

    global_orig = attn_orig.mean(axis=(0, 1))
    global_pruned = attn_pruned.mean(axis=(0, 1))

    plot_attention_comparison(
        global_orig, global_pruned, token_list, tags,
        os.path.join(args.out_dir, "attention_prune_comparison.png"),
    )

    last_layer_orig = attn_orig[-1].mean(axis=0)
    last_layer_pruned = attn_pruned[-1].mean(axis=0)
    cls_orig = last_layer_orig[0, :len(token_list)]
    cls_pruned = last_layer_pruned[0, :len(token_list)]

    plot_cls_attention_shift(
        cls_orig, cls_pruned, token_list, tags,
        os.path.join(args.out_dir, "attention_prune_cls_shift.png"),
    )

    print(f"Saved visualizations to: {args.out_dir}")
    print("Blue/bold labels = content words (boosted); plain labels = function words (suppressed).")


if __name__ == "__main__":
    main()
