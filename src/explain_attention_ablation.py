from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DatasetDict
from tqdm import tqdm

from src.data.loaders import load_splits
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.model_loader import load_model, load_tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--out_dir", default="assets/attention_ablation", help="Output directory")
    ap.add_argument("--max_samples", type=int, default=500, help="Max test samples for speed")
    return ap.parse_args()


def evaluate_accuracy(model, tokenizer, dataset, max_length, text_field, label_field, max_samples):
    model.eval()
    y_true, y_pred = [], []
    count = 0
    with torch.no_grad():
        for row in tqdm(dataset, desc="Evaluating", unit="samples"):
            if count >= max_samples:
                break
            text = row[text_field]
            label = int(row[label_field])
            inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            out = model(**inputs)
            pred = int(out.logits.argmax(dim=-1).cpu().item())
            y_true.append(label)
            y_pred.append(pred)
            count += 1
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def prune_heads(model, heads_to_prune):
    for layer_idx, head_indices in heads_to_prune.items():
        if layer_idx < len(model.bert.encoder.layer):
            layer = model.bert.encoder.layer[layer_idx]
            attn = layer.attention.self
            num_heads = attn.num_attention_heads
            head_dim = attn.attention_head_size
            all_head_size = num_heads * head_dim
            valid = [h for h in head_indices if 0 <= h < num_heads]
            if not valid:
                continue

            q_weight = attn.query.weight.data
            k_weight = attn.key.weight.data
            v_weight = attn.value.weight.data
            q_bias = attn.query.bias.data if attn.query.bias is not None else None
            k_bias = attn.key.bias.data if attn.key.bias is not None else None
            v_bias = attn.value.bias.data if attn.value.bias is not None else None
            out_proj = layer.attention.output
            out_weight = out_proj.dense.weight.data
            out_bias = out_proj.dense.bias.data if out_proj.dense.bias is not None else None

            mask = torch.ones(num_heads)
            for h in valid:
                mask[h] = 0

            for h in valid:
                q_weight[h * head_dim:(h + 1) * head_dim, :] = 0
                k_weight[h * head_dim:(h + 1) * head_dim, :] = 0
                v_weight[h * head_dim:(h + 1) * head_dim, :] = 0
                if q_bias is not None:
                    q_bias[h * head_dim:(h + 1) * head_dim] = 0
                if k_bias is not None:
                    k_bias[h * head_dim:(h + 1) * head_dim] = 0
                if v_bias is not None:
                    v_bias[h * head_dim:(h + 1) * head_dim] = 0
                out_weight[:, h * head_dim:(h + 1) * head_dim] = 0


def compute_cls_head_importance(model, tokenizer, text, max_length=128):
    model.eval()
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = torch.stack(outputs.attentions).squeeze(1).cpu().numpy()
    num_layers, num_heads, seq_len, _ = attentions.shape
    cls_importance = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            cls_importance[l, h] = attentions[l, h, 0, 0]
    return cls_importance


def plot_ablation_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    categories = [r["label"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(categories)), accuracies, color="steelblue")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Attention Head/Layer Pruning Ablation")
    ax.axhline(y=accuracies[0], color="red", linestyle="--", alpha=0.5, label=f"Baseline: {accuracies[0]:.4f}")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    ax.set_ylim(max(0, min(accuracies) - 0.03), 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_layers.png"), dpi=200)
    plt.close()

    json_path = os.path.join(out_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(42)
    ensure_dir(args.out_dir)

    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    model_cfg = cfg["model"]

    splits = load_splits(
        data_cfg["data_dir"],
        data_cfg["train_file"],
        data_cfg["valid_file"],
        data_cfg["test_file"],
    )
    dsd = DatasetDict(splits)
    text_field = task_cfg.get("text_field", "text")
    label_field = task_cfg.get("label_field", "label")
    max_length = int(model_cfg.get("max_length", 256))

    results = []

    def load_fresh_model():
        tokenizer = load_tokenizer(args.ckpt, use_fast=True)
        model = load_model(args.ckpt, attn_implementation="eager")
        return model, tokenizer

    print("=" * 50)
    print("  Baseline (no pruning)")
    model, tokenizer = load_fresh_model()
    baseline_acc = evaluate_accuracy(model, tokenizer, dsd["test"], max_length, text_field, label_field, args.max_samples)
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")
    results.append({"label": "Baseline (0%)", "accuracy": baseline_acc, "pruned": "none"})

    print("\n" + "=" * 50)
    print("  Ablation: Prune by Layer Groups")
    layer_groups = [
        ("Bottom L1-L4", list(range(0, 4))),
        ("Middle L5-L8", list(range(4, 8))),
        ("Top L9-L12", list(range(8, 12))),
    ]
    for group_name, layer_indices in layer_groups:
        model, tokenizer = load_fresh_model()
        heads_to_prune = {l: list(range(12)) for l in layer_indices}
        prune_heads(model, heads_to_prune)
        acc = evaluate_accuracy(model, tokenizer, dsd["test"], max_length, text_field, label_field, args.max_samples)
        print(f"  {group_name}: {acc:.4f} (Δ={acc - baseline_acc:+.4f})")
        results.append({"label": f"Prune {group_name}", "accuracy": acc, "pruned": group_name})

    print("\n" + "=" * 50)
    print("  Ablation: Prune by Head Percentage")
    for pct in [0.1, 0.2, 0.3, 0.4, 0.5]:
        model, tokenizer = load_fresh_model()
        sample_text = dsd["test"][0][text_field]
        cls_imp = compute_cls_head_importance(model, tokenizer, sample_text)
        flat = [(l, h, cls_imp[l, h]) for l in range(12) for h in range(12)]
        flat.sort(key=lambda x: x[2], reverse=True)
        n_remove = int(12 * 12 * pct)
        heads_to_prune = {}
        for l, h, _ in flat[:n_remove]:
            heads_to_prune.setdefault(l, []).append(h)
        prune_heads(model, heads_to_prune)
        acc = evaluate_accuracy(model, tokenizer, dsd["test"], max_length, text_field, label_field, args.max_samples)
        print(f"  Prune {pct*100:.0f}% ({n_remove} heads, [CLS]-heavy first): {acc:.4f} (Δ={acc - baseline_acc:+.4f})")
        results.append({"label": f"[CLS]-Heavy {pct*100:.0f}%", "accuracy": acc, "pruned": f"cls_heavy_{pct}"})

    print("\n" + "=" * 50)
    print("  Ablation: Prune by Random Heads")
    for pct in [0.1, 0.3, 0.5]:
        model, tokenizer = load_fresh_model()
        rng = np.random.RandomState(42)
        all_heads = [(l, h) for l in range(12) for h in range(12)]
        rng.shuffle(all_heads)
        n_remove = int(12 * 12 * pct)
        heads_to_prune = {}
        for l, h in all_heads[:n_remove]:
            heads_to_prune.setdefault(l, []).append(h)
        prune_heads(model, heads_to_prune)
        acc = evaluate_accuracy(model, tokenizer, dsd["test"], max_length, text_field, label_field, args.max_samples)
        print(f"  Random {pct*100:.0f}% ({n_remove} heads): {acc:.4f} (Δ={acc - baseline_acc:+.4f})")
        results.append({"label": f"Random {pct*100:.0f}%", "accuracy": acc, "pruned": f"random_{pct}"})

    plot_ablation_results(results, args.out_dir)
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
