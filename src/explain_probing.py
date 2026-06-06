from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import DatasetDict
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.data.loaders import load_splits
from src.explain_attention_prune import CONTENT_POS, FUNCTION_POS, map_pos_to_tokens
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.model_loader import load_model, load_tokenizer

POS_GROUPS = {
    "Noun":      {"n", "nr", "ns", "nt", "nz", "ng"},
    "Verb":      {"v", "vd", "vn", "vf", "vx", "vi"},
    "Adjective": {"a", "ad", "an", "ag", "al"},
    "Adverb":    {"d", "dg"},
    "Pronoun":   {"r", "rr", "rz", "ry", "rg"},
    "Numeral":   {"m", "mq", "qv"},
    "Classifier":{"q", "qt", "qv"},
    "Preposition":{"p", "pba", "pbei"},
    "Conjunction":{"c", "cc"},
    "Particle":  {"u", "uzhe", "ule", "uguo", "ude1", "ude2", "ude3", "usuo", "udeng", "uyy", "udh", "uls", "uzhi", "ulian"},
    "Other_Func": {"f", "t", "e", "y", "o", "h", "k", "w", "x"},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--max_samples", type=int, default=500, help="Test samples for probing")
    ap.add_argument("--out_dir", default="assets/probing", help="Output directory")
    return ap.parse_args()


def group_pos_tag(tag: str) -> str:
    for group, members in POS_GROUPS.items():
        if tag in members:
            return group
    return "Other"


def extract_all_hidden_states(model, tokenizer, texts, max_length):
    """Return lists of (token, hidden-vec, pos-tag) per layer, across all samples.

    Returns:
        per_layer_data: list of 13 dicts, each with keys "X" (np.array [N, 768]),
                        "y_binary" (0/1), "y_group" (str labels).
    """
    model.eval()
    per_layer = [{"X": [], "y_binary": [], "y_group": []} for _ in range(13)]

    for text in tqdm(texts, desc="Extracting hidden states", unit="texts"):
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tags = map_pos_to_tokens(text, token_list)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx, hidden in enumerate(outputs.hidden_states):
            vecs = hidden.squeeze(0).cpu().numpy()
            for tok_idx in range(len(token_list)):
                tag = tags[tok_idx]
                if tag == "special":
                    continue
                per_layer[layer_idx]["X"].append(vecs[tok_idx])
                per_layer[layer_idx]["y_binary"].append(1 if tag in CONTENT_POS else 0)
                per_layer[layer_idx]["y_group"].append(group_pos_tag(tag))

    for d in per_layer:
        d["X"] = np.array(d["X"])
        d["y_binary"] = np.array(d["y_binary"])
        d["y_group"] = np.array(d["y_group"])

    return per_layer


def probe_accuracy(X, y, max_samples=8000):
    mask = y != -1
    X, y = X[mask], y[mask]
    if len(X) < 50:
        return 0.0, None
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    n = len(X)
    split = int(n * 0.8)
    clf = RidgeClassifier()
    clf.fit(X[:split], y[:split])
    y_pred = clf.predict(X[split:])
    acc = accuracy_score(y[split:], y_pred)

    per_class = {}
    for cls in set(y):
        mask_c = y[split:] == cls
        if mask_c.sum() > 0:
            per_class[cls] = float((y_pred[mask_c] == cls).mean())
    return acc, per_class


def plot_results(layer_binary, layer_group, group_names, group_matrix, out_dir):
    ensure_dir(out_dir)
    layers = list(range(13))
    labels = ["Emb"] + [f"L{i}" for i in range(1, 13)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(layers, layer_binary, "o-", color="steelblue", lw=2, ms=6,
            label="Content vs Function (binary)")
    ax.plot(layers, layer_group, "s-", color="darkorange", lw=2, ms=6,
            label="POS Group (multi-class)")

    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.4, label="Chance (binary)")

    best_idx = int(np.argmax(layer_binary))
    ax.annotate(f"L{best_idx}: {layer_binary[best_idx]:.3f}",
                xy=(best_idx, layer_binary[best_idx]),
                xytext=(best_idx + 1.2, layer_binary[best_idx] + 0.015),
                fontsize=9, color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8))

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title("Linguistic Knowledge Emergence Across BERT Layers", fontsize=14)
    ax.set_xticks(layers)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probing_accuracy_curves.png"), dpi=200)
    plt.close()

    n_groups = len(group_names)
    fig, ax = plt.subplots(figsize=(14, max(5, n_groups * 0.45)))
    sns.heatmap(
        group_matrix, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
        xticklabels=labels, yticklabels=group_names,
        cbar_kws={"label": "Accuracy"}, vmin=0, vmax=1,
        linewidths=0.5,
    )
    ax.set_title("Per-Category POS Probe Accuracy Across Layers", fontsize=13)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("POS Group", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probing_pos_heatmap.png"), dpi=200)
    plt.close()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    model_cfg = cfg["model"]

    splits = load_splits(
        data_cfg["data_dir"], data_cfg["train_file"],
        data_cfg["valid_file"], data_cfg["test_file"],
    )
    dsd = DatasetDict(splits)

    text_field = task_cfg.get("text_field", "text")
    max_length = int(model_cfg.get("max_length", 256))
    samples = list(dsd["test"])[: args.max_samples]
    texts = [row[text_field] for row in samples]
    print(f"Probing on {len(texts)} test samples")

    tokenizer = load_tokenizer(args.ckpt, use_fast=True)
    model = load_model(args.ckpt, attn_implementation="eager")
    per_layer = extract_all_hidden_states(model, tokenizer, texts, max_length)

    total_tokens = sum(d["X"].shape[0] for d in per_layer)
    print(f"Total probe tokens per layer: ~{total_tokens // 13}")

    layer_binary = []
    layer_group = []
    group_names = list(POS_GROUPS.keys())
    group_matrix = np.zeros((len(group_names), 13))

    for layer_idx in range(13):
        data = per_layer[layer_idx]
        acc_b, _ = probe_accuracy(data["X"], data["y_binary"])
        layer_binary.append(acc_b)

        le = LabelEncoder()
        y_enc = le.fit_transform(data["y_group"])
        acc_g, per_cls = probe_accuracy(data["X"], y_enc)
        layer_group.append(acc_g)

        if per_cls:
            for enc_id, cls_acc in per_cls.items():
                cls_name = le.inverse_transform([enc_id])[0]
                if cls_name in group_names:
                    row = group_names.index(cls_name)
                    group_matrix[row, layer_idx] = cls_acc

        label = labels[layer_idx] if (labels := ["Emb"] + [f"L{i}" for i in range(1, 13)]) else f"L{layer_idx}"
        print(f"  {label:4s}  binary={acc_b:.4f}  group={acc_g:.4f}")

    plot_results(layer_binary, layer_group, group_names, group_matrix, args.out_dir)

    result = {
        "num_samples": len(texts),
        "total_tokens_per_layer": int(total_tokens // 13),
        "group_names": group_names,
        "accuracy_binary": {f"layer_{i}": round(float(a), 4) for i, a in enumerate(layer_binary)},
        "accuracy_group": {f"layer_{i}": round(float(a), 4) for i, a in enumerate(layer_group)},
        "best_binary_layer": int(np.argmax(layer_binary)),
        "best_binary_acc": round(float(max(layer_binary)), 4),
        "best_group_layer": int(np.argmax(layer_group)),
        "best_group_acc": round(float(max(layer_group)), 4),
    }
    with open(os.path.join(args.out_dir, "probing_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.out_dir}/")
    print(f"Best binary layer: L{result['best_binary_layer']} ({result['best_binary_acc']:.4f})")
    print(f"Best group  layer: L{result['best_group_layer']} ({result['best_group_acc']:.4f})")


if __name__ == "__main__":
    main()
