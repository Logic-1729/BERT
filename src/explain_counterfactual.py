from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.common import ensure_dir, load_yaml
from src.utils.model_loader import load_model, load_tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint dir")
    ap.add_argument("--text", required=True, help="Text to generate counterfactual for")
    ap.add_argument("--max_edits", type=int, default=5, help="Max tokens to mask before giving up")
    ap.add_argument("--out_dir", default="assets/counterfactual", help="Output directory")
    return ap.parse_args()


def get_prediction(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    pred_cls = probs.argmax(dim=-1).item()
    confidence = probs[0, pred_cls].item()
    return pred_cls, confidence, probs


def compute_token_importance(model, inputs, tokenizer, pred_cls):
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    seq_len = input_ids.shape[1]
    mask_id = tokenizer.mask_token_id

    _, base_conf, _ = get_prediction(model, inputs)
    drops = []

    for i in range(seq_len):
        tok = input_ids[0, i].item()
        if tok in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
            continue

        modified_ids = input_ids.clone()
        modified_ids[0, i] = mask_id
        with torch.no_grad():
            outputs = model(input_ids=modified_ids, attention_mask=attention_mask)
        probs = outputs.logits.softmax(dim=-1)
        new_conf = probs[0, pred_cls].item()
        drops.append((i, base_conf - new_conf))

    drops.sort(key=lambda x: x[1], reverse=True)
    return drops


def find_counterfactual(model, inputs, tokenizer, token_list, importance_ranking, max_edits):
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"].clone()
    mask_id = tokenizer.mask_token_id
    pred_cls, base_conf, _ = get_prediction(model, inputs)

    masked_positions = []
    for pos, _drop in importance_ranking:
        input_ids[0, pos] = mask_id
        masked_positions.append(pos)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs.logits.softmax(dim=-1)
        new_pred = probs.argmax(dim=-1).item()
        new_conf = probs[0, pred_cls].item()

        if new_pred != pred_cls:
            flipped_cls = new_pred
            flipped_conf = probs[0, new_pred].item()
            return {
                "original_class": pred_cls,
                "original_confidence": base_conf,
                "flipped_class": flipped_cls,
                "flipped_confidence": flipped_conf,
                "masked_positions": masked_positions,
                "num_edits": len(masked_positions),
            }

        if len(masked_positions) >= max_edits:
            break

    if len(importance_ranking) >= 2:
        input_ids = inputs["input_ids"].clone()
        top_k = min(max_edits, len(importance_ranking))
        for pos, _ in importance_ranking[:top_k]:
            input_ids[0, pos] = mask_id
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs.logits.softmax(dim=-1)
        new_pred = probs.argmax(dim=-1).item()
        if new_pred != pred_cls:
            return {
                "original_class": pred_cls,
                "original_confidence": base_conf,
                "flipped_class": new_pred,
                "flipped_confidence": probs[0, new_pred].item(),
                "masked_positions": [p for p, _ in importance_ranking[:top_k]],
                "num_edits": top_k,
            }

    return None


def generate_counterfactual_text(token_list, masked_positions):
    tokens = list(token_list)
    for pos in masked_positions:
        if pos < len(tokens):
            tokens[pos] = "[MASK]"
    return "".join(t for t in tokens if not t.startswith("##")).replace("##", "")


def plot_counterfactual(
    token_list, importance, masked_positions, orig_class, orig_conf,
    flipped_class, flipped_conf, out_path,
):
    n = len(token_list)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["steelblue"] * n
    for pos in masked_positions:
        if pos < n:
            colors[pos] = "darkorange"

    xlabels = []
    for i, t in enumerate(token_list):
        if i in masked_positions:
            xlabels.append(f"[{t}]")
        else:
            xlabels.append(t)

    axes[0].bar(range(n), [importance.get(i, 0) for i in range(n)], color=colors)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Confidence Drop When Masked")
    axes[0].set_title("Token Importance (confidence drop on mask)")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for pos in masked_positions:
        if pos < n:
            axes[0].annotate(
                "MASKED", (pos, importance.get(pos, 0)),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="darkorange", fontweight="bold",
            )

    categories = ["Original", "Counterfactual"]
    confidences = [orig_conf, flipped_conf]
    bar_colors = ["steelblue", "darkorange"]
    axes[1].bar(categories, confidences, color=bar_colors, width=0.4)
    axes[1].set_ylabel("Confidence")
    axes[1].set_title(f"Prediction Flip (class {orig_class} → {flipped_class})")
    for i, (cat, conf) in enumerate(zip(categories, confidences)):
        axes[1].text(i, conf + 0.01, f"{conf:.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.1)

    plt.suptitle(
        f"Counterfactual Explanation: {len(masked_positions)} token(s) flipped prediction",
        fontsize=13,
    )
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

    pred_cls, base_conf, _ = get_prediction(model, inputs)
    print(f"Input: {text}")
    print(f"Predicted class: {pred_cls} (confidence: {base_conf:.4f})")

    importance = compute_token_importance(model, inputs, tokenizer, pred_cls)
    importance_dict = {pos: drop for pos, drop in importance}

    print(f"\nTop-10 most influential tokens:")
    for pos, drop in importance[:10]:
        print(f"  [{pos:2d}] {token_list[pos]:10s}  Δconf={drop:.4f}")

    result = find_counterfactual(
        model, inputs, tokenizer, token_list, importance, args.max_edits,
    )

    if result is None:
        print(f"\nCould not flip prediction with up to {args.max_edits} masked tokens.")
        print("The model is highly confident in its prediction.")
        result = {
            "original_class": pred_cls,
            "original_confidence": base_conf,
            "flipped": False,
            "max_edits_tried": args.max_edits,
        }
    else:
        cf_text = generate_counterfactual_text(token_list, result["masked_positions"])
        print(f"\nCounterfactual found with {result['num_edits']} edit(s)!")
        print(f"  Original:       {text}")
        print(f"  Counterfactual: {cf_text}")
        print(f"  Class: {result['original_class']} → {result['flipped_class']}")
        print(f"  Confidence: {result['original_confidence']:.4f} → {result['flipped_confidence']:.4f}")

        plot_counterfactual(
            token_list, importance_dict, result["masked_positions"],
            result["original_class"], result["original_confidence"],
            result["flipped_class"], result["flipped_confidence"],
            os.path.join(args.out_dir, "counterfactual_explanation.png"),
        )

    json_path = os.path.join(args.out_dir, "counterfactual_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({**result, "text": text}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
