from __future__ import annotations

import argparse
import sys
from typing import List

import torch
from datasets import DatasetDict
from tqdm import tqdm

from src.data.loaders import load_splits
from src.explain_attention_prune import (
    CONTENT_POS,
    FUNCTION_POS,
    boosted_attention_mask,
    map_pos_to_tokens,
)
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.model_loader import load_model, load_tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument("--ckpt", required=True, help="Checkpoint dir")
    ap.add_argument("--max_samples", type=int, default=0, help="Limit test samples (0 = all)")
    ap.add_argument("--boost", type=float, default=1.0, help="Additive bias for content words")
    ap.add_argument("--suppress", type=float, default=-0.5, help="Additive bias for function words")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

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

    tokenizer = load_tokenizer(args.ckpt, use_fast=True)
    model = load_model(args.ckpt, attn_implementation="eager")
    model.eval()

    text_field = task_cfg.get("text_field", "text")
    label_field = task_cfg.get("label_field", "label")
    max_length = int(model_cfg.get("max_length", 256))

    samples = list(dsd["test"])
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    y_true: List[int] = []
    y_vanilla: List[int] = []
    y_pruned: List[int] = []

    for row in tqdm(samples, desc="Evaluating", unit="samples"):
        text = row[text_field]
        label = int(row[label_field])
        y_true.append(label)

        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

        with torch.no_grad():
            out_vanilla = model(**inputs)
            pred_vanilla = int(out_vanilla.logits.argmax(dim=-1).cpu().item())
        y_vanilla.append(pred_vanilla)

        token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tags = map_pos_to_tokens(text, token_list)

        mask_4d = boosted_attention_mask(
            inputs["attention_mask"], tags, boost=args.boost, suppress=args.suppress
        )

        with torch.no_grad():
            out_pruned = model(
                input_ids=inputs["input_ids"], attention_mask=mask_4d
            )
            pred_pruned = int(out_pruned.logits.argmax(dim=-1).cpu().item())
        y_pruned.append(pred_pruned)

    acc_vanilla = sum(1 for t, p in zip(y_true, y_vanilla) if t == p) / len(y_true)
    acc_pruned = sum(1 for t, p in zip(y_true, y_pruned) if t == p) / len(y_true)

    print(f"\n{'='*60}")
    print(f"Test samples: {len(y_true)}")
    print(f"Boost={args.boost}, Suppress={args.suppress}")
    print(f"Vanilla accuracy:        {acc_vanilla:.4f} ({acc_vanilla*100:.2f}%)")
    print(f"POS-Guided accuracy:     {acc_pruned:.4f} ({acc_pruned*100:.2f}%)")
    diff = acc_pruned - acc_vanilla
    print(f"Difference:              {diff:+.4f} ({diff*100:+.2f}%)")

    same = sum(1 for v, p in zip(y_vanilla, y_pruned) if v == p)
    changed = len(y_true) - same
    print(f"Predictions changed:     {changed}/{len(y_true)} ({changed/len(y_true)*100:.1f}%)")

    changed_correct = 0
    changed_wrong = 0
    for t, v, p in zip(y_true, y_vanilla, y_pruned):
        if v != p:
            if p == t:
                changed_correct += 1
            else:
                changed_wrong += 1

    if changed > 0:
        print(f"  → fixed (wrong→right):  {changed_correct}")
        print(f"  → broken (right→wrong): {changed_wrong}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
