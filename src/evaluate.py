from __future__ import annotations

import argparse
import json
import os
from typing import List

from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.loaders import load_splits
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.metrics import build_classification_report
from src.utils.plot import plot_confusion_matrix


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    ap.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint dir (e.g., outputs/bert_thucnews). Must contain config.json + model weights.",
    )
    ap.add_argument("--assets_dir", default="assets", help="Where to save plots")
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

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    model.eval()

    text_field = task_cfg.get("text_field", "text")
    label_field = task_cfg.get("label_field", "label")
    max_length = int(model_cfg.get("max_length", 256))

    def encode(text: str):
        return tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

    y_true: List[int] = []
    y_pred: List[int] = []

    import torch

    for row in dsd["test"]:
        text = row[text_field]
        label = int(row[label_field])
        inputs = encode(text)
        with torch.no_grad():
            out = model(**inputs)
            pred = int(out.logits.argmax(dim=-1).cpu().item())
        y_true.append(label)
        y_pred.append(pred)

    ensure_dir(args.assets_dir)
    cm_path = os.path.join(args.assets_dir, f"confusion_matrix_{task_cfg['name']}.png")
    plot_confusion_matrix(y_true, y_pred, out_path=cm_path, normalize=True)

    report = build_classification_report(y_true, y_pred)
    report_path = os.path.join(args.assets_dir, f"classification_report_{task_cfg['name']}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Saved classification report to: {report_path}")


if __name__ == "__main__":
    main()
