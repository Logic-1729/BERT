from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.loaders import load_splits
from src.utils.common import ensure_dir, load_yaml, set_seed
from src.utils.metrics import softmax


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/*.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    splits = load_splits(
        data_cfg["data_dir"],
        data_cfg["train_file"],
        data_cfg["valid_file"],
        data_cfg["test_file"],
    )
    dsd = DatasetDict(splits)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"], use_fast=True)

    text_field = task_cfg.get("text_field", "text")
    label_field = task_cfg.get("label_field", "label")
    max_length = int(model_cfg.get("max_length", 256))

    def tokenize_fn(batch: Dict):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
        )

    dsd = dsd.map(tokenize_fn, batched=True)
    dsd = dsd.rename_column(label_field, "labels")
    dsd.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    num_labels = int(task_cfg["num_labels"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["pretrained_name"], num_labels=num_labels
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        from sklearn.metrics import f1_score

        macro_f1 = float(f1_score(labels, preds, average="macro"))
        return {"accuracy": acc, "macro_f1": macro_f1}

    ensure_dir(train_cfg["output_dir"])

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 32)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        eval_strategy=str(train_cfg.get("eval_strategy", "epoch")),
        save_strategy=str(train_cfg.get("save_strategy", "epoch")),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "macro_f1")),
        greater_is_better=bool(train_cfg.get("greater_is_better", True)),
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model + tokenizer
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    # quick test prediction dump
    preds = trainer.predict(dsd["test"])
    prob = softmax(preds.predictions)
    out_path = os.path.join(train_cfg["output_dir"], "test_predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in prob:
            row = {
                "pred": int(p.argmax()),
                "prob": [float(x) for x in p],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved model to: {train_cfg['output_dir']}")
    print(f"Saved test predictions to: {out_path}")


if __name__ == "__main__":
    main()
