from __future__ import annotations

import os

from datasets import Dataset


def _read_jsonl(path: str) -> Dataset:
    import json

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def load_splits(data_dir: str, train_file: str, valid_file: str, test_file: str):
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)
    test_path = os.path.join(data_dir, test_file)

    for p in [train_path, valid_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing dataset file: {p}. Please prepare data following docs/DATASETS.md"
            )

    return {
        "train": _read_jsonl(train_path),
        "validation": _read_jsonl(valid_path),
        "test": _read_jsonl(test_path),
    }
