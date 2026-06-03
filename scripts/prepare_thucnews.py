#!/usr/bin/env python3
"""Prepare THUCNews dataset for training.

This script can work in two modes:

  A) Auto-download:
     python scripts/prepare_thucnews.py

  B) Manual download (if auto fails due to network):
     1. Download THUCNews.jsonl manually (browser / download manager / VPN)
        from: https://hf-mirror.com/datasets/SirlyDreamer/THUCNews/resolve/main/THUCNews.jsonl
     2. Place it at: data/thucnews/THUCNews.jsonl
     3. Run: python scripts/prepare_thucnews.py --local data/thucnews/THUCNews.jsonl

  Options:
     --max_per_class N    Samples per class (default 5000, use --full for all)
     --local PATH         Use a local JSONL file instead of downloading
     --full               Use all samples (no per-class limit)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter

DATA_DIR = "data/thucnews"
DEFAULT_MAX_PER_CLASS = 5000
SEED = 42

SPLIT_MAP = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "test": "test.jsonl",
}

DOWNLOAD_URL = "https://hf-mirror.com/datasets/SirlyDreamer/THUCNews/resolve/main/THUCNews.jsonl"


def download_with_wget(url: str, dest: str) -> bool:
    """Download a file with wget, supporting resume and retries."""
    print(f"Downloading: {url}")
    print(f"Target:     {dest}")
    print()
    # wget -c: resume, -t 0: infinite retries, --retry-connrefused: retry on connection refused
    cmd = [
        "wget", "-c", "-t", "0", "--retry-connrefused",
        "--timeout=30", "--waitretry=10",
        "-O", dest, url,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def read_jsonl(path: str):
    """Yield rows from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def train_valid_test_split(rows: list, seed: int):
    """Shuffle and split rows into 80/10/10."""
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)
    return {
        "train": rows[:train_end],
        "validation": rows[train_end:valid_end],
        "test": rows[valid_end:],
    }


def sample_per_class(rows: list, max_per_class: int | None) -> list:
    """Uniformly sample up to max_per_class rows per label."""
    if max_per_class is None:
        return rows
    by_label: dict[int, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    sampled = []
    for label in sorted(by_label):
        sampled.extend(by_label[label][:max_per_class])
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Prepare THUCNews dataset")
    parser.add_argument("--local", type=str, default=None, help="Path to local JSONL file")
    parser.add_argument("--full", action="store_true", help="Use all samples")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help=f"Samples per class in train (default: {DEFAULT_MAX_PER_CLASS})")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- get the JSONL file ----
    if args.local:
        jsonl_path = args.local
        if not os.path.exists(jsonl_path):
            print(f"File not found: {jsonl_path}")
            sys.exit(1)
        print(f"Using local file: {jsonl_path}")
    else:
        jsonl_path = os.path.join(DATA_DIR, "THUCNews.jsonl")
        if os.path.exists(jsonl_path):
            print(f"Found existing file: {jsonl_path}")
            print("  (Delete it to force re-download)")
        else:
            ok = download_with_wget(DOWNLOAD_URL, jsonl_path)
            if not ok:
                print()
                print("=" * 60)
                print("  Auto-download failed (network issue).")
                print()
                print("  Manual download steps:")
                print(f"  1. Download from: {DOWNLOAD_URL}")
                print(f"     (use a browser, download manager, or VPN)")
                print(f"  2. Save to: {jsonl_path}")
                print(f"  3. Re-run: python scripts/prepare_thucnews.py")
                print("=" * 60)
                sys.exit(1)

    # ---- determine sampling ----
    if args.full:
        max_per_class = None
        print("Using ALL samples (--full)")
    elif args.max_per_class is not None:
        max_per_class = args.max_per_class
        print(f"Max per class: {max_per_class}")
    else:
        max_per_class = DEFAULT_MAX_PER_CLASS
        print(f"Max per class: {max_per_class} (default). Use --full for all data.")

    # ---- load & validate ----
    print("\nLoading JSONL ...")
    all_rows = list(read_jsonl(jsonl_path))
    print(f"Total samples: {len(all_rows)}")

    # Detect field names (dataset may use different keys)
    sample = all_rows[0]
    print(f"Sample keys: {list(sample.keys())}")

    # text field
    if "text" in sample:
        text_key = "text"
    elif "sentence" in sample:
        text_key = "sentence"
    else:
        print(f"Error: no text field found. Keys: {list(sample.keys())}")
        sys.exit(1)

    # label field: prefer integer columns (label_id, label), fall back to string label
    if "label_id" in sample:
        label_key = "label_id"
        raw_label = sample[label_key]
        label_is_int = isinstance(raw_label, int) or (isinstance(raw_label, str) and raw_label.isdigit())
    elif "label" in sample:
        label_key = "label"
        raw_label = sample[label_key]
        label_is_int = isinstance(raw_label, int) or (isinstance(raw_label, str) and raw_label.isdigit())
    else:
        print(f"Error: no label field found. Keys: {list(sample.keys())}")
        sys.exit(1)

    # Build label name → id mapping from data
    label_name_to_id: dict[str, int] = {}
    if not label_is_int:
        # label is string like '体育', collect unique labels and assign IDs
        label_names = sorted(set(r[label_key] for r in all_rows))
        label_name_to_id = {name: i for i, name in enumerate(label_names)}
        print(f"Label mapping: {label_name_to_id}")

    # Normalize to {"text": ..., "label": int}
    rows = []
    for r in all_rows:
        if label_is_int:
            lbl = int(r[label_key])
        else:
            lbl = label_name_to_id[r[label_key]]
        rows.append({"text": r[text_key], "label": lbl})

    label_counts = Counter(r["label"] for r in rows)
    print(f"Labels found: {dict(sorted(label_counts.items()))}")

    # ---- split ----
    print("\nSplitting 80/10/10 ...")
    splits = train_valid_test_split(rows, SEED)

    # ---- sample & write ----
    print()
    total = 0
    for split_name, filename in SPLIT_MAP.items():
        split_max = None if max_per_class is None else max(int(max_per_class * {
            "train": 1.0, "validation": 0.1, "test": 0.1,
        }[split_name]), 200)

        split_rows = splits[split_name]
        before = len(split_rows)
        split_rows = sample_per_class(split_rows, split_max)
        after = len(split_rows)

        path = os.path.join(DATA_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        cnt = Counter(r["label"] for r in split_rows)
        print(f"  {split_name:>10}: {before:>6} → {after:>6} 分布={dict(sorted(cnt.items()))}")
        print(f"              → {path}")
        total += after

    # ---- save label mapping ----
    if label_is_int:
        # labels are ints, build name mapping from data if 'label' column has string names
        if "label" in sample and not (isinstance(sample["label"], int) or sample["label"].isdigit()):
            unique_labels = sorted(set(r["label"] for r in all_rows))
            label2id = {name: i for i, name in enumerate(unique_labels)}
        else:
            num_labels = max(r["label"] for r in rows) + 1
            label2id = {str(i): i for i in range(num_labels)}
    else:
        label2id = label_name_to_id

    with open(os.path.join(DATA_DIR, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({v: k for k, v in label2id.items()}, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {total} samples saved to {DATA_DIR}/")
    print()
    print("Next step:")
    print("  python -m src.train --config configs/bert_thucnews.yaml")


if __name__ == "__main__":
    main()
