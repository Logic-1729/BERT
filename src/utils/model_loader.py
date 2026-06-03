"""Model loading with ModelScope fallback for Chinese users.

Usage:
    from src.utils.model_loader import load_model, load_tokenizer
    model = load_model("hfl/chinese-macbert-base", num_labels=10)
    tokenizer = load_tokenizer("hfl/chinese-macbert-base")
"""

from __future__ import annotations

import os
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _try_download_modelscope(model_id: str) -> str | None:
    """Try to download model from ModelScope. Returns local path or None on failure."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        return None
    try:
        path = snapshot_download(model_id)
        # strip revision suffix if any
        if "@" in path and path.count("@") == 1:
            path = path.rsplit("@", 1)[0]
        return path
    except Exception:
        return None


def _try_download_huggingface(model_id: str) -> str | None:
    """Try to download model from HuggingFace. Returns local path or None on failure."""
    try:
        from transformers import AutoConfig
        # Just try to fetch the config to verify accessibility
        AutoConfig.from_pretrained(model_id)
        return model_id  # HF can load directly from ID
    except Exception:
        return None


def resolve_model_path(model_id: str, prefer_modelscope: bool = True) -> str:
    """Resolve a model ID to a loadable path.

    When prefer_modelscope=True (default), tries ModelScope first, then HF.
    When prefer_modelscope=False, uses HF directly (respecting HF_ENDPOINT if set).
    """
    # If it's already a local path, use it directly
    if os.path.isdir(model_id):
        return model_id

    if prefer_modelscope:
        path = _try_download_modelscope(model_id)
        if path is not None:
            print(f"[model] Loaded from ModelScope: {model_id}")
            return path

    # Fall back to HuggingFace
    print(f"[model] Loading from HuggingFace: {model_id}")
    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"[model]   using mirror: {hf_endpoint}")
    return model_id


def load_tokenizer(model_id: str, **kwargs):
    """Load tokenizer with ModelScope-first strategy."""
    path = resolve_model_path(model_id)
    use_fast = kwargs.pop("use_fast", True)
    return AutoTokenizer.from_pretrained(path, use_fast=use_fast, **kwargs)


def load_model(model_id: str, num_labels: int | None = None, **kwargs):
    """Load sequence classification model with ModelScope-first strategy."""
    path = resolve_model_path(model_id)
    if num_labels is not None:
        kwargs["num_labels"] = num_labels
    return AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
