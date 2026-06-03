from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report


def build_classification_report(
    y_true: List[int], y_pred: List[int], target_names: List[str] | None = None
) -> Dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)
