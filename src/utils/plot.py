from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None,
    out_path: str = "assets/confusion_matrix.png",
    normalize: bool = True,
    figsize=(10, 8),
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
