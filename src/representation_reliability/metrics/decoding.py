"""Decoding metrics for probe evaluation.

A high AUROC means a held-out linear decoder separates the target variable
from hidden states; it establishes *decodability*, not causal use.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def majority_baseline_accuracy(y: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return float(counts.max() / len(y)) if len(y) else float("nan")


def class_balance(y: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(y).astype(int)
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    return {
        "n": len(arr),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "frac_positive": (n_pos / len(arr)) if len(arr) else None,
    }


def classification_metrics(
    y_true: np.ndarray, scores_or_preds: np.ndarray
) -> dict[str, Any]:
    """AUROC / AUPRC / balanced accuracy for binary labels.

    ``scores_or_preds`` may be continuous decision scores (preferred).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores_or_preds, dtype=np.float64)
    out: dict[str, Any] = {
        "auroc": None,
        "auprc": None,
        "balanced_accuracy": None,
        "n_eval": len(y_true),
    }
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        # Degenerate evaluation set: record explicitly instead of guessing.
        out["note"] = "degenerate evaluation set (single class or empty)"
        if len(y_true):
            preds = (scores >= 0).astype(int)
            out["balanced_accuracy"] = float(
                balanced_accuracy_score(y_true, preds)
            )
        return out
    out["auroc"] = float(roc_auc_score(y_true, scores))
    out["auprc"] = float(average_precision_score(y_true, scores))
    preds = (scores >= 0).astype(int)
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, preds))
    return out
