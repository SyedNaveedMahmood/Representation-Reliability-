"""Metrics subsystem."""

from .bootstrap import bootstrap_ci
from .decoding import (
    balanced_accuracy_score,  # noqa: F401 re-export convenience
    class_balance,
    classification_metrics,
    majority_baseline_accuracy,
)

__all__ = [
    "bootstrap_ci",
    "class_balance",
    "classification_metrics",
    "majority_baseline_accuracy",
]
