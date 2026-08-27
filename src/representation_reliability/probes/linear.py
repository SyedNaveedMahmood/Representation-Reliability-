"""Linear (logistic-regression) probing with validation-only C selection.

Protocol guarantees:
- standardization is fitted on training data only;
- regularization strength ``C`` is selected on the *validation* split only;
- evaluation happens after hyperparameter selection;
- coefficients, scaler statistics and chosen C are persisted.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..metrics.decoding import classification_metrics


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: Sequence[float],
    seed: int = 0,
    class_weight: str | None = "balanced",
    standardize: bool = True,
) -> dict[str, Any]:
    """Fit scaler + logistic regression; select C on validation AUROC."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_train = np.asarray(y_train).astype(int)
    y_val = np.asarray(y_val).astype(int)

    if not set(np.unique(y_train)).issuperset({0, 1}) or len(np.unique(y_train)) < 2:
        raise ValueError("training labels must contain both classes")
    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "validation labels are single-class; cannot select C — check splits"
        )

    scaler = StandardScaler(with_mean=True, with_std=True)
    if standardize:
        # Fitted on TRAIN only. Constant dimensions get scale 1 to avoid div0.
        Ztr = scaler.fit_transform(X_train)
        stds = scaler.scale_.copy()
        safe = stds > 0
        scaler.scale_[~safe] = 1.0
        Ztr[:, ~safe] = Ztr[:, ~safe]  # already centered; zero-var stays 0
        Zval = (scaler.transform(X_val))
        Zval[:, ~safe] = 0.0
        scaler_stats = {"mean": scaler.mean_, "scale": scaler.scale_}
        train_for_fit, val_for_select = Ztr, Zval
    else:
        scaler_stats = {"mean": None, "scale": None}
        train_for_fit, val_for_select = X_train, X_val

    from sklearn.metrics import roc_auc_score

    best_c, best_auroc = None, -np.inf
    per_c_validation_scores: dict[str, float] = {}
    for c in sorted(float(x) for x in c_grid):
        clf = LogisticRegression(
            C=c, class_weight=class_weight, max_iter=2000,
            solver="lbfgs", random_state=seed,
        )
        clf.fit(train_for_fit, y_train)
        scores = clf.decision_function(val_for_select)
        auroc = float(roc_auc_score(y_val, scores))
        per_c_validation_scores[str(c)] = auroc
        if auroc > best_auroc:
            best_c, best_auroc = c, auroc

    final_clf = LogisticRegression(
        C=float(best_c), class_weight=class_weight, max_iter=2000,
        solver="lbfgs", random_state=seed,
    )
    final_clf.fit(train_for_fit, y_train)

    return {
        "classifier": final_clf,
        "scaler_mean": scaler_stats["mean"],
        "scaler_scale": scaler_stats["scale"],
        "chosen_C": float(best_c),
        "validation_auroc_by_C": per_c_validation_scores,
        "validation_auroc_best": best_auroc,
        "standardized": bool(standardize),
    }


def transform_features(fit_result: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Apply training-fitted standardization to new features."""
    X = np.asarray(X, dtype=np.float64)
    mean, scale = fit_result["scaler_mean"], fit_result["scaler_scale"]
    if fit_result.get("standardized") and mean is not None:
        return (X - mean) / scale
    return X


def raw_probe_direction(fit_result: dict[str, Any]) -> np.ndarray:
    """Recover the classifier direction in unstandardized hidden coordinates."""
    coef = np.asarray(fit_result["classifier"].coef_[0], dtype=np.float64)
    scale = fit_result.get("scaler_scale")
    if fit_result.get("standardized") and scale is not None:
        return coef / np.asarray(scale, dtype=np.float64)
    return coef.copy()


def evaluate_probe(
    fit_result: dict[str, Any],
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, Any]:
    """Metrics on held-out evaluation rows (after model selection)."""
    Xt = transform_features(fit_result, X_eval)
    scores = fit_result["classifier"].decision_function(Xt)
    metrics = classification_metrics(np.asarray(y_eval), scores)
    return metrics


def save_probe(fit_result: dict[str, Any], path: str | Path) -> Path:
    """Persist probe artifacts as a portable .npz archive (no pickled objects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "coef": fit_result["classifier"].coef_,
        "intercept": fit_result["classifier"].intercept_,
        "chosen_C": np.float64(fit_result["chosen_C"]),
        "standardized": int(bool(fit_result["standardized"])),
    }
    for key in ("scaler_mean", "scaler_scale"):
        value = fit_result[key]
        payload[key] = value if value is not None else np.array([np.nan])
    np.savez(path, **payload)
    return path


def load_probe(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path)) as z:
        return {
            "coef": z["coef"],
            "intercept": z["intercept"],
            "chosen_C": float(z["chosen_C"]),
            "standardized": bool(int(z["standardized"])),
            "scaler_mean": z["scaler_mean"],
            "scaler_scale": z["scaler_scale"],
            "scores": lambda X: X @ z["coef"][0] + z["intercept"][0],
        }


# ---------------------------------------------------------------------------
# Random-feature baseline (optional control)


def random_feature_baseline(
    n_rows: int,
    dim: int,
    seed: int,
) -> np.ndarray:
    """Gaussian features matched to hidden-state dimensionality."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_rows, dim))


def shuffle_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic label permutation used by the random-label control."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return np.asarray(y)[perm]


def randomized_control_labels(
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Clean random-label null: independent permutations of train AND val.

    Returns ``(y_train_shuffled, y_val_shuffled, (train_seed, val_seed))``.
    The validation permutation is derived independently from the train
    permutation so hyperparameter selection happens fully under the null;
    evaluation still uses the real, untouched discovery-test labels.
    """
    train_seed = int(seed) * 1_000_003 + 17
    val_seed = int(seed) * 1_000_033 + 91
    return (
        shuffle_labels(y_train, seed=train_seed),
        shuffle_labels(y_val, seed=val_seed),
        (train_seed, val_seed),
    )
