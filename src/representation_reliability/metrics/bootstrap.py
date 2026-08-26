"""Bootstrap confidence intervals for held-out metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI of ``metric_fn(y_true, scores)``.

    Resamples evaluation rows with replacement. If a resample is degenerate
    (single class) the metric value for that resample is dropped.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    point = metric_fn(y_true, scores)
    stats: list[float] = []
    for _ in range(int(n_bootstraps)):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            val = metric_fn(yt, ys)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        stats.append(float(val))
    alpha = 1.0 - confidence_level
    if stats:
        lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        mean = float(np.mean(stats))
        std = float(np.std(stats, ddof=1)) if len(stats) > 1 else 0.0
    else:
        lo = hi = mean = std = float("nan")
    return {
        "point_estimate": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_mean": mean,
        "bootstrap_std": std,
        "n_bootstrap_kept": len(stats),
        "confidence_level": confidence_level,
    }
