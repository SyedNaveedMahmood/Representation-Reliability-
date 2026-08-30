"""Temporal decay curves and causal/representation half-lives for E15.

Frozen by ``docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_PROTOCOL.md`` sections 5 and 8.1.

Two rules from the design are enforced here rather than left to the caller:

* a half-life is reported **only** when the relative curve is sufficiently smooth
  and monotone, otherwise the summary is refused and the raw curve stands;
* a curve that never reaches the threshold inside the grid yields a
  **right-censored** result, never an extrapolated number.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .causal import cluster_bootstrap_mean_ci

# Frozen smoothness criterion (protocol 8.1).
MAX_SPEARMAN_RHO = -0.70
MAX_UPWARD_STEP = 0.15
HALF_LIFE_THRESHOLD = 0.5


def relative_curve(
    horizons: Sequence[int],
    values: Sequence[float],
    *,
    floor: float = 0.0,
) -> dict[str, Any]:
    """Normalise a decay curve by its value at the earliest horizon.

    ``floor`` is subtracted before normalising, which is how the decodability
    curve is normalised against chance (AUROC floor 0.5).
    """
    ks = [int(k) for k in horizons]
    ys = np.asarray(values, dtype=np.float64)
    if len(ks) != len(ys) or len(ks) < 2:
        raise ValueError("a decay curve needs at least two aligned horizon points")
    if list(ks) != sorted(ks) or len(set(ks)) != len(ks):
        raise ValueError("horizons must be strictly increasing and unique")
    if not np.isfinite(ys).all():
        raise ValueError("curve values must be finite")
    base = float(ys[0]) - float(floor)
    if abs(base) <= 1e-12:
        raise ValueError("baseline horizon value is degenerate; cannot normalise")
    rel = (ys - float(floor)) / base
    return {
        "horizons": ks,
        "values": [float(v) for v in ys],
        "baseline": base,
        "floor": float(floor),
        "relative": [float(v) for v in rel],
        "baseline_positive": bool(base > 0.0),
    }


def curve_smoothness(horizons: Sequence[int], relative: Sequence[float]) -> dict[str, Any]:
    """Frozen monotonicity diagnostics for a normalised decay curve."""
    ks = np.asarray([int(k) for k in horizons], dtype=np.float64)
    rel = np.asarray(relative, dtype=np.float64)
    if len(ks) != len(rel) or len(ks) < 2:
        raise ValueError("smoothness needs at least two aligned points")
    if float(np.ptp(rel)) <= 1e-12:
        rho = 0.0
    else:
        rho = float(spearmanr(ks, rel).statistic)
    steps = np.diff(rel)
    max_up = float(np.max(steps)) if len(steps) else 0.0
    return {
        "spearman_rho": rho,
        "max_upward_step": max_up,
        "monotone_enough": bool(rho <= MAX_SPEARMAN_RHO and max_up <= MAX_UPWARD_STEP),
        "criterion": {
            "max_spearman_rho": MAX_SPEARMAN_RHO,
            "max_upward_step": MAX_UPWARD_STEP,
        },
    }


def half_life(
    horizons: Sequence[int],
    relative: Sequence[float],
    *,
    baseline_positive: bool,
    baseline_ci_excludes_zero: bool,
    threshold: float = HALF_LIFE_THRESHOLD,
) -> dict[str, Any]:
    """Interpolated half-life, or an explicit refusal.

    Returns ``status`` in ``{estimated, right_censored, not_estimable}``. The
    design allows a half-life summary only for a sufficiently smooth curve whose
    baseline effect is itself real, so both preconditions are inputs here rather
    than assumptions.
    """
    ks = [int(k) for k in horizons]
    rel = np.asarray(relative, dtype=np.float64)
    smooth = curve_smoothness(ks, rel)
    reasons: list[str] = []
    if not baseline_positive:
        reasons.append("baseline_at_k0_not_positive")
    if not baseline_ci_excludes_zero:
        reasons.append("baseline_ci_includes_zero")
    if not smooth["monotone_enough"]:
        reasons.append("curve_not_sufficiently_monotone")
    if reasons:
        return {
            "status": "not_estimable",
            "value": None,
            "reasons": reasons,
            "smoothness": smooth,
            "threshold": float(threshold),
        }
    thr = float(threshold)
    for index in range(1, len(ks)):
        previous, current = float(rel[index - 1]), float(rel[index])
        if current <= thr:
            if previous <= thr:
                # Already at or below threshold before this point: the crossing
                # happened at or before the previous grid horizon.
                value = float(ks[index - 1])
            else:
                span = previous - current
                frac = (previous - thr) / span if span > 1e-12 else 0.0
                value = float(ks[index - 1]) + frac * float(ks[index] - ks[index - 1])
            return {
                "status": "estimated",
                "value": float(value),
                "bracket": [int(ks[index - 1]), int(ks[index])],
                "reasons": [],
                "smoothness": smooth,
                "threshold": thr,
            }
    return {
        "status": "right_censored",
        "value": None,
        "censored_at": int(ks[-1]),
        "final_relative": float(rel[-1]),
        "reasons": ["curve_never_reached_threshold_inside_grid"],
        "smoothness": smooth,
        "threshold": thr,
    }


def horizon_condition_summary(
    rows: pd.DataFrame,
    *,
    value_col: str = "delta_margin_toward_expected",
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Cluster-bootstrapped mean effect per (horizon, condition)."""
    required = {"horizon", "condition", "pair_id", value_col}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"intervention rows missing columns: {sorted(missing)}")
    out: list[dict[str, Any]] = []
    grouped = rows.groupby(["horizon", "condition"], dropna=False, sort=True)
    for offset, (key, block) in enumerate(grouped):
        horizon, condition = key
        ci = cluster_bootstrap_mean_ci(
            block[value_col].to_numpy(float),
            block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=int(seed) + offset * 13,
        )
        record = {
            "horizon": int(horizon),
            "condition": str(condition),
            "mean_effect": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "n_rows": ci["n_rows"],
            "n_pairs": ci["n_clusters"],
        }
        for extra in ("delta_norm", "activation_norm", "delta_over_activation_norm"):
            if extra in block.columns:
                record[f"mean_{extra}"] = float(block[extra].mean())
        out.append(record)
    return pd.DataFrame(out).sort_values(["horizon", "condition"]).reset_index(drop=True)


def paired_horizon_contrast(
    rows: pd.DataFrame,
    *,
    horizon: int,
    treatment: str,
    control: str,
    value_col: str = "delta_margin_toward_expected",
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Per-episode paired treatment-minus-control contrast at one horizon.

    Control arms with several random directions are averaged within each episode
    first, so a control's seed count cannot inflate its cluster weight.
    """
    block = rows[rows["horizon"].astype(int) == int(horizon)]
    treat = block[block["condition"].astype(str) == str(treatment)]
    ctrl = block[block["condition"].astype(str) == str(control)]
    if len(treat) == 0 or len(ctrl) == 0:
        raise ValueError(
            f"missing rows for {treatment} vs {control} at horizon {horizon}"
        )
    keys = ["base_sample_id", "pair_id"]
    t = treat.groupby(keys, as_index=False)[value_col].mean()
    c = ctrl.groupby(keys, as_index=False)[value_col].mean()
    merged = t.merge(c, on=keys, suffixes=("_t", "_c"), how="inner")
    if len(merged) != len(t):
        raise RuntimeError("paired horizon contrast lost treatment episodes")
    diff = (
        merged[f"{value_col}_t"].to_numpy(float)
        - merged[f"{value_col}_c"].to_numpy(float)
    )
    ci = cluster_bootstrap_mean_ci(
        diff,
        merged["pair_id"].astype(str).tolist(),
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=int(seed),
    )
    return {
        "horizon": int(horizon),
        "treatment": str(treatment),
        "control": str(control),
        "mean_difference": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n_rows": ci["n_rows"],
        "n_pairs": ci["n_clusters"],
        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
    }


def shuffled_decision_null(
    signed_margin_change: Sequence[float],
    expected_labels: Sequence[int],
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Diagnostic null that permutes the future-decision mapping (design 8).

    ``signed_margin_change`` is the *unoriented* Yes-minus-No margin change; the
    orientation ``+1`` / ``-1`` comes from the expected counterfactual label. The
    null permutes which episode's expected label is applied to which episode's
    observed change, destroying the state-to-decision correspondence while
    preserving both marginal distributions.
    """
    raw = np.asarray(signed_margin_change, dtype=np.float64)
    labels = np.asarray(expected_labels, dtype=int)
    if len(raw) != len(labels) or len(raw) == 0:
        raise ValueError("null inputs must be aligned and non-empty")
    if not np.isfinite(raw).all():
        raise ValueError("null inputs must be finite")
    orientation = np.where(labels == 1, 1.0, -1.0)
    observed = float(np.mean(raw * orientation))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_permutations), dtype=np.float64)
    for index in range(int(n_permutations)):
        draws[index] = float(np.mean(raw * rng.permutation(orientation)))
    exceed = int(np.sum(np.abs(draws) >= abs(observed)))
    return {
        "observed": observed,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
        "null_abs_p": float((exceed + 1) / (int(n_permutations) + 1)),
        "n_permutations": int(n_permutations),
        "exceeds_null": bool((exceed + 1) / (int(n_permutations) + 1) < 0.05),
    }


def curve_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    horizon_col: str,
    value_col: str,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Resample whole horizon curves as units.

    A base episode is rendered at every horizon, so the episode - not the row -
    is the independent unit and its entire trajectory must be resampled
    together. Returns the per-horizon point estimates and CIs plus the raw
    ``draws`` matrix ``[n_bootstraps, n_horizons]``, so that contrasts between
    horizons or between components can be computed on the *same* resampling and
    inherit its dependence structure.
    """
    required = {cluster_col, horizon_col, value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"curve bootstrap missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("curve bootstrap received no rows")
    horizons = sorted(int(k) for k in frame[horizon_col].unique())
    clusters = sorted(map(str, frame[cluster_col].astype(str).unique()))
    if len(clusters) < 2:
        raise ValueError("curve bootstrap needs at least two clusters")

    # Per cluster, the per-horizon sum and count, so a resampled curve is an
    # exact mean over the resampled clusters rather than a mean of means.
    index = {int(k): position for position, k in enumerate(horizons)}
    sums = np.zeros((len(clusters), len(horizons)), dtype=np.float64)
    counts = np.zeros((len(clusters), len(horizons)), dtype=np.float64)
    cluster_index = {name: position for position, name in enumerate(clusters)}
    for cluster, horizon, value in zip(
        frame[cluster_col].astype(str), frame[horizon_col].astype(int),
        frame[value_col].astype(float),
    ):
        row = cluster_index[str(cluster)]
        column = index[int(horizon)]
        sums[row, column] += float(value)
        counts[row, column] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.where(
            counts.sum(axis=0) > 0, sums.sum(axis=0) / counts.sum(axis=0), np.nan
        )

    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_bootstraps), len(horizons)), dtype=np.float64)
    n_clusters = len(clusters)
    for draw in range(int(n_bootstraps)):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        numerator = sums[picked].sum(axis=0)
        denominator = counts[picked].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[draw] = np.where(denominator > 0, numerator / denominator, np.nan)

    alpha = (1.0 - float(confidence_level)) / 2.0
    low = np.nanquantile(draws, alpha, axis=0)
    high = np.nanquantile(draws, 1.0 - alpha, axis=0)
    return {
        "horizons": horizons,
        "point": [float(v) for v in point],
        "ci_low": [float(v) for v in low],
        "ci_high": [float(v) for v in high],
        "n_clusters": n_clusters,
        "draws": draws,
    }


def bootstrap_two_sided_p(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a contrast being different from zero."""
    values = np.asarray(draws, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    below = float(np.mean(values <= 0.0))
    above = float(np.mean(values >= 0.0))
    return float(min(1.0, 2.0 * min(below, above)))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over a named family."""
    items = [(name, float(p)) for name, p in p_values.items() if np.isfinite(p)]
    if not items:
        return {name: float("nan") for name in p_values}
    items.sort(key=lambda pair: pair[1])
    total = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        value = min(1.0, (total - rank) * p)
        running = max(running, value)
        adjusted[name] = running
    for name, p in p_values.items():
        if not np.isfinite(p):
            adjusted[name] = float("nan")
    return adjusted
