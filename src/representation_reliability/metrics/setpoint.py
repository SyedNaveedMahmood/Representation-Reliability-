"""Validation-only targets and causal summaries for E01B-1 setpoints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .causal import cluster_bootstrap_mean_ci

GRID_QUANTILES: tuple[tuple[str, float], ...] = (
    ("Q05", 0.05),
    ("Q25", 0.25),
    ("Q50", 0.50),
    ("Q75", 0.75),
    ("Q95", 0.95),
)


def validation_setpoint_targets(
    coordinates: Sequence[float],
    labels: Sequence[int],
    native_margins: Sequence[float],
) -> dict[str, Any]:
    """Construct the frozen class medians, pooled grid, and validation scales."""
    q = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    margins = np.asarray(native_margins, dtype=np.float64)
    if len(q) == 0 or len(q) != len(y) or len(q) != len(margins):
        raise ValueError("validation coordinates, labels, and margins must align")
    if not np.isfinite(q).all() or not np.isfinite(margins).all():
        raise ValueError("validation references must be finite")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("validation labels must contain exactly classes 0 and 1")
    sigma_q = float(np.std(q, ddof=1))
    sigma_margin = float(np.std(margins, ddof=1))
    if sigma_q <= 1e-12 or sigma_margin <= 1e-12:
        raise ValueError("validation reference variance is zero or degenerate")
    q0 = float(np.median(q[y == 0]))
    q1 = float(np.median(q[y == 1]))
    if not q1 > q0:
        raise RuntimeError(
            "probe orientation invariant failed: validation class-1 median "
            "must exceed class-0 median"
        )
    grid = {
        name: float(np.quantile(q, quantile, method="linear"))
        for name, quantile in GRID_QUANTILES
    }
    if any(a >= b for a, b in zip(grid.values(), list(grid.values())[1:])):
        raise RuntimeError("validation quantile targets are not strictly increasing")
    return {
        "q0_star": q0,
        "q1_star": q1,
        "grid": grid,
        "sigma_q_validation": sigma_q,
        "sigma_margin_validation": sigma_margin,
        "n_validation": len(q),
        "n_validation_label0": int((y == 0).sum()),
        "n_validation_label1": int((y == 1).sum()),
    }


def validation_standardized_effect(
    delta_q: float,
    oriented_delta_margin: float,
    *,
    sigma_q_validation: float,
    sigma_margin_validation: float,
) -> dict[str, float]:
    """Compute signed Δq_z, oriented Δm_z, and guarded exploratory kappa_z."""
    sigma_q = float(sigma_q_validation)
    sigma_m = float(sigma_margin_validation)
    if sigma_q <= 1e-12 or sigma_m <= 1e-12:
        raise ValueError("validation standardization scale is zero or degenerate")
    delta_q_z = float(delta_q) / sigma_q
    delta_m_z = float(oriented_delta_margin) / sigma_m
    kappa_z = delta_m_z / abs(delta_q_z) if abs(delta_q_z) > 1e-12 else float("nan")
    return {
        "delta_q_z": float(delta_q_z),
        "delta_m_z": float(delta_m_z),
        "kappa_z": float(kappa_z),
    }


def within_base_centered_slope(
    rows: pd.DataFrame,
    *,
    target_col: str = "q_target",
    margin_col: str = "intervened_yes_no_margin",
    base_col: str = "base_sample_id",
) -> float:
    """Fixed-effect slope after centering target and margin within each base."""
    required = {target_col, margin_col, base_col}
    missing = required - set(rows.columns)
    if missing or len(rows) == 0:
        raise ValueError(f"grid rows missing required columns: {sorted(missing)}")
    working = rows[[base_col, target_col, margin_col]].copy()
    if not np.isfinite(working[[target_col, margin_col]].to_numpy(float)).all():
        raise ValueError("grid slope inputs must be finite")
    x = working[target_col] - working.groupby(base_col)[target_col].transform("mean")
    y = working[margin_col] - working.groupby(base_col)[margin_col].transform("mean")
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        raise ValueError("within-base target variation is zero")
    return float(np.dot(x, y) / denominator)


def grid_example_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-base Spearman and exact monotonicity over the five frozen targets."""
    required = {
        "base_sample_id",
        "pair_id",
        "target_name",
        "q_target",
        "intervened_yes_no_margin",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"grid rows missing required columns: {sorted(missing)}")
    expected_names = {name for name, _ in GRID_QUANTILES}
    output: list[dict[str, Any]] = []
    for base_id, block in rows.groupby("base_sample_id", sort=True):
        if len(block) != len(GRID_QUANTILES) or set(block["target_name"]) != expected_names:
            raise RuntimeError(f"base {base_id} does not have the complete frozen grid")
        ordered = block.sort_values("q_target")
        q = ordered["q_target"].to_numpy(float)
        margins = ordered["intervened_yes_no_margin"].to_numpy(float)
        if np.any(np.diff(q) <= 0):
            raise RuntimeError(f"base {base_id} has non-increasing q targets")
        # A flat output trajectory has no rank association. SciPy reports NaN
        # for this degenerate case, but zero is the scientifically meaningful
        # finite value for an absent dose response.
        rho = (
            0.0
            if float(np.ptp(margins)) <= 1e-12
            else float(spearmanr(q, margins).statistic)
        )
        output.append(
            {
                "base_sample_id": str(base_id),
                "pair_id": str(ordered["pair_id"].iloc[0]),
                "spearman": rho,
                "spearman_positive": bool(rho > 0.0),
                "spearman_ge_0_8": bool(rho >= 0.8),
                "monotonic_nondecreasing": bool(np.all(np.diff(margins) >= -1e-12)),
                "per_base_slope": within_base_centered_slope(ordered),
            }
        )
    return pd.DataFrame(output)


def _cluster_bootstrap_statistic(
    rows: pd.DataFrame,
    statistic,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    cluster_text = rows["pair_id"].astype(str).to_numpy()
    unique = np.asarray(sorted(set(cluster_text)), dtype=object)
    indices = [np.flatnonzero(cluster_text == cluster) for cluster in unique]
    rng = np.random.default_rng(int(seed))
    draws = np.empty(max(1, int(n_bootstraps)), dtype=np.float64)
    for index in range(len(draws)):
        chosen = rng.integers(0, len(unique), size=len(unique))
        sample_indices = np.concatenate([indices[position] for position in chosen])
        draws[index] = float(statistic(rows.iloc[sample_indices]))
    if not np.isfinite(draws).all():
        raise ValueError("pair-cluster bootstrap produced non-finite statistics")
    tail = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize_grid_response(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Target-level raw evidence summaries and clustered within-base slope."""
    examples = grid_example_metrics(rows)
    aggregates: list[dict[str, Any]] = []
    target_order = {name: index for index, (name, _q) in enumerate(GRID_QUANTILES)}
    for name, block in rows.groupby("target_name", sort=False):
        ci = cluster_bootstrap_mean_ci(
            block["intervened_yes_no_margin"].to_numpy(float),
            block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=seed + target_order[str(name)] * 17,
        )
        aggregates.append(
            {
                "target_name": str(name),
                "q_target": float(block["q_target"].iloc[0]),
                "n_rows": len(block),
                "n_pairs": int(block["pair_id"].nunique()),
                "mean_native_margin": ci["mean"],
                "median_native_margin": float(block["intervened_yes_no_margin"].median()),
                "margin_ci_low": ci["ci_low"],
                "margin_ci_high": ci["ci_high"],
            }
        )
    target_metrics = pd.DataFrame(aggregates).sort_values("q_target").reset_index(drop=True)
    point_slope = within_base_centered_slope(rows)
    slope_low, slope_high = _cluster_bootstrap_statistic(
        rows,
        within_base_centered_slope,
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed + 701,
    )
    summary = {
        "median_within_base_spearman": float(examples["spearman"].median()),
        "fraction_spearman_positive": float(examples["spearman_positive"].mean()),
        "fraction_spearman_ge_0_8": float(examples["spearman_ge_0_8"].mean()),
        "fraction_monotonic_nondecreasing": float(
            examples["monotonic_nondecreasing"].mean()
        ),
        "within_base_centered_slope": point_slope,
        "within_base_centered_slope_ci_low": slope_low,
        "within_base_centered_slope_ci_high": slope_high,
    }
    return target_metrics, examples, summary


def aggregate_setpoint_rows(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Aggregate binary-target conditions while preserving control seeds."""
    required = {
        "condition",
        "target_name",
        "direction_seed",
        "pair_id",
        "delta_margin_toward_target",
        "expected_target_after",
        "actual_target_flip",
        "delta_q_z",
        "delta_m_z",
        "kappa_z",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"setpoint rows missing required columns: {sorted(missing)}")
    binary = rows[rows["target_label"].notna()].copy()
    output: list[dict[str, Any]] = []
    for index, (key, block) in enumerate(
        binary.groupby(["condition", "target_name", "direction_seed"], dropna=False, sort=True)
    ):
        condition, target_name, direction_seed = key
        effect = cluster_bootstrap_mean_ci(
            block["delta_margin_toward_target"].to_numpy(float),
            block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=seed + index * 31,
        )
        output.append(
            {
                "condition": condition,
                "target_name": target_name,
                "direction_seed": direction_seed,
                "n_rows": len(block),
                "n_pairs": int(block["pair_id"].nunique()),
                "mean_delta_margin_toward_target": effect["mean"],
                "median_delta_margin_toward_target": float(
                    block["delta_margin_toward_target"].median()
                ),
                "effect_ci_low": effect["ci_low"],
                "effect_ci_high": effect["ci_high"],
                "actual_target_flip_rate": float(block["actual_target_flip"].mean()),
                "expected_target_rate_after": float(block["expected_target_after"].mean()),
                "mean_delta_q_z": float(block["delta_q_z"].mean()),
                "mean_oriented_delta_q_z": float(block["oriented_delta_q_z"].mean()),
                "mean_delta_m_z": float(block["delta_m_z"].mean()),
                "mean_kappa_z": float(block["kappa_z"].mean(skipna=True)),
            }
        )
    return pd.DataFrame(output)


def paired_setpoint_control_contrast(
    rows: pd.DataFrame,
    *,
    control: str,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Opposite-median minus seed-averaged norm-matched control."""
    treatment = rows[rows["condition"] == "source_free_opposite_class_median"]
    comparison = rows[rows["condition"] == control]
    keys = ["base_sample_id", "pair_id"]
    left = treatment.groupby(keys, as_index=False)["delta_margin_toward_target"].mean()
    right = comparison.groupby(keys, as_index=False)["delta_margin_toward_target"].mean()
    merged = left.merge(right, on=keys, validate="one_to_one", suffixes=("_t", "_c"))
    if len(merged) != len(left) or len(left) == 0:
        raise RuntimeError(f"paired setpoint contrast lost rows for {control}")
    difference = (
        merged["delta_margin_toward_target_t"].to_numpy(float)
        - merged["delta_margin_toward_target_c"].to_numpy(float)
    )
    ci = cluster_bootstrap_mean_ci(
        difference,
        merged["pair_id"].astype(str).tolist(),
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "treatment": "source_free_opposite_class_median",
        "control": control,
        "mean_difference": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n_rows": ci["n_rows"],
        "n_pairs": ci["n_clusters"],
    }
