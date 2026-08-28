"""Causal-effect summaries for intervention sweeps.

All discovery statistics operate on directed examples but bootstrap at the
matched-pair cluster level so the two directions from one counterfactual pair
are never treated as independent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def margin_toward_label(margin_yes_minus_no: float, label: int) -> float:
    """Orient a Yes-minus-No margin so larger means more support for ``label``."""
    if int(label) not in (0, 1):
        raise ValueError("binary label must be 0 or 1")
    return float(margin_yes_minus_no) if int(label) == 1 else -float(margin_yes_minus_no)


def counterfactual_outcome(
    prediction_before: int, prediction_after: int, expected_label: int
) -> dict[str, int]:
    """Distinguish target accuracy after intervention from an actual flip."""
    values = (prediction_before, prediction_after, expected_label)
    if any(int(value) not in (0, 1) for value in values):
        raise ValueError("predictions and expected_label must be binary")
    hit_after = int(int(prediction_after) == int(expected_label))
    flip = int(
        int(prediction_before) != int(expected_label)
        and int(prediction_after) == int(expected_label)
    )
    return {
        "expected_label_after": hit_after,
        "counterfactual_flip": flip,
    }


def cluster_bootstrap_mean_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    """Mean and cluster-bootstrap CI, resampling matched pairs with replacement."""
    x = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=object)
    if len(x) != len(clusters) or len(x) == 0:
        raise ValueError("values and cluster_ids must be equally sized and non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("bootstrap values must be finite")
    cluster_text = clusters.astype(str)
    unique = np.asarray(sorted(set(map(str, clusters))), dtype=object)
    by_cluster = {cid: x[cluster_text == cid] for cid in unique}
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(max(1, int(n_bootstraps))):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_cluster[cid] for cid in sampled])
        draws.append(float(rows.mean()))
    alpha = (1.0 - float(confidence_level)) / 2.0
    lo, hi = np.quantile(np.asarray(draws), [alpha, 1.0 - alpha])
    return {
        "mean": float(x.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_rows": len(x),
        "n_clusters": len(unique),
    }


def aggregate_intervention_rows(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Aggregate effect and flip-rate by condition/alpha with pair-cluster CIs."""
    required = {
        "condition",
        "alpha",
        "pair_id",
        "delta_margin_toward_expected",
        "expected_label",
        "prediction_after",
        "counterfactual_flip",
        "delta_norm",
        "activation_norm",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"intervention rows missing columns: {sorted(missing)}")
    out: list[dict[str, Any]] = []
    group_cols = ["condition", "alpha"]
    if "direction_seed" in rows.columns:
        group_cols.append("direction_seed")
    grouped = rows.groupby(group_cols, dropna=False, sort=True)
    for gi, (key, block) in enumerate(grouped):
        key_tuple = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_cols, key_tuple))
        effect = cluster_bootstrap_mean_ci(
            block["delta_margin_toward_expected"].to_numpy(float),
            block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=int(seed) + gi * 17,
        )
        flips = (
            block["prediction_after"].to_numpy(int)
            == block["expected_label"].to_numpy(int)
        ).astype(float)
        flip_ci = cluster_bootstrap_mean_ci(
            flips,
            block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=int(seed) + gi * 17 + 1,
        )
        actual_flip_ci = cluster_bootstrap_mean_ci(
            block["counterfactual_flip"].to_numpy(float),
            block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=int(seed) + gi * 17 + 2,
        )
        ratios = block["delta_norm"].to_numpy(float) / np.maximum(
            block["activation_norm"].to_numpy(float), 1e-12
        )
        out.append(
            {
                **meta,
                "mean_delta_margin_toward_expected": effect["mean"],
                "effect_ci_low": effect["ci_low"],
                "effect_ci_high": effect["ci_high"],
                "expected_label_rate_after": flip_ci["mean"],
                "expected_label_rate_after_ci_low": flip_ci["ci_low"],
                "expected_label_rate_after_ci_high": flip_ci["ci_high"],
                "counterfactual_flip_rate": actual_flip_ci["mean"],
                "counterfactual_flip_ci_low": actual_flip_ci["ci_low"],
                "counterfactual_flip_ci_high": actual_flip_ci["ci_high"],
                "mean_delta_norm": float(block["delta_norm"].mean()),
                "mean_delta_over_activation_norm": float(ratios.mean()),
                "n_rows": len(block),
                "n_pairs": int(block["pair_id"].nunique()),
            }
        )
    return pd.DataFrame(out)


def paired_control_contrast(
    rows: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    alpha: float,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Paired treatment-minus-control effect, averaging control seeds per base."""
    block = rows[np.isclose(rows["alpha"].to_numpy(float), float(alpha))].copy()
    treat = block[block["condition"] == treatment]
    ctrl = block[block["condition"] == control]
    if len(treat) == 0 or len(ctrl) == 0:
        raise ValueError("treatment/control rows missing")
    keys = ["base_sample_id", "pair_id"]
    t = treat.groupby(keys, as_index=False)["delta_margin_toward_expected"].mean()
    c = ctrl.groupby(keys, as_index=False)["delta_margin_toward_expected"].mean()
    merged = t.merge(c, on=keys, suffixes=("_t", "_c"), how="inner")
    if len(merged) != len(t):
        raise RuntimeError("paired control contrast lost treatment examples")
    diff = (
        merged["delta_margin_toward_expected_t"].to_numpy(float)
        - merged["delta_margin_toward_expected_c"].to_numpy(float)
    )
    ci = cluster_bootstrap_mean_ci(
        diff,
        merged["pair_id"].astype(str).tolist(),
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "treatment": treatment,
        "control": control,
        "alpha": float(alpha),
        "mean_difference": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n_rows": ci["n_rows"],
        "n_pairs": ci["n_clusters"],
    }


def dose_response_summary(rows: pd.DataFrame, condition: str) -> dict[str, float]:
    """Simple aggregate slope/correlation of effect against alpha."""
    block = rows[rows["condition"] == condition].copy()
    by_alpha = (
        block.groupby("alpha", as_index=False)["delta_margin_toward_expected"].mean()
        .sort_values("alpha")
    )
    if len(by_alpha) < 2:
        return {"slope": float("nan"), "pearson_r": float("nan")}
    x = by_alpha["alpha"].to_numpy(float)
    y = by_alpha["delta_margin_toward_expected"].to_numpy(float)
    slope = float(np.polyfit(x, y, deg=1)[0])
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0 else 0.0
    return {"slope": slope, "pearson_r": r}
