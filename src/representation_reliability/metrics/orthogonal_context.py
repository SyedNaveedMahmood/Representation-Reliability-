"""Pair-cluster summaries for E01B-2 orthogonal-context modulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .causal import cluster_bootstrap_mean_ci

CONTEXT_CONDITIONS = (
    "coordinate_only",
    "matched_orthogonal",
    "same_family_shuffled_orthogonal",
    "different_family_shuffled_orthogonal",
    "same_label_orthogonal",
    "random_orthogonal",
)


def attach_context_increments(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach the paired condition-minus-coordinate-only primary estimand."""
    required = {
        "base_sample_id",
        "pair_id",
        "condition",
        "delta_margin_toward_target",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"context rows missing columns: {sorted(missing)}")
    baseline = rows[rows["condition"] == "coordinate_only"]
    if baseline["base_sample_id"].duplicated().any() or len(baseline) == 0:
        raise RuntimeError("coordinate-only baseline must be exactly one row per base")
    baseline_map = baseline.set_index("base_sample_id")[
        "delta_margin_toward_target"
    ].to_dict()
    output = rows.copy()
    if not set(output["base_sample_id"]).issubset(baseline_map):
        raise RuntimeError("context increment lacks coordinate-only baseline")
    output["coordinate_only_delta_margin"] = output["base_sample_id"].map(
        baseline_map
    )
    output["context_increment_vs_coordinate_only"] = (
        output["delta_margin_toward_target"]
        - output["coordinate_only_delta_margin"]
    )
    return output


def _condition_block(
    rows: pd.DataFrame, condition: str, context_strength: float
) -> pd.DataFrame:
    if condition == "coordinate_only":
        block = rows[rows["condition"] == condition]
    else:
        block = rows[
            (rows["condition"] == condition)
            & np.isclose(rows["lambda_context"].to_numpy(float), context_strength)
        ]
    keys = ["base_sample_id", "pair_id"]
    return block.groupby(keys, as_index=False).agg(
        delta_margin_toward_target=("delta_margin_toward_target", "mean"),
        context_increment_vs_coordinate_only=(
            "context_increment_vs_coordinate_only",
            "mean",
        ),
    )


def aggregate_context_rows(
    rows: pd.DataFrame,
    *,
    context_strengths: tuple[float, ...],
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    index = 0
    for strength in context_strengths:
        for condition in CONTEXT_CONDITIONS:
            block = _condition_block(rows, condition, strength)
            if len(block) == 0:
                raise RuntimeError(f"missing E01B-2 condition {condition} at {strength}")
            effect = cluster_bootstrap_mean_ci(
                block["delta_margin_toward_target"].to_numpy(float),
                block["pair_id"].astype(str).tolist(),
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level,
                seed=seed + index * 19,
            )
            increment = cluster_bootstrap_mean_ci(
                block["context_increment_vs_coordinate_only"].to_numpy(float),
                block["pair_id"].astype(str).tolist(),
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level,
                seed=seed + index * 19 + 1,
            )
            output.append(
                {
                    "condition": condition,
                    "lambda_context": float(strength if condition != "coordinate_only" else 0.0),
                    "comparison_lambda": float(strength),
                    "n_rows": len(block),
                    "n_pairs": int(block["pair_id"].nunique()),
                    "mean_effect": effect["mean"],
                    "median_effect": float(block["delta_margin_toward_target"].median()),
                    "effect_ci_low": effect["ci_low"],
                    "effect_ci_high": effect["ci_high"],
                    "mean_context_increment": increment["mean"],
                    "increment_ci_low": increment["ci_low"],
                    "increment_ci_high": increment["ci_high"],
                }
            )
            index += 1
    return pd.DataFrame(output)


def paired_context_contrast(
    rows: pd.DataFrame,
    *,
    left_condition: str,
    right_condition: str,
    context_strength: float,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    left = _condition_block(rows, left_condition, context_strength)
    right = _condition_block(rows, right_condition, context_strength)
    keys = ["base_sample_id", "pair_id"]
    merged = left.merge(right, on=keys, validate="one_to_one", suffixes=("_l", "_r"))
    if len(merged) != len(left) or len(left) == 0:
        raise RuntimeError("paired E01B-2 contrast lost rows")
    difference = (
        merged["delta_margin_toward_target_l"].to_numpy(float)
        - merged["delta_margin_toward_target_r"].to_numpy(float)
    )
    ci = cluster_bootstrap_mean_ci(
        difference,
        merged["pair_id"].astype(str).tolist(),
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "left_condition": left_condition,
        "right_condition": right_condition,
        "lambda_context": float(context_strength),
        "mean_difference": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n_rows": ci["n_rows"],
        "n_pairs": ci["n_clusters"],
    }


def attach_trace_context_increments(trace_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["base_sample_id", "trace_layer"]
    baseline = trace_rows[trace_rows["condition"] == "coordinate_only"]
    if baseline.duplicated(keys).any() or len(baseline) == 0:
        raise RuntimeError("trace coordinate-only baseline is not unique")
    reference = baseline[
        keys + ["oriented_delta_q_z", "oriented_delta_native_margin_z"]
    ].rename(
        columns={
            "oriented_delta_q_z": "coordinate_only_delta_q_z",
            "oriented_delta_native_margin_z": "coordinate_only_delta_m_z",
        }
    )
    merged = trace_rows.merge(reference, on=keys, validate="many_to_one")
    merged["context_increment_delta_q_z"] = (
        merged["oriented_delta_q_z"] - merged["coordinate_only_delta_q_z"]
    )
    merged["context_increment_delta_m_z"] = (
        merged["oriented_delta_native_margin_z"]
        - merged["coordinate_only_delta_m_z"]
    )
    return merged


def aggregate_trace_context(
    trace_rows: pd.DataFrame,
    *,
    context_strengths: tuple[float, ...],
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for strength in context_strengths:
        for condition in CONTEXT_CONDITIONS:
            if condition == "coordinate_only":
                block = trace_rows[trace_rows["condition"] == condition]
            else:
                block = trace_rows[
                    (trace_rows["condition"] == condition)
                    & np.isclose(
                        trace_rows["lambda_context"].to_numpy(float), strength
                    )
                ]
            block = block.groupby(
                ["base_sample_id", "pair_id", "trace_layer"], as_index=False
            ).mean(numeric_only=True)
            for layer, layer_rows in block.groupby("trace_layer", sort=True):
                output.append(
                    {
                        "condition": condition,
                        "lambda_context": float(strength if condition != "coordinate_only" else 0.0),
                        "comparison_lambda": float(strength),
                        "trace_layer": int(layer),
                        "n_rows": len(layer_rows),
                        "mean_delta_q_z": float(layer_rows["oriented_delta_q_z"].mean()),
                        "mean_delta_m_z": float(
                            layer_rows["oriented_delta_native_margin_z"].mean()
                        ),
                        "mean_context_increment_delta_q_z": float(
                            layer_rows["context_increment_delta_q_z"].mean()
                        ),
                        "mean_context_increment_delta_m_z": float(
                            layer_rows["context_increment_delta_m_z"].mean()
                        ),
                    }
                )
    return pd.DataFrame(output)
