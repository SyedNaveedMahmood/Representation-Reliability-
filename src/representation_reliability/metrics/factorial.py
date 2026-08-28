"""Factorial estimands and pair-cluster summaries for E01B-3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .causal import cluster_bootstrap_mean_ci

FACTORIAL_CONTEXTS = (
    "matched_orthogonal",
    "same_family_shuffled_orthogonal",
    "different_family_shuffled_orthogonal",
    "same_label_orthogonal",
    "random_orthogonal",
)


def factorial_estimands(
    y00: np.ndarray | float,
    y10: np.ndarray | float,
    y01: np.ndarray | float,
    y11: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Return the frozen four-arm E01B-3 decomposition."""
    clean = np.asarray(y00, dtype=np.float64)
    semantic = np.asarray(y10, dtype=np.float64)
    context = np.asarray(y01, dtype=np.float64)
    combined = np.asarray(y11, dtype=np.float64)
    q0 = semantic - clean
    additive = context - clean
    q_context = combined - context
    interaction = (combined - semantic) - additive
    if not np.allclose(interaction, q_context - q0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("factorial interaction algebra failed")
    return {
        "Q0": q0,
        "A_context": additive,
        "Q_context": q_context,
        "G_interaction": interaction,
    }


def _seed_key(frame: pd.DataFrame) -> pd.Series:
    seed = pd.to_numeric(frame["direction_seed"], errors="coerce")
    return seed.fillna(-1).astype(np.int64)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} missing columns: {sorted(missing)}")


def build_factorial_rows(
    prior_rows: pd.DataFrame,
    context_only_rows: pd.DataFrame,
    *,
    y11_run_id: str,
    y01_run_id: str,
    numeric_tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Merge compatible E01B-2 and new Y01 evidence into factorial rows."""
    common = {
        "model_id",
        "resolved_revision",
        "base_sample_id",
        "pair_id",
        "relation_family",
        "target_label",
        "q_base",
        "q_target",
        "condition",
        "lambda_context",
        "direction_seed",
        "context_source_id",
        "context_selection_seed",
        "context_applied_norm",
        "token_id",
        "token_index",
        "margin_toward_target_before",
        "margin_toward_target_after",
    }
    _require_columns(prior_rows, common, "E01B-2 rows")
    _require_columns(context_only_rows, common, "E01B-3 Y01 rows")
    prior = prior_rows.copy()
    y01 = context_only_rows.copy()
    prior["_seed_key"] = _seed_key(prior)
    y01["_seed_key"] = _seed_key(y01)
    coordinate = prior[prior["condition"] == "coordinate_only"].copy()
    if coordinate["base_sample_id"].duplicated().any():
        raise RuntimeError("duplicate E01B-2 coordinate-only rows")
    y11 = prior[prior["condition"].isin(FACTORIAL_CONTEXTS)].copy()
    keys = ["base_sample_id", "condition", "lambda_context", "_seed_key"]
    if y11.duplicated(keys).any() or y01.duplicated(keys).any():
        raise RuntimeError("duplicate factorial arm identity")
    merged = y01.merge(
        y11,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_y01", "_y11"),
    )
    if len(merged) != len(y01) or len(merged) != len(y11):
        raise RuntimeError("Y01/Y11 factorial identity mismatch")
    merged = merged.merge(
        coordinate[
            [
                "base_sample_id",
                "margin_toward_target_before",
                "margin_toward_target_after",
                "q_base",
                "q_after",
            ]
        ],
        on="base_sample_id",
        validate="many_to_one",
    )

    exact_pairs = (
        ("model_id_y01", "model_id_y11"),
        ("resolved_revision_y01", "resolved_revision_y11"),
        ("pair_id_y01", "pair_id_y11"),
        ("relation_family_y01", "relation_family_y11"),
        ("target_label_y01", "target_label_y11"),
        ("context_source_id_y01", "context_source_id_y11"),
        ("context_selection_seed_y01", "context_selection_seed_y11"),
        ("token_id_y01", "token_id_y11"),
        ("token_index_y01", "token_index_y11"),
    )
    mismatch_count = 0
    for left, right in exact_pairs:
        lhs = merged[left].fillna("<NA>").astype(str)
        rhs = merged[right].fillna("<NA>").astype(str)
        mismatch_count += int((lhs != rhs).sum())
    numeric_pairs = (
        ("q_target_y01", "q_target_y11"),
        ("context_applied_norm_y01", "context_applied_norm_y11"),
        ("q_base_y01", "q_base_y11"),
        ("margin_toward_target_before_y01", "margin_toward_target_before_y11"),
    )
    max_numeric = 0.0
    for left, right in numeric_pairs:
        deviation = np.abs(merged[left].to_numpy(float) - merged[right].to_numpy(float))
        max_numeric = max(max_numeric, float(np.max(deviation, initial=0.0)))
        mismatch_count += int(np.sum(deviation > float(numeric_tolerance)))
    if mismatch_count:
        raise RuntimeError(
            f"factorial evidence compatibility mismatch: {mismatch_count} "
            f"(max numeric deviation {max_numeric})"
        )

    y00 = merged["margin_toward_target_before"].to_numpy(float)
    y10 = merged["margin_toward_target_after"].to_numpy(float)
    y01_values = merged["margin_toward_target_after_y01"].to_numpy(float)
    y11_values = merged["margin_toward_target_after_y11"].to_numpy(float)
    estimands = factorial_estimands(y00, y10, y01_values, y11_values)
    output = pd.DataFrame(
        {
            "model_id": merged["model_id_y01"],
            "resolved_revision": merged["resolved_revision_y01"],
            "base_sample_id": merged["base_sample_id"],
            "pair_id": merged["pair_id_y01"],
            "relation_family": merged["relation_family_y01"],
            "target_label": merged["target_label_y01"].astype(int),
            "condition": merged["condition"],
            "lambda_context": merged["lambda_context"].astype(float),
            "direction_seed": merged["direction_seed_y01"],
            "context_source_id": merged["context_source_id_y01"],
            "context_selection_seed": merged["context_selection_seed_y01"],
            "context_applied_norm": merged["context_applied_norm_y01"].astype(float),
            "q_target": merged["q_target_y01"].astype(float),
            "Y00": y00,
            "Y10": y10,
            "Y01": y01_values,
            "Y11": y11_values,
            **estimands,
            "y00_run_id": str(y11_run_id),
            "y10_run_id": str(y11_run_id),
            "y01_run_id": str(y01_run_id),
            "y11_run_id": str(y11_run_id),
            "confirmation_accessed": False,
        }
    )
    if not np.isfinite(
        output[
            ["Y00", "Y10", "Y01", "Y11", "Q0", "A_context", "Q_context", "G_interaction"]
        ].to_numpy(float)
    ).all():
        raise RuntimeError("non-finite factorial evidence")
    return output, {
        "compatibility_mismatches": 0,
        "max_numeric_identity_deviation": max_numeric,
        "n_factorial_rows": len(output),
    }


def average_random_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    """Average random seeds per directed example before inference."""
    nonrandom = rows[rows["condition"] != "random_orthogonal"].copy()
    random = rows[rows["condition"] == "random_orthogonal"].copy()
    if random.empty:
        return nonrandom
    group_keys = [
        "model_id",
        "base_sample_id",
        "pair_id",
        "relation_family",
        "target_label",
        "condition",
        "lambda_context",
    ]
    candidates = [
        "Y00",
        "Y10",
        "Y01",
        "Y11",
        "Q0",
        "A_context",
        "Q_context",
        "G_interaction",
        "A_q_z",
        "G_q_z",
        "A_margin_z",
        "G_margin_z",
    ]
    numeric = [column for column in candidates if column in random.columns]
    if "trace_layer" in random.columns:
        group_keys.append("trace_layer")
    averaged = random.groupby(group_keys, as_index=False)[numeric].mean()
    averaged["direction_seed"] = np.nan
    return pd.concat([nonrandom, averaged], ignore_index=True, sort=False)


def aggregate_factorial_metrics(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize additive and interaction effects by condition and lambda."""
    averaged = average_random_seeds(rows)
    additive_rows: list[dict[str, float | int | str]] = []
    interaction_rows: list[dict[str, float | int | str]] = []
    for index, ((condition, strength), block) in enumerate(
        averaged.groupby(["condition", "lambda_context"], sort=True)
    ):
        for metric, output in (("A_context", additive_rows), ("G_interaction", interaction_rows)):
            ci = cluster_bootstrap_mean_ci(
                block[metric].to_numpy(float),
                block["pair_id"].astype(str).tolist(),
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level,
                seed=int(seed) + index * 101 + (0 if metric == "A_context" else 37),
            )
            output.append(
                {
                    "condition": str(condition),
                    "lambda_context": float(strength),
                    "n_examples": len(block),
                    "n_pairs": int(block["pair_id"].nunique()),
                    "mean": ci["mean"],
                    "median": float(np.median(block[metric].to_numpy(float))),
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                }
            )
    return pd.DataFrame(additive_rows), pd.DataFrame(interaction_rows)


def paired_factorial_contrast(
    rows: pd.DataFrame,
    *,
    metric: str,
    left_condition: str,
    right_condition: str,
    context_strength: float,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | str]:
    """Pair a factorial contrast by base identity, then cluster by pair."""
    if metric not in {"A_context", "G_interaction"}:
        raise ValueError("factorial contrast metric must be A_context or G_interaction")
    averaged = average_random_seeds(rows)
    block = averaged[np.isclose(averaged["lambda_context"], context_strength)]
    left = block[block["condition"] == left_condition][
        ["base_sample_id", "pair_id", metric]
    ].rename(columns={metric: "left"})
    right = block[block["condition"] == right_condition][
        ["base_sample_id", "pair_id", metric]
    ].rename(columns={metric: "right", "pair_id": "pair_id_right"})
    paired = left.merge(right, on="base_sample_id", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("factorial contrast lost matched base examples")
    if not (paired["pair_id"].astype(str) == paired["pair_id_right"].astype(str)).all():
        raise RuntimeError("factorial contrast pair identity mismatch")
    values = paired["left"].to_numpy(float) - paired["right"].to_numpy(float)
    ci = cluster_bootstrap_mean_ci(
        values,
        paired["pair_id"].astype(str).tolist(),
        n_bootstraps=n_bootstraps,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "metric": metric,
        "left_condition": left_condition,
        "right_condition": right_condition,
        "lambda_context": float(context_strength),
        "n_examples": len(paired),
        "n_pairs": int(paired["pair_id"].nunique()),
        "mean_difference": ci["mean"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
    }


def build_factorial_trace_rows(
    prior_trace: pd.DataFrame,
    context_only_trace: pd.DataFrame,
    target_labels: pd.DataFrame,
    *,
    layer_references: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Construct A/G trace decompositions from reused and new trace arms."""
    prior = prior_trace.copy()
    y01 = context_only_trace.copy()
    prior["_seed_key"] = _seed_key(prior)
    y01["_seed_key"] = _seed_key(y01)
    coordinate = prior[prior["condition"] == "coordinate_only"].copy()
    coord_keys = ["base_sample_id", "trace_layer"]
    if coordinate.duplicated(coord_keys).any():
        raise RuntimeError("duplicate coordinate-only trace identity")
    y11 = prior[prior["condition"].isin(FACTORIAL_CONTEXTS)].copy()
    keys = ["base_sample_id", "condition", "lambda_context", "_seed_key", "trace_layer"]
    merged = y01.merge(
        y11,
        on=keys,
        validate="one_to_one",
        suffixes=("_y01", "_y11"),
    )
    if len(merged) != len(y01) or len(merged) != len(y11):
        raise RuntimeError("factorial trace Y01/Y11 identity mismatch")
    merged = merged.merge(
        coordinate[
            coord_keys
            + [
                "clean_truth_coordinate",
                "intervened_truth_coordinate",
                "clean_native_yes_no_margin",
                "intervened_native_yes_no_margin",
            ]
        ],
        on=coord_keys,
        validate="many_to_one",
    )
    labels = target_labels[["base_sample_id", "target_label"]].drop_duplicates()
    if labels["base_sample_id"].duplicated().any():
        raise RuntimeError("duplicate target-label identity")
    merged = merged.merge(labels, on="base_sample_id", validate="many_to_one")
    orientation = np.where(merged["target_label"].to_numpy(int) == 1, 1.0, -1.0)
    q00 = merged["clean_truth_coordinate"].to_numpy(float)
    q10 = merged["intervened_truth_coordinate"].to_numpy(float)
    q01 = merged["intervened_truth_coordinate_y01"].to_numpy(float)
    q11 = merged["intervened_truth_coordinate_y11"].to_numpy(float)
    m00 = merged["clean_native_yes_no_margin"].to_numpy(float)
    m10 = merged["intervened_native_yes_no_margin"].to_numpy(float)
    m01 = merged["intervened_native_yes_no_margin_y01"].to_numpy(float)
    m11 = merged["intervened_native_yes_no_margin_y11"].to_numpy(float)
    q_est = factorial_estimands(q00, q10, q01, q11)
    m_est = factorial_estimands(m00, m10, m01, m11)
    sigma_q = np.asarray(
        [
            float(layer_references[str(int(layer))]["sigma_q_validation"])
            for layer in merged["trace_layer"]
        ]
    )
    sigma_m = np.asarray(
        [
            float(layer_references[str(int(layer))]["sigma_margin_validation"])
            for layer in merged["trace_layer"]
        ]
    )
    if np.any(sigma_q <= 0.0) or np.any(sigma_m <= 0.0):
        raise RuntimeError("invalid validation-only trace scale")
    return pd.DataFrame(
        {
            "model_id": merged["model_id_y01"],
            "base_sample_id": merged["base_sample_id"],
            "pair_id": merged["pair_id_y01"],
            "relation_family": merged["relation_family_y01"],
            "target_label": merged["target_label"],
            "condition": merged["condition"],
            "lambda_context": merged["lambda_context"],
            "direction_seed": merged["direction_seed_y01"],
            "trace_layer": merged["trace_layer"].astype(int),
            "A_q": orientation * q_est["A_context"],
            "G_q": orientation * q_est["G_interaction"],
            "A_margin": orientation * m_est["A_context"],
            "G_margin": orientation * m_est["G_interaction"],
            "A_q_z": orientation * q_est["A_context"] / sigma_q,
            "G_q_z": orientation * q_est["G_interaction"] / sigma_q,
            "A_margin_z": orientation * m_est["A_context"] / sigma_m,
            "G_margin_z": orientation * m_est["G_interaction"] / sigma_m,
            "confirmation_accessed": False,
        }
    )


def aggregate_factorial_trace(rows: pd.DataFrame) -> pd.DataFrame:
    averaged = average_random_seeds(rows)
    metrics = ["A_q_z", "G_q_z", "A_margin_z", "G_margin_z"]
    output: list[dict[str, float | int | str]] = []
    for (condition, strength, layer), block in averaged.groupby(
        ["condition", "lambda_context", "trace_layer"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "condition": str(condition),
            "lambda_context": float(strength),
            "trace_layer": int(layer),
            "n_examples": len(block),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(block[metric].mean())
        output.append(row)
    return pd.DataFrame(output)


def relation_family_factorial_metrics(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Frozen lambda-1 A/G summaries without family selection."""
    averaged = average_random_seeds(rows)
    block = averaged[np.isclose(averaged["lambda_context"], 1.0)]
    output: list[dict[str, float | int | str]] = []
    for index, ((family, condition), group) in enumerate(
        block.groupby(["relation_family", "condition"], sort=True)
    ):
        row: dict[str, float | int | str] = {
            "relation_family": str(family),
            "condition": str(condition),
            "lambda_context": 1.0,
            "n_examples": len(group),
            "n_pairs": int(group["pair_id"].nunique()),
        }
        for offset, metric in enumerate(("A_context", "G_interaction")):
            ci = cluster_bootstrap_mean_ci(
                group[metric].to_numpy(float),
                group["pair_id"].astype(str).tolist(),
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level,
                seed=seed + index * 97 + offset * 31,
            )
            row[f"mean_{metric}"] = ci["mean"]
            row[f"{metric}_ci_low"] = ci["ci_low"]
            row[f"{metric}_ci_high"] = ci["ci_high"]
        output.append(row)
    return pd.DataFrame(output)
