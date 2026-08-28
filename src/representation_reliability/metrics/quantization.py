"""Pure E14 factorial and cross-precision statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .causal import cluster_bootstrap_mean_ci

FACTORIAL_EVIDENCE_COLUMNS = (
    "q_base",
    "q_target",
    "q_after_y01",
    "q_after_y11",
    "context_norm",
    "context_dot_u",
    "Y00",
    "Y10",
    "Y01",
    "Y11",
    "Q0",
    "A",
    "Q_context",
    "G",
)
TRACE_EVIDENCE_COLUMNS = (
    "q00",
    "q10",
    "q01",
    "q11",
    "m00",
    "m10",
    "m01",
    "m11",
    "A_q_z",
    "G_q_z",
    "A_margin_z",
    "G_margin_z",
)


def factorial_components(y00, y10, y01, y11) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (y00, y10, y01, y11)]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("factorial arms must have identical shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("factorial arms must be finite")
    clean, q_only, context_only, combined = arrays
    q0 = q_only - clean
    additive = context_only - clean
    q_context = combined - context_only
    interaction = q_context - q0
    return {"Q0": q0, "A": additive, "Q_context": q_context, "G": interaction}


def average_random_contexts(rows: pd.DataFrame) -> pd.DataFrame:
    """Average preregistered random seeds per base before paired contrasts."""
    required = {"context", "base_sample_id", "pair_id", "Q0", "A", "G"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"factorial rows missing columns: {sorted(missing)}")
    structured = rows[rows["context"] != "random"].copy()
    random = rows[rows["context"] == "random"].copy()
    if random.empty:
        raise ValueError("random factorial rows are missing")
    group = ["base_sample_id", "pair_id"]
    keep = [column for column in ("relation_family", "target_label") if column in random]
    averaged = random.groupby(group, as_index=False)[["Q0", "A", "Q_context", "G"]].mean()
    for column in keep:
        metadata = random.groupby(group, as_index=False)[column].first()
        averaged = averaged.merge(metadata, on=group, validate="one_to_one")
    averaged["context"] = "random"
    averaged["direction_seed"] = np.nan
    return pd.concat([structured, averaged], ignore_index=True, sort=False)


def summarize_factorial(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    averaged = average_random_contexts(rows)
    output: list[dict[str, Any]] = []
    for index, (context, block) in enumerate(averaged.groupby("context", sort=True)):
        row: dict[str, Any] = {
            "context": str(context),
            "n_examples": len(block),
            "n_pairs": int(block["pair_id"].nunique()),
        }
        for offset, metric in enumerate(("Q0", "A", "G")):
            result = cluster_bootstrap_mean_ci(
                block[metric].to_numpy(float),
                block["pair_id"].astype(str).tolist(),
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level,
                seed=int(seed) + index * 101 + offset,
            )
            row[f"mean_{metric}"] = result["mean"]
            row[f"{metric}_ci_low"] = result["ci_low"]
            row[f"{metric}_ci_high"] = result["ci_high"]
        output.append(row)
    table = pd.DataFrame(output)
    matched = averaged[averaged["context"] == "matched"]
    random = averaged[averaged["context"] == "random"]
    keys = ["base_sample_id", "pair_id"]
    merged = matched.merge(random, on=keys, validate="one_to_one", suffixes=("_m", "_r"))
    if len(merged) != len(matched) or len(matched) == 0:
        raise RuntimeError("matched/random factorial pairing is incomplete")
    contrasts: dict[str, Any] = {}
    for offset, metric in enumerate(("A", "G")):
        difference = merged[f"{metric}_m"].to_numpy(float) - merged[f"{metric}_r"].to_numpy(float)
        contrasts[f"{metric}_matched_minus_random"] = cluster_bootstrap_mean_ci(
            difference,
            merged["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=int(seed) + 900 + offset,
        )
    return table, contrasts


def percent_change(value: float, reference: float, *, epsilon: float = 1e-12) -> float | None:
    if abs(float(reference)) <= float(epsilon):
        return None
    return 100.0 * (float(value) - float(reference)) / abs(float(reference))


def evidence_is_finite(
    factorial_rows: pd.DataFrame,
    trace_rows: pd.DataFrame,
) -> bool:
    """Check measured evidence while allowing null provenance identifiers."""
    missing_factorial = set(FACTORIAL_EVIDENCE_COLUMNS) - set(factorial_rows.columns)
    missing_trace = set(TRACE_EVIDENCE_COLUMNS) - set(trace_rows.columns)
    if missing_factorial or missing_trace:
        raise ValueError(
            f"missing finite-evidence columns: factorial={sorted(missing_factorial)}, "
            f"trace={sorted(missing_trace)}"
        )
    return bool(
        np.isfinite(factorial_rows[list(FACTORIAL_EVIDENCE_COLUMNS)].to_numpy(float)).all()
        and np.isfinite(trace_rows[list(TRACE_EVIDENCE_COLUMNS)].to_numpy(float)).all()
    )
