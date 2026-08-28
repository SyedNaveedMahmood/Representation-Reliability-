"""CPU-only mechanism analysis for completed E01A discovery traces.

This module reads immutable per-example E01A evidence. It never loads a model,
regenerates behavior, or accesses the confirmation split.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # isort: skip


REQUIRED_RUN_FILES = (
    "intervention_rows.parquet",
    "trace_rows.parquet",
    "aggregate_metrics.parquet",
    "control_contrasts.parquet",
    "e01a_metrics.json",
    "manifest.json",
    "status.json",
)
EXPECTED_ALPHAS = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
EXPECTED_TRACE_LAYERS = (17, 20, 23, 27)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def validate_completed_run(run_dir: Path) -> dict[str, Any]:
    """Validate the predeclared full-discovery artifact contract."""
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} missing required evidence: {missing}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "e01a_metrics.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete" or metrics.get("status") != "complete":
        raise RuntimeError(f"E01A run is not complete: {run_dir}")
    if metrics.get("confirmation_accessed") is not False:
        raise RuntimeError(f"confirmation-access invariant failed: {run_dir}")
    if int(metrics.get("n_base_examples", -1)) != 300 or int(metrics.get("n_pairs", -1)) != 150:
        raise RuntimeError(f"unexpected full-discovery sample count: {run_dir}")
    if tuple(map(int, metrics.get("trace_layers", ()))) != EXPECTED_TRACE_LAYERS:
        raise RuntimeError(f"unexpected trace-layer grid: {run_dir}")
    if not np.allclose(metrics.get("alphas", ()), EXPECTED_ALPHAS, atol=0.0, rtol=0.0):
        raise RuntimeError(f"unexpected alpha grid: {run_dir}")
    return metrics


def deduplicate_clean_traces(
    trace_rows: pd.DataFrame, *, tolerance: float = 1e-12
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Check repeated clean traces and return one baseline per base/layer."""
    columns = (
        "base_sample_id",
        "pair_id",
        "trace_layer",
        "clean_truth_coordinate",
        "clean_native_yes_no_margin",
    )
    _require_columns(trace_rows, columns, "trace rows")
    keys = ["base_sample_id", "trace_layer"]
    grouped = trace_rows.groupby(keys, sort=True, dropna=False)
    q_range = grouped["clean_truth_coordinate"].agg(lambda x: float(x.max() - x.min()))
    m_range = grouped["clean_native_yes_no_margin"].agg(lambda x: float(x.max() - x.min()))
    pair_n = grouped["pair_id"].nunique()
    q_max = float(q_range.max())
    m_max = float(m_range.max())
    if int(pair_n.max()) != 1:
        raise RuntimeError("a base/layer trace group maps to multiple pair IDs")
    if q_max > float(tolerance) or m_max > float(tolerance):
        raise RuntimeError(
            "clean trace baselines disagree across repeated conditions: "
            f"truth={q_max}, native_margin={m_max}, tolerance={tolerance}"
        )
    clean = grouped[
        ["pair_id", "clean_truth_coordinate", "clean_native_yes_no_margin"]
    ].first().reset_index()
    if clean.duplicated(keys).any():
        raise RuntimeError("clean trace deduplication did not produce unique keys")
    return clean, {
        "max_clean_truth_coordinate_disagreement": q_max,
        "max_clean_native_margin_disagreement": m_max,
    }


def expected_label_identity(intervention_rows: pd.DataFrame) -> pd.DataFrame:
    """Build the smallest safe per-base identity table for trace orientation."""
    columns = (
        "base_sample_id",
        "pair_id",
        "expected_label",
        "relation_family",
        "gold_label",
        "base_prediction",
    )
    _require_columns(intervention_rows, columns, "intervention rows")
    grouped = intervention_rows.groupby("base_sample_id", sort=True, dropna=False)
    for column in columns[1:]:
        if int(grouped[column].nunique(dropna=False).max()) != 1:
            raise RuntimeError(f"{column} is not invariant within base_sample_id")
    identity = grouped[list(columns[1:])].first().reset_index()
    labels = set(identity["expected_label"].astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError(f"expected_label is not binary: {sorted(labels)}")
    return identity


def orient_trace_changes(
    trace_rows: pd.DataFrame, intervention_rows: pd.DataFrame
) -> pd.DataFrame:
    """Merge expected labels into traces and orient changes toward the target."""
    _require_columns(
        trace_rows,
        (
            "base_sample_id",
            "pair_id",
            "condition",
            "alpha",
            "direction_seed",
            "trace_layer",
            "delta_truth_coordinate",
            "delta_native_yes_no_margin",
        ),
        "trace rows",
    )
    identity = expected_label_identity(intervention_rows)
    merged = trace_rows.merge(
        identity,
        on="base_sample_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_identity"),
    )
    if len(merged) != len(trace_rows) or merged["expected_label"].isna().any():
        raise RuntimeError("trace/expected-label merge lost or duplicated evidence")
    if not (merged["pair_id"].astype(str) == merged["pair_id_identity"].astype(str)).all():
        raise RuntimeError("trace/intervention pair identity mismatch")
    merged = merged.drop(columns="pair_id_identity")
    sign = np.where(merged["expected_label"].to_numpy(int) == 1, 1.0, -1.0)
    merged["orientation_sign"] = sign
    merged["oriented_delta_truth_coordinate"] = (
        sign * merged["delta_truth_coordinate"].to_numpy(float)
    )
    merged["oriented_delta_native_margin"] = (
        sign * merged["delta_native_yes_no_margin"].to_numpy(float)
    )
    return merged


def standardize_changes(values: Sequence[float], clean_sd: float) -> np.ndarray:
    """Scale changes by clean discovery SD, rejecting undefined scales."""
    scale = float(clean_sd)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"clean standard deviation must be finite and positive, got {scale}")
    result = np.asarray(values, dtype=np.float64) / scale
    if not np.isfinite(result).all():
        raise ValueError("standardized changes contain non-finite values")
    return result


def cluster_bootstrap_draws(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster_col: str = "pair_id",
    n_bootstraps: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Resample whole clusters and recompute an arbitrary scalar statistic."""
    if cluster_col not in frame or len(frame) == 0:
        raise ValueError("bootstrap frame must be non-empty and contain cluster_col")
    cluster_text = frame[cluster_col].astype(str).to_numpy()
    unique = np.asarray(sorted(set(cluster_text)), dtype=object)
    indices = [np.flatnonzero(cluster_text == cluster) for cluster in unique]
    equal_cluster_sizes = len({len(index) for index in indices}) == 1
    index_matrix = np.stack(indices) if equal_cluster_sizes else None
    rng = np.random.default_rng(int(seed))
    draws = np.empty(max(1, int(n_bootstraps)), dtype=np.float64)
    for index in range(len(draws)):
        selected = rng.integers(0, len(unique), size=len(unique))
        if index_matrix is not None:
            sample_indices = index_matrix[selected].reshape(-1)
        else:
            sample_indices = np.concatenate([indices[position] for position in selected])
        sample = frame.iloc[sample_indices]
        draws[index] = float(statistic(sample))
    if not np.isfinite(draws).all():
        raise ValueError("bootstrap statistic produced non-finite draws")
    return draws


def _cluster_bootstrap_mean_draws(
    frame: pd.DataFrame,
    value_col: str,
    *,
    n_bootstraps: int,
    seed: int,
) -> np.ndarray:
    """Vectorized pair-cluster bootstrap draws for a column mean."""
    grouped = frame.groupby("pair_id", sort=True)[value_col]
    sums = grouped.sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(sums), size=(max(1, int(n_bootstraps)), len(sums)))
    return sums[selected].sum(axis=1) / counts[selected].sum(axis=1)


def _cluster_bootstrap_conversion_draws(
    frame: pd.DataFrame, *, n_bootstraps: int, seed: int
) -> np.ndarray:
    """Vectorized clustered draws of the no-intercept conversion slope."""
    working = frame[["pair_id", "delta_q_z", "delta_m_z"]].copy()
    working["xy"] = working["delta_q_z"] * working["delta_m_z"]
    working["xx"] = working["delta_q_z"] ** 2
    grouped = working.groupby("pair_id", sort=True)[["xy", "xx"]].sum()
    xy = grouped["xy"].to_numpy(float)
    xx = grouped["xx"].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(xy), size=(max(1, int(n_bootstraps)), len(xy)))
    denominator = xx[selected].sum(axis=1)
    if np.any(denominator <= 1e-12):
        raise ValueError("bootstrap truth-coordinate changes have insufficient variance")
    return xy[selected].sum(axis=1) / denominator


def _cluster_bootstrap_ratio_draws(
    frame: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    *,
    n_bootstraps: int,
    seed: int,
) -> np.ndarray:
    """Vectorized clustered ratio-of-means bootstrap."""
    grouped = frame.groupby("pair_id", sort=True)[[numerator_col, denominator_col]].sum()
    numerator = grouped[numerator_col].to_numpy(float)
    denominator = grouped[denominator_col].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(
        0, len(numerator), size=(max(1, int(n_bootstraps)), len(numerator))
    )
    denominator_draws = denominator[selected].sum(axis=1)
    if np.any(np.abs(denominator_draws) <= 1e-12):
        raise ValueError("bootstrap L17 mean is too small for propagation retention")
    return numerator[selected].sum(axis=1) / denominator_draws


def _cluster_bootstrap_regression_coefficients(
    frame: pd.DataFrame,
    *,
    include_alpha: bool,
    n_bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Bootstrap all source-regression coefficients from cluster sufficient statistics."""
    design, outcome, names = source_regression_design(frame, include_alpha=include_alpha)
    cluster_text = frame["pair_id"].astype(str).to_numpy()
    unique = sorted(set(cluster_text))
    xtx = []
    xty = []
    for cluster in unique:
        mask = cluster_text == cluster
        block_design = design[mask]
        block_outcome = outcome[mask]
        xtx.append(block_design.T @ block_design)
        xty.append(block_design.T @ block_outcome)
    xtx_array = np.stack(xtx)
    xty_array = np.stack(xty)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(
        0, len(unique), size=(max(1, int(n_bootstraps)), len(unique))
    )
    draw_xtx = xtx_array[selected].sum(axis=1)
    draw_xty = xty_array[selected].sum(axis=1)
    try:
        coefficients = np.linalg.solve(draw_xtx, draw_xty[..., np.newaxis]).squeeze(-1)
    except np.linalg.LinAlgError:
        coefficients = np.stack(
            [np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(draw_xtx, draw_xty)]
        )
    if not np.isfinite(coefficients).all():
        raise ValueError("source regression bootstrap produced non-finite coefficients")
    return coefficients, names


def _interval(draws: Sequence[float], confidence_level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(np.asarray(draws, dtype=float), [tail, 1.0 - tail])
    return float(low), float(high)


def propagation_ratio(layer_mean: float, baseline_mean: float) -> float:
    """Compute downstream retention relative to L17 with a denominator guard."""
    denominator = float(baseline_mean)
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise ValueError("L17 mean is too small for a stable propagation ratio")
    return float(layer_mean) / denominator


def conversion_slope(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    """Fit no-intercept and intercept-included descriptive conversion slopes."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        raise ValueError("conversion arrays must have equal length >= 2")
    if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
        raise ValueError("conversion arrays must be finite")
    denominator = float(x_arr @ x_arr)
    if denominator <= 1e-12:
        raise ValueError("truth-coordinate changes have insufficient variance")
    beta = float((x_arr @ y_arr) / denominator)
    residual = y_arr - beta * x_arr
    uncentered_total = float(y_arr @ y_arr)
    r2 = 1.0 - float(residual @ residual) / uncentered_total if uncentered_total > 0 else 0.0
    design = np.column_stack([np.ones(len(x_arr)), x_arr])
    intercept, beta_with_intercept = np.linalg.lstsq(design, y_arr, rcond=None)[0]
    fitted = intercept + beta_with_intercept * x_arr
    centered_total = float(np.sum((y_arr - y_arr.mean()) ** 2))
    intercept_r2 = (
        1.0 - float(np.sum((y_arr - fitted) ** 2)) / centered_total
        if centered_total > 0
        else 0.0
    )
    return {
        "beta": beta,
        "r2_uncentered": float(r2),
        "intercept": float(intercept),
        "beta_with_intercept": float(beta_with_intercept),
        "r2_with_intercept": float(intercept_r2),
    }


def merge_matched_shuffled(intervention_rows: pd.DataFrame) -> pd.DataFrame:
    """Create one alpha-1 matched/shuffled comparison per base sample."""
    required = (
        "base_sample_id",
        "pair_id",
        "condition",
        "alpha",
        "base_truth_coordinate",
        "source_truth_coordinate",
        "delta_truth_coordinate",
        "delta_margin_toward_expected",
        "expected_label",
    )
    _require_columns(intervention_rows, required, "intervention rows")
    block = intervention_rows[
        intervention_rows["condition"].isin(["truth_coordinate", "shuffled_coordinate"])
        & np.isclose(intervention_rows["alpha"].to_numpy(float), 1.0)
    ].copy()
    keys = ["base_sample_id", "condition"]
    if block.duplicated(keys).any():
        raise RuntimeError("matched/shuffled alpha-1 identity is not one row per base/condition")
    matched = block[block["condition"] == "truth_coordinate"].drop(columns="condition")
    shuffled = block[block["condition"] == "shuffled_coordinate"].drop(columns="condition")
    merged = matched.merge(
        shuffled,
        on="base_sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_matched", "_shuffled"),
    )
    if len(merged) != len(matched) or len(merged) != len(shuffled):
        raise RuntimeError("matched/shuffled merge lost base samples")
    for column in ("pair_id", "expected_label", "base_truth_coordinate"):
        left = merged[f"{column}_matched"]
        right = merged[f"{column}_shuffled"]
        if column == "base_truth_coordinate":
            equal = np.allclose(left.to_numpy(float), right.to_numpy(float), atol=1e-12)
        else:
            equal = (left.astype(str) == right.astype(str)).all()
        if not equal:
            raise RuntimeError(f"matched/shuffled {column} identity mismatch")
    sign = np.where(merged["expected_label_matched"].to_numpy(int) == 1, 1.0, -1.0)
    merged["matched_oriented_delta_q"] = sign * merged[
        "delta_truth_coordinate_matched"
    ].to_numpy(float)
    merged["shuffled_oriented_delta_q"] = sign * merged[
        "delta_truth_coordinate_shuffled"
    ].to_numpy(float)
    merged["matched_oriented_effect"] = merged[
        "delta_margin_toward_expected_matched"
    ].to_numpy(float)
    merged["shuffled_oriented_effect"] = merged[
        "delta_margin_toward_expected_shuffled"
    ].to_numpy(float)
    return merged


def source_regression_design(
    rows: pd.DataFrame, *, include_alpha: bool = False
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the coordinate/source regression design with an explicit intercept."""
    required = (
        "condition",
        "alpha",
        "expected_label",
        "delta_truth_coordinate",
        "delta_margin_toward_expected",
    )
    _require_columns(rows, required, "source regression rows")
    allowed = {"truth_coordinate", "shuffled_coordinate"}
    if not set(rows["condition"].unique()) <= allowed:
        raise ValueError("source regression accepts only truth and shuffled coordinate rows")
    if np.isclose(rows["alpha"].to_numpy(float), 0.0).any():
        raise ValueError("source regression excludes alpha=0 rows")
    sign = np.where(rows["expected_label"].to_numpy(int) == 1, 1.0, -1.0)
    oriented_q = sign * rows["delta_truth_coordinate"].to_numpy(float)
    matched = (rows["condition"].to_numpy(str) == "truth_coordinate").astype(float)
    columns = [np.ones(len(rows)), oriented_q, matched]
    names = ["intercept", "beta_coordinate", "beta_matched_indicator"]
    if include_alpha:
        columns.append(rows["alpha"].to_numpy(float))
        names.append("beta_alpha")
    design = np.column_stack(columns)
    outcome = rows["delta_margin_toward_expected"].to_numpy(float)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("source regression design is rank deficient")
    return design, outcome, names


def _correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) < 2 or np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(x_arr, y_arr).statistic),
        "spearman": float(spearmanr(x_arr, y_arr).statistic),
    }


def _describe_absolute(values: Sequence[float]) -> dict[str, float]:
    array = np.abs(np.asarray(values, dtype=float))
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _trace_layer_analysis(
    oriented: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    model_label: str,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    truth = oriented[oriented["condition"] == "truth_coordinate"].copy()
    clean_scales = clean.groupby("trace_layer").agg(
        clean_q_sd=("clean_truth_coordinate", lambda x: float(np.std(x, ddof=1))),
        clean_m_sd=("clean_native_yes_no_margin", lambda x: float(np.std(x, ddof=1))),
    )
    truth = truth.merge(clean_scales, on="trace_layer", validate="many_to_one")
    truth["delta_q_z"] = truth.groupby("trace_layer", group_keys=False).apply(
        lambda block: pd.Series(
            standardize_changes(
                block["oriented_delta_truth_coordinate"], block["clean_q_sd"].iloc[0]
            ),
            index=block.index,
        ),
        include_groups=False,
    )
    truth["delta_m_z"] = truth.groupby("trace_layer", group_keys=False).apply(
        lambda block: pd.Series(
            standardize_changes(
                block["oriented_delta_native_margin"], block["clean_m_sd"].iloc[0]
            ),
            index=block.index,
        ),
        include_groups=False,
    )
    primary = truth[np.isclose(truth["alpha"].to_numpy(float), 1.0)].copy()
    layer17 = primary[primary["trace_layer"] == 17][
        ["base_sample_id", "pair_id", "delta_q_z"]
    ].rename(columns={"delta_q_z": "delta_q_z_l17"})
    rows: list[dict[str, Any]] = []
    draws_by_layer: dict[int, dict[str, np.ndarray]] = {}
    for alpha in EXPECTED_ALPHAS:
        alpha_block = truth[np.isclose(truth["alpha"].to_numpy(float), alpha)]
        for layer in EXPECTED_TRACE_LAYERS:
            block = alpha_block[alpha_block["trace_layer"] == layer].copy()
            row: dict[str, Any] = {
                "model": model_label,
                "alpha": float(alpha),
                "trace_layer": int(layer),
                "n": len(block),
                "n_pairs": int(block["pair_id"].nunique()),
                "clean_q_sd": float(block["clean_q_sd"].iloc[0]),
                "clean_m_sd": float(block["clean_m_sd"].iloc[0]),
                "mean_oriented_delta_q": float(block["oriented_delta_truth_coordinate"].mean()),
                "median_oriented_delta_q": float(
                    block["oriented_delta_truth_coordinate"].median()
                ),
                "mean_delta_q_z": float(block["delta_q_z"].mean()),
                "median_delta_q_z": float(block["delta_q_z"].median()),
                "mean_oriented_delta_native_margin": float(
                    block["oriented_delta_native_margin"].mean()
                ),
                "median_oriented_delta_native_margin": float(
                    block["oriented_delta_native_margin"].median()
                ),
                "mean_delta_m_z": float(block["delta_m_z"].mean()),
                "median_delta_m_z": float(block["delta_m_z"].median()),
                "fraction_correct_intervention_sign": float(
                    (np.sign(float(alpha)) * block["delta_q_z"] > 0).mean()
                )
                if not np.isclose(alpha, 0.0)
                else float((np.abs(block["delta_q_z"]) <= 1e-12).mean()),
                "standardization": "discovery-standardized",
            }
            if np.isclose(alpha, 1.0):
                q_draws = _cluster_bootstrap_mean_draws(
                    block,
                    "delta_q_z",
                    n_bootstraps=n_bootstraps,
                    seed=seed + layer * 101,
                )
                m_draws = _cluster_bootstrap_mean_draws(
                    block,
                    "delta_m_z",
                    n_bootstraps=n_bootstraps,
                    seed=seed + layer * 101 + 1,
                )
                q_raw_draws = _cluster_bootstrap_mean_draws(
                    block,
                    "oriented_delta_truth_coordinate",
                    n_bootstraps=n_bootstraps,
                    seed=seed + layer * 101 + 2,
                )
                m_raw_draws = _cluster_bootstrap_mean_draws(
                    block,
                    "oriented_delta_native_margin",
                    n_bootstraps=n_bootstraps,
                    seed=seed + layer * 101 + 3,
                )
                beta_draws = _cluster_bootstrap_conversion_draws(
                    block,
                    n_bootstraps=n_bootstraps,
                    seed=seed + layer * 101 + 4,
                )
                q_low, q_high = _interval(q_draws)
                m_low, m_high = _interval(m_draws)
                qr_low, qr_high = _interval(q_raw_draws)
                mr_low, mr_high = _interval(m_raw_draws)
                beta_low, beta_high = _interval(beta_draws)
                conversion = conversion_slope(block["delta_q_z"], block["delta_m_z"])
                joined = block.merge(
                    layer17,
                    on=["base_sample_id", "pair_id"],
                    how="inner",
                    validate="one_to_one",
                )
                corr_depth = _correlation(joined["delta_q_z_l17"], joined["delta_q_z"])
                row.update(
                    {
                        "delta_q_z_ci_low": q_low,
                        "delta_q_z_ci_high": q_high,
                        "delta_m_z_ci_low": m_low,
                        "delta_m_z_ci_high": m_high,
                        "oriented_delta_q_ci_low": qr_low,
                        "oriented_delta_q_ci_high": qr_high,
                        "oriented_delta_native_margin_ci_low": mr_low,
                        "oriented_delta_native_margin_ci_high": mr_high,
                        "q17_to_layer_pearson": corr_depth["pearson"],
                        "q17_to_layer_spearman": corr_depth["spearman"],
                        "q_m_pearson": _correlation(block["delta_q_z"], block["delta_m_z"])[
                            "pearson"
                        ],
                        "q_m_spearman": _correlation(block["delta_q_z"], block["delta_m_z"])[
                            "spearman"
                        ],
                        "conversion_beta": conversion["beta"],
                        "conversion_beta_ci_low": beta_low,
                        "conversion_beta_ci_high": beta_high,
                        "conversion_r2_uncentered": conversion["r2_uncentered"],
                        "conversion_intercept": conversion["intercept"],
                        "conversion_beta_with_intercept": conversion[
                            "beta_with_intercept"
                        ],
                        "conversion_r2_with_intercept": conversion["r2_with_intercept"],
                    }
                )
                draws_by_layer[layer] = {
                    "mean_delta_q_z": q_draws,
                    "mean_delta_m_z": m_draws,
                    "conversion_beta": beta_draws,
                }
            rows.append(row)
    metrics = pd.DataFrame(rows)
    primary_metrics = metrics[np.isclose(metrics["alpha"].to_numpy(float), 1.0)].copy()
    l17_mean = float(primary_metrics.loc[primary_metrics["trace_layer"] == 17, "mean_delta_q_z"].iloc[0])
    l17_block = primary[primary["trace_layer"] == 17]
    for layer in EXPECTED_TRACE_LAYERS:
        index = metrics.index[
            np.isclose(metrics["alpha"].to_numpy(float), 1.0)
            & (metrics["trace_layer"].to_numpy(int) == layer)
        ][0]
        layer_mean = float(metrics.loc[index, "mean_delta_q_z"])
        metrics.loc[index, "propagation_retention"] = propagation_ratio(layer_mean, l17_mean)
        wide = l17_block[["base_sample_id", "pair_id", "delta_q_z"]].merge(
            primary[primary["trace_layer"] == layer][
                ["base_sample_id", "pair_id", "delta_q_z"]
            ],
            on=["base_sample_id", "pair_id"],
            validate="one_to_one",
            suffixes=("_l17", "_layer"),
        )
        retention_draws = _cluster_bootstrap_ratio_draws(
            wide,
            "delta_q_z_layer",
            "delta_q_z_l17",
            n_bootstraps=n_bootstraps,
            seed=seed + layer * 211,
        )
        low, high = _interval(retention_draws)
        metrics.loc[index, "propagation_retention_ci_low"] = low
        metrics.loc[index, "propagation_retention_ci_high"] = high
        draws_by_layer[layer]["propagation_retention"] = retention_draws
    return metrics, draws_by_layer


def _source_equivalence_analysis(
    rows: pd.DataFrame, *, model_label: str, n_bootstraps: int, seed: int
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    merged = merge_matched_shuffled(rows)
    target_corr = _correlation(
        merged["source_truth_coordinate_matched"],
        merged["source_truth_coordinate_shuffled"],
    )
    delta_corr = _correlation(
        merged["matched_oriented_delta_q"], merged["shuffled_oriented_delta_q"]
    )
    effect_corr = _correlation(
        merged["matched_oriented_effect"], merged["shuffled_oriented_effect"]
    )
    regression_rows = rows[
        rows["condition"].isin(["truth_coordinate", "shuffled_coordinate"])
        & ~np.isclose(rows["alpha"].to_numpy(float), 0.0)
    ].copy()
    sign = np.where(regression_rows["expected_label"].to_numpy(int) == 1, 1.0, -1.0)
    oriented_q = sign * regression_rows["delta_truth_coordinate"].to_numpy(float)
    alpha = regression_rows["alpha"].to_numpy(float)
    q_alpha_corr = float(pearsonr(oriented_q, alpha).statistic)
    full_design, outcome, _full_names = source_regression_design(
        regression_rows, include_alpha=True
    )
    standardized_predictors = full_design[:, 1:]
    standardized_predictors = (standardized_predictors - standardized_predictors.mean(axis=0)) / standardized_predictors.std(axis=0, ddof=1)
    condition_number = float(np.linalg.cond(standardized_predictors))
    include_alpha = abs(q_alpha_corr) <= 0.8 and condition_number <= 10.0
    design, outcome, names = source_regression_design(
        regression_rows, include_alpha=include_alpha
    )
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    fitted = design @ coefficients
    total = float(np.sum((outcome - outcome.mean()) ** 2))
    r2 = 1.0 - float(np.sum((outcome - fitted) ** 2)) / total if total > 0 else 0.0

    coefficient_draws, bootstrap_names = _cluster_bootstrap_regression_coefficients(
        regression_rows,
        include_alpha=include_alpha,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    if bootstrap_names != names:
        raise RuntimeError("source regression bootstrap changed coefficient order")
    regression_output: list[dict[str, Any]] = []
    for coefficient, estimate in zip(names, coefficients):
        draws = coefficient_draws[:, names.index(coefficient)]
        low, high = _interval(draws)
        regression_output.append(
            {
                "model": model_label,
                "coefficient": coefficient,
                "estimate": float(estimate),
                "ci_low": low,
                "ci_high": high,
                "include_alpha": include_alpha,
                "r2": r2,
                "n": len(regression_rows),
                "n_pairs": int(regression_rows["pair_id"].nunique()),
            }
        )
    payload = {
        "model": model_label,
        "n": len(merged),
        "target_coordinate_correlation": target_corr,
        "oriented_delta_q_correlation": delta_corr,
        "matched_shuffled_output_effect_correlation": effect_corr,
        "absolute_target_coordinate_difference": _describe_absolute(
            merged["source_truth_coordinate_matched"].to_numpy(float)
            - merged["source_truth_coordinate_shuffled"].to_numpy(float)
        ),
        "absolute_oriented_delta_q_difference": _describe_absolute(
            merged["matched_oriented_delta_q"].to_numpy(float)
            - merged["shuffled_oriented_delta_q"].to_numpy(float)
        ),
        "coordinate_alpha_pearson": q_alpha_corr,
        "standardized_predictor_condition_number": condition_number,
        "alpha_retained": include_alpha,
        "regression_r2": r2,
        "regression": {row["coefficient"]: row for row in regression_output},
    }
    return payload, merged, pd.DataFrame(regression_output)


def _relation_and_behavior_analysis(
    rows: pd.DataFrame,
    oriented_trace: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    model_label: str,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = rows[
        (rows["condition"] == "truth_coordinate")
        & np.isclose(rows["alpha"].to_numpy(float), 1.0)
    ].copy()
    trace_primary = oriented_trace[
        (oriented_trace["condition"] == "truth_coordinate")
        & np.isclose(oriented_trace["alpha"].to_numpy(float), 1.0)
        & oriented_trace["trace_layer"].isin([17, 27])
    ].copy()
    scales = clean.groupby("trace_layer").agg(
        q_sd=("clean_truth_coordinate", lambda x: float(np.std(x, ddof=1))),
        m_sd=("clean_native_yes_no_margin", lambda x: float(np.std(x, ddof=1))),
    )
    l17 = trace_primary[trace_primary["trace_layer"] == 17][
        ["base_sample_id", "oriented_delta_truth_coordinate"]
    ].copy()
    l17["delta_q_z_l17"] = standardize_changes(
        l17["oriented_delta_truth_coordinate"], scales.loc[17, "q_sd"]
    )
    l27 = trace_primary[trace_primary["trace_layer"] == 27][
        ["base_sample_id", "oriented_delta_native_margin"]
    ].copy()
    l27["delta_m_z_l27"] = standardize_changes(
        l27["oriented_delta_native_margin"], scales.loc[27, "m_sd"]
    )
    primary = primary.merge(
        l17[["base_sample_id", "delta_q_z_l17"]],
        on="base_sample_id",
        validate="one_to_one",
    ).merge(
        l27[["base_sample_id", "delta_m_z_l27"]],
        on="base_sample_id",
        validate="one_to_one",
    )
    family_rows: list[dict[str, Any]] = []
    for index, (family, block) in enumerate(primary.groupby("relation_family", sort=True)):
        draws = _cluster_bootstrap_mean_draws(
            block,
            "delta_margin_toward_expected",
            n_bootstraps=n_bootstraps,
            seed=seed + index * 29,
        )
        low, high = _interval(draws)
        family_rows.append(
            {
                "model": model_label,
                "relation_family": str(family),
                "n": len(block),
                "n_pairs": int(block["pair_id"].nunique()),
                "mean_oriented_output_effect": float(
                    block["delta_margin_toward_expected"].mean()
                ),
                "effect_ci_low": low,
                "effect_ci_high": high,
                "mean_delta_q_z_l17": float(block["delta_q_z_l17"].mean()),
                "mean_delta_m_z_l27": float(block["delta_m_z_l27"].mean()),
                "standardization": "discovery-standardized",
            }
        )
    behavior_rows: list[dict[str, Any]] = []
    primary["base_prediction_correct"] = (
        primary["base_prediction"].to_numpy(int) == primary["gold_label"].to_numpy(int)
    )
    for index, (correct, block) in enumerate(primary.groupby("base_prediction_correct")):
        draws = _cluster_bootstrap_mean_draws(
            block,
            "delta_margin_toward_expected",
            n_bootstraps=n_bootstraps,
            seed=seed + 500 + index * 31,
        )
        low, high = _interval(draws)
        behavior_rows.append(
            {
                "model": model_label,
                "base_prediction_stratum": "correct" if bool(correct) else "incorrect",
                "n": len(block),
                "n_pairs": int(block["pair_id"].nunique()),
                "mean_oriented_output_effect": float(
                    block["delta_margin_toward_expected"].mean()
                ),
                "effect_ci_low": low,
                "effect_ci_high": high,
                "counterfactual_flip_rate": float(block["counterfactual_flip"].mean()),
            }
        )
    return pd.DataFrame(family_rows), pd.DataFrame(behavior_rows)


def _plot_trajectories(metrics: pd.DataFrame, figure_dir: Path) -> None:
    primary = metrics[np.isclose(metrics["alpha"].to_numpy(float), 1.0)]
    specifications = (
        (
            "figure_1_truth_coordinate_propagation.png",
            "mean_delta_q_z",
            "delta_q_z_ci_low",
            "delta_q_z_ci_high",
            "Mean discovery-standardized oriented Δq",
        ),
        (
            "figure_2_native_readout_trajectory.png",
            "mean_delta_m_z",
            "delta_m_z_ci_low",
            "delta_m_z_ci_high",
            "Mean discovery-standardized oriented Δm",
        ),
        (
            "figure_3_conversion_slope.png",
            "conversion_beta",
            "conversion_beta_ci_low",
            "conversion_beta_ci_high",
            "Discovery standardized conversion slope",
        ),
    )
    for filename, value, low, high, ylabel in specifications:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        for model, block in primary.groupby("model", sort=True):
            block = block.sort_values("trace_layer")
            y = block[value].to_numpy(float)
            yerr = np.vstack([y - block[low].to_numpy(float), block[high].to_numpy(float) - y])
            ax.errorbar(block["trace_layer"], y, yerr=yerr, marker="o", capsize=3, label=model)
        ax.set_xlabel("Trace layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(EXPECTED_TRACE_LAYERS)
        ax.axhline(0.0, linewidth=0.8)
        ax.legend()
        fig.savefig(figure_dir / filename, dpi=200)
        plt.close(fig)


def _plot_source_equivalence(comparisons: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    specs = (
        (
            "figure_4_matched_vs_shuffled_coordinate_targets.png",
            "matched_oriented_delta_q",
            "shuffled_oriented_delta_q",
            "Matched oriented Δq target",
            "Shuffled oriented Δq target",
        ),
        (
            "figure_5_matched_vs_shuffled_causal_effects.png",
            "matched_oriented_effect",
            "shuffled_oriented_effect",
            "Matched oriented Δmargin",
            "Shuffled oriented Δmargin",
        ),
    )
    for filename, xcol, ycol, xlabel, ylabel in specs:
        fig, axes = plt.subplots(1, len(comparisons), figsize=(10.5, 4.5), constrained_layout=True)
        axes_array = np.atleast_1d(axes)
        for ax, (model, frame) in zip(axes_array, sorted(comparisons.items())):
            x = frame[xcol].to_numpy(float)
            y = frame[ycol].to_numpy(float)
            ax.scatter(x, y, s=12, alpha=0.65)
            low = float(min(x.min(), y.min()))
            high = float(max(x.max(), y.max()))
            ax.plot([low, high], [low, high], linewidth=0.8)
            ax.set_title(model)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        fig.savefig(figure_dir / filename, dpi=200)
        plt.close(fig)


def _plot_relation_families(metrics: pd.DataFrame, figure_dir: Path) -> None:
    families = sorted(metrics["relation_family"].unique())
    x = np.arange(len(families), dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    models = sorted(metrics["model"].unique())
    offsets = np.linspace(-0.12, 0.12, len(models))
    for offset, model in zip(offsets, models):
        block = metrics[metrics["model"] == model].set_index("relation_family").loc[families]
        y = block["mean_oriented_output_effect"].to_numpy(float)
        yerr = np.vstack(
            [y - block["effect_ci_low"].to_numpy(float), block["effect_ci_high"].to_numpy(float) - y]
        )
        ax.errorbar(x + offset, y, yerr=yerr, marker="o", capsize=3, label=model)
    ax.set_xticks(x, families, rotation=25, ha="right")
    ax.set_ylabel("Mean oriented output effect at α=1")
    ax.axhline(0.0, linewidth=0.8)
    ax.legend()
    fig.savefig(figure_dir / "figure_6_relation_family_effects.png", dpi=200)
    plt.close(fig)


def analyze_e01a_trace_mechanism(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    n_bootstraps: int = 2000,
    seed: int = 240817,
) -> dict[str, Any]:
    """Analyze exactly two completed E01A discovery runs and write artifacts."""
    if len(run_dirs) != 2:
        raise ValueError("trace mechanism analysis requires exactly two run directories")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_trace_metrics: list[pd.DataFrame] = []
    all_relation_metrics: list[pd.DataFrame] = []
    all_behavior_metrics: list[pd.DataFrame] = []
    all_regression_metrics: list[pd.DataFrame] = []
    source_payload: dict[str, Any] = {}
    source_comparisons: dict[str, pd.DataFrame] = {}
    integrity: dict[str, Any] = {}
    bootstrap_draws: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    primary_frames: dict[str, pd.DataFrame] = {}

    for run_index, run_dir_value in enumerate(run_dirs):
        run_dir = Path(run_dir_value)
        metrics_json = validate_completed_run(run_dir)
        raw = pd.read_parquet(run_dir / "intervention_rows.parquet")
        trace = pd.read_parquet(run_dir / "trace_rows.parquet")
        model_id = str(metrics_json["model_id"])
        model_label = model_id.rsplit("/", 1)[-1]
        clean, consistency = deduplicate_clean_traces(trace)
        oriented = orient_trace_changes(trace, raw)
        trace_metrics, model_draws = _trace_layer_analysis(
            oriented,
            clean,
            model_label=model_label,
            n_bootstraps=n_bootstraps,
            seed=seed + run_index * 100_000,
        )
        relation_metrics, behavior_metrics = _relation_and_behavior_analysis(
            raw,
            oriented,
            clean,
            model_label=model_label,
            n_bootstraps=n_bootstraps,
            seed=seed + run_index * 100_000 + 50_000,
        )
        source_metrics, comparison, regression_metrics = _source_equivalence_analysis(
            raw,
            model_label=model_label,
            n_bootstraps=n_bootstraps,
            seed=seed + run_index * 100_000 + 70_000,
        )
        all_trace_metrics.append(trace_metrics)
        all_relation_metrics.append(relation_metrics)
        all_behavior_metrics.append(behavior_metrics)
        all_regression_metrics.append(regression_metrics)
        source_payload[model_label] = source_metrics
        source_comparisons[model_label] = comparison
        bootstrap_draws[model_label] = model_draws
        primary_frames[model_label] = oriented[
            (oriented["condition"] == "truth_coordinate")
            & np.isclose(oriented["alpha"].to_numpy(float), 1.0)
        ]
        integrity[model_label] = {
            "run_dir": run_dir.as_posix(),
            "status": metrics_json["status"],
            "confirmation_accessed": metrics_json["confirmation_accessed"],
            "n_base_examples": metrics_json["n_base_examples"],
            "n_pairs": metrics_json["n_pairs"],
            "trace_layers": metrics_json["trace_layers"],
            "alphas": metrics_json["alphas"],
            **consistency,
        }

    trace_metrics = pd.concat(all_trace_metrics, ignore_index=True)
    relation_metrics = pd.concat(all_relation_metrics, ignore_index=True)
    behavior_metrics = pd.concat(all_behavior_metrics, ignore_index=True)
    regression_metrics = pd.concat(all_regression_metrics, ignore_index=True)
    model_labels = sorted(bootstrap_draws)
    if len(model_labels) != 2:
        raise RuntimeError("the two run directories did not resolve to distinct model IDs")
    smaller = next((label for label in model_labels if "0.6B" in label), model_labels[0])
    larger = next((label for label in model_labels if "1.7B" in label), model_labels[1])
    if smaller == larger:
        raise RuntimeError("could not distinguish the two model scales")
    cross_scale: dict[str, Any] = {
        "difference_orientation": f"{larger} minus {smaller}",
        "bootstrap": "independent pair-cluster resampling within model",
        "layers": {},
    }
    primary = trace_metrics[np.isclose(trace_metrics["alpha"].to_numpy(float), 1.0)]
    for layer in EXPECTED_TRACE_LAYERS:
        layer_payload: dict[str, Any] = {}
        for statistic in ("mean_delta_q_z", "mean_delta_m_z", "conversion_beta"):
            left = bootstrap_draws[larger][layer][statistic]
            right = bootstrap_draws[smaller][layer][statistic]
            difference_draws = left - right
            low, high = _interval(difference_draws)
            larger_point = float(
                primary.loc[
                    (primary["model"] == larger) & (primary["trace_layer"] == layer),
                    statistic,
                ].iloc[0]
            )
            smaller_point = float(
                primary.loc[
                    (primary["model"] == smaller) & (primary["trace_layer"] == layer),
                    statistic,
                ].iloc[0]
            )
            layer_payload[statistic] = {
                "difference": larger_point - smaller_point,
                "ci_low": low,
                "ci_high": high,
                larger: larger_point,
                smaller: smaller_point,
            }
        cross_scale["layers"][str(layer)] = layer_payload

    ids_small = set(primary_frames[smaller]["base_sample_id"].astype(str))
    ids_large = set(primary_frames[larger]["base_sample_id"].astype(str))
    cross_scale["sample_ids_identical"] = ids_small == ids_large
    cross_scale["paired_sensitivity"] = {}
    if ids_small == ids_large:
        for layer in EXPECTED_TRACE_LAYERS:
            s_scale = primary.loc[
                (primary["model"] == smaller) & (primary["trace_layer"] == layer),
                ["clean_q_sd", "clean_m_sd"],
            ].iloc[0]
            l_scale = primary.loc[
                (primary["model"] == larger) & (primary["trace_layer"] == layer),
                ["clean_q_sd", "clean_m_sd"],
            ].iloc[0]
            s = primary_frames[smaller]
            s = s[s["trace_layer"] == layer][
                ["base_sample_id", "pair_id", "oriented_delta_truth_coordinate", "oriented_delta_native_margin"]
            ].copy()
            l = primary_frames[larger]
            l = l[l["trace_layer"] == layer][
                ["base_sample_id", "pair_id", "oriented_delta_truth_coordinate", "oriented_delta_native_margin"]
            ].copy()
            s["qz"] = s["oriented_delta_truth_coordinate"] / float(s_scale["clean_q_sd"])
            s["mz"] = s["oriented_delta_native_margin"] / float(s_scale["clean_m_sd"])
            l["qz"] = l["oriented_delta_truth_coordinate"] / float(l_scale["clean_q_sd"])
            l["mz"] = l["oriented_delta_native_margin"] / float(l_scale["clean_m_sd"])
            paired = l.merge(
                s,
                on=["base_sample_id", "pair_id"],
                validate="one_to_one",
                suffixes=("_large", "_small"),
            )
            paired_layer: dict[str, Any] = {}
            for metric in ("qz", "mz"):
                paired["difference"] = paired[f"{metric}_large"] - paired[f"{metric}_small"]
                draws = _cluster_bootstrap_mean_draws(
                    paired,
                    "difference",
                    n_bootstraps=n_bootstraps,
                    seed=seed + 900_000 + layer * 31 + (0 if metric == "qz" else 1),
                )
                low, high = _interval(draws)
                paired_layer[metric] = {
                    "difference": float(paired["difference"].mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            cross_scale["paired_sensitivity"][str(layer)] = paired_layer

    trace_metrics.to_parquet(output_dir / "trace_layer_metrics.parquet", index=False)
    relation_metrics.to_parquet(output_dir / "relation_family_metrics.parquet", index=False)
    behavior_metrics.to_parquet(output_dir / "behavior_strata_metrics.parquet", index=False)
    regression_metrics.to_parquet(output_dir / "source_regression_metrics.parquet", index=False)
    _write_json(output_dir / "cross_scale_metrics.json", cross_scale)
    _write_json(output_dir / "source_equivalence_metrics.json", source_payload)
    _write_json(output_dir / "integrity_checks.json", integrity)
    _plot_trajectories(trace_metrics, figure_dir)
    _plot_source_equivalence(source_comparisons, figure_dir)
    _plot_relation_families(relation_metrics, figure_dir)
    summary = {
        "integrity": integrity,
        "cross_scale": cross_scale,
        "source_equivalence": source_payload,
        "behavior_strata": behavior_metrics.to_dict(orient="records"),
        "relation_families": relation_metrics.to_dict(orient="records"),
    }
    _write_json(output_dir / "analysis_summary.json", summary)
    return summary
