"""Preregistered inference for the E14 precision confirmation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .confirmation import cluster_sign_flip_pvalue, pair_cluster_bootstrap_ci
from .quantization import average_random_contexts


def holm_adjust_family(p_values: Mapping[str, float]) -> dict[str, float]:
    if set(p_values) != {"H14.1", "H14.2", "H14.3"}:
        raise ValueError("E14 Holm family must contain exactly H14.1-H14.3")
    ordered = sorted(p_values.items(), key=lambda item: float(item[1]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * float(value)))
        adjusted[name] = running
    return adjusted


def _actionability(rows: pd.DataFrame) -> pd.DataFrame:
    averaged = average_random_contexts(rows)
    matched = averaged[averaged["context"] == "matched"]
    random = averaged[averaged["context"] == "random"]
    keys = ["base_sample_id", "pair_id"]
    joined = matched.merge(random, on=keys, validate="one_to_one", suffixes=("_m", "_r"))
    return pd.DataFrame(
        {
            "base_sample_id": joined["base_sample_id"],
            "pair_id": joined["pair_id"],
            "Q": joined["Q0_m"].to_numpy(float),
            "A": joined["A_m"].to_numpy(float) - joined["A_r"].to_numpy(float),
            "G": joined["G_m"].to_numpy(float) - joined["G_r"].to_numpy(float),
        }
    )


def _paired_degradation(
    bf16: pd.DataFrame,
    int4: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    keys = ["base_sample_id", "pair_id"]
    merged = bf16[keys + [metric]].merge(
        int4[keys + [metric]], on=keys, validate="one_to_one", suffixes=("_bf16", "_int4")
    )
    merged["effect"] = merged[f"{metric}_bf16"] - merged[f"{metric}_int4"]
    return merged[keys + ["effect"]]


def clustered_auroc_floor(
    rows: pd.DataFrame,
    *,
    floor: float,
    n_draws: int,
    seed: int,
) -> dict[str, float]:
    labels = rows["gold_label"].to_numpy(int)
    observed = float(roc_auc_score(labels, rows["native_probe_score"].to_numpy(float)))
    clusters = sorted(rows["pair_id"].astype(str).unique())
    blocks = {key: rows[rows["pair_id"].astype(str) == key] for key in clusters}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), dtype=np.float64)
    for index in range(len(draws)):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        draw = pd.concat([blocks[key] for key in selected])
        draws[index] = roc_auc_score(
            draw["gold_label"].to_numpy(int), draw["native_probe_score"].to_numpy(float)
        )
    return {
        "estimate": observed - float(floor),
        "auroc": observed,
        "ci_low": float(np.quantile(draws, 0.025)) - float(floor),
        "ci_high": float(np.quantile(draws, 0.975)) - float(floor),
        "raw_p": float((np.sum(draws < float(floor)) + 1) / (len(draws) + 1)),
    }


def evaluate_e14_hypotheses(
    behavior_by_precision: Mapping[str, pd.DataFrame],
    factorial_by_precision: Mapping[str, pd.DataFrame],
    *,
    bootstrap_draws: int = 10_000,
    randomization_draws: int = 100_000,
    seed: int = 20261404,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate the exact H14.1-H14.3 family on lambda-one paired rows."""
    h1 = clustered_auroc_floor(
        behavior_by_precision["int4"], floor=0.99, n_draws=bootstrap_draws, seed=seed
    )
    action = {key: _actionability(value) for key, value in factorial_by_precision.items()}
    details: dict[str, pd.DataFrame] = {}
    records = [
        {
            "hypothesis": "H14.1",
            "estimate": h1["estimate"],
            "ci_low": h1["ci_low"],
            "ci_high": h1["ci_high"],
            "raw_p": h1["raw_p"],
            "criterion": "D_native_INT4 - 0.99",
        }
    ]
    for offset, (name, metric) in enumerate((("H14.2", "G"), ("H14.3", "A")), start=1):
        paired = _paired_degradation(action["bf16"], action["int4"], metric)
        estimate, low, high = pair_cluster_bootstrap_ci(
            paired["effect"],
            paired["pair_id"],
            n_draws=bootstrap_draws,
            seed=seed + offset,
        )
        raw_p = cluster_sign_flip_pvalue(
            paired["effect"],
            paired["pair_id"],
            n_draws=randomization_draws,
            seed=seed + 100 + offset,
        )
        details[name] = paired
        records.append(
            {
                "hypothesis": name,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "raw_p": raw_p,
                "criterion": f"BF16 - INT4 {metric} matched-random",
            }
        )
    frame = pd.DataFrame(records)
    adjusted = holm_adjust_family(dict(zip(frame["hypothesis"], frame["raw_p"])))
    frame["holm_p"] = frame["hypothesis"].map(adjusted)
    frame["verdict"] = np.where(
        (frame["estimate"] >= 0) & (frame["holm_p"] < 0.05), "PASS", "FAIL"
    )
    return frame, details


def classify_e14_confirmation(primary: pd.DataFrame) -> str:
    verdicts = dict(zip(primary["hypothesis"], primary["verdict"]))
    if all(verdicts.get(name) == "PASS" for name in ("H14.1", "H14.2", "H14.3")):
        return "strong"
    if verdicts.get("H14.1") == "PASS" and verdicts.get("H14.2") == "PASS":
        return "partial"
    return "failed"
