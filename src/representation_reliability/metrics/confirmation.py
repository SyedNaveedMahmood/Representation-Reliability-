"""Locked primary inference for the E01 actionability confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

PRIMARY_HYPOTHESES = ("H1", "H2", "H3", "H4")


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values without changing family size."""
    if set(p_values) != set(PRIMARY_HYPOTHESES):
        raise ValueError("Holm family must contain exactly H1-H4")
    ordered = sorted(((str(k), float(v)) for k, v in p_values.items()), key=lambda x: x[1])
    if any(not 0.0 <= value <= 1.0 for _key, value in ordered):
        raise ValueError("p-values must be finite values in [0, 1]")
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (size - rank) * value))
        adjusted[key] = running
    return adjusted


def _pair_means(values: Sequence[float], pair_ids: Sequence[str]) -> np.ndarray:
    frame = pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "pair_id": list(map(str, pair_ids))}
    )
    if frame.empty or not np.isfinite(frame["value"].to_numpy()).all():
        raise ValueError("cluster values must be non-empty and finite")
    return frame.groupby("pair_id", sort=True)["value"].mean().to_numpy(float)


def pair_cluster_bootstrap_ci(
    values: Sequence[float],
    pair_ids: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and percentile CI after resampling pair-level means."""
    pairs = _pair_means(values, pair_ids)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), dtype=np.float64)
    for index in range(len(draws)):
        draws[index] = float(np.mean(rng.choice(pairs, size=len(pairs), replace=True)))
    tail = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(np.mean(pairs)), float(low), float(high)


def cluster_sign_flip_pvalue(
    values: Sequence[float],
    pair_ids: Sequence[str],
    *,
    n_draws: int,
    seed: int,
) -> float:
    """One-sided pair-cluster sign-flip p-value for a positive mean."""
    pairs = _pair_means(values, pair_ids)
    observed = float(np.mean(pairs))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    remaining = int(n_draws)
    while remaining:
        count = min(4096, remaining)
        signs = rng.integers(0, 2, size=(count, len(pairs)), dtype=np.int8) * 2 - 1
        draws = np.mean(signs * pairs[None, :], axis=1)
        exceed += int(np.sum(draws >= observed))
        remaining -= count
    return float((exceed + 1) / (int(n_draws) + 1))


def _by_base(rows: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    block = rows[rows["condition"].astype(str) == condition]
    if block.empty:
        raise RuntimeError(f"missing confirmation condition {condition}")
    return block.groupby(["base_sample_id", "pair_id"], as_index=False)[metric].mean()


def paired_condition_values(
    rows: pd.DataFrame,
    *,
    left: str,
    right: str | None,
    metric: str,
) -> pd.DataFrame:
    """Build a directed-example paired effect, averaging frozen random seeds."""
    lhs = _by_base(rows, left, metric).rename(columns={metric: "left"})
    if right is None:
        lhs["effect"] = lhs["left"]
        return lhs[["base_sample_id", "pair_id", "effect"]]
    rhs = _by_base(rows, right, metric).rename(
        columns={metric: "right", "pair_id": "pair_id_right"}
    )
    paired = lhs.merge(rhs, on="base_sample_id", validate="one_to_one")
    if len(paired) != len(lhs) or len(paired) != len(rhs):
        raise RuntimeError(f"paired contrast lost rows: {left} - {right}")
    if not (paired["pair_id"].astype(str) == paired["pair_id_right"].astype(str)).all():
        raise RuntimeError("paired contrast pair identity mismatch")
    paired["effect"] = paired["left"] - paired["right"]
    return paired[["base_sample_id", "pair_id", "effect"]]


def _component(
    rows: pd.DataFrame,
    *,
    left: str,
    right: str | None,
    metric: str,
    n_bootstraps: int,
    n_randomizations: int,
    seed: int,
) -> dict[str, object]:
    paired = paired_condition_values(rows, left=left, right=right, metric=metric)
    estimate, low, high = pair_cluster_bootstrap_ci(
        paired["effect"], paired["pair_id"], n_draws=n_bootstraps, seed=seed
    )
    p_value = cluster_sign_flip_pvalue(
        paired["effect"], paired["pair_id"], n_draws=n_randomizations, seed=seed + 1
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "raw_p": p_value,
        "pair_values": paired,
    }


def _composite_min_ci(
    components: Sequence[dict[str, object]], *, n_draws: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(n_draws), len(components)), dtype=np.float64)
    for column, component in enumerate(components):
        frame = component["pair_values"]
        assert isinstance(frame, pd.DataFrame)
        pairs = _pair_means(frame["effect"], frame["pair_id"])
        local = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        for row in range(int(n_draws)):
            draws[row, column] = float(np.mean(local.choice(pairs, len(pairs), replace=True)))
    low, high = np.quantile(np.min(draws, axis=1), [0.025, 0.975])
    return float(low), float(high)


def evaluate_primary_hypotheses(
    scalar_by_model: Mapping[str, pd.DataFrame],
    factorial_by_model: Mapping[str, pd.DataFrame],
    *,
    n_bootstraps: int = 10_000,
    n_randomizations: int = 100_000,
    seed: int = 20260831,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, float | str]]]]:
    """Evaluate the exactly four preregistered H1-H4 hypotheses."""
    models = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
    if set(scalar_by_model) != set(models) or set(factorial_by_model) != set(models):
        raise ValueError("confirmation inference requires both frozen checkpoints")

    h1_components: list[dict[str, object]] = []
    details: dict[str, list[dict[str, float | str]]] = {key: [] for key in PRIMARY_HYPOTHESES}
    for model_index, model in enumerate(models):
        scalar = scalar_by_model[model]
        for item_index, (left, right, label) in enumerate(
            (
                ("source_free_opposite_class_median", None, "Q0"),
                ("source_free_opposite_class_median", "random_direction", "Q0-random"),
                ("source_free_opposite_class_median", "orthogonal_random", "Q0-orthogonal"),
            )
        ):
            component = _component(
                scalar,
                left=left,
                right=right,
                metric="delta_margin_toward_target",
                n_bootstraps=n_bootstraps,
                n_randomizations=n_randomizations,
                seed=seed + model_index * 1000 + item_index * 31,
            )
            h1_components.append(component)
            details["H1"].append(
                {
                    "model": model,
                    "component": label,
                    "estimate": float(component["estimate"]),
                    "ci_low": float(component["ci_low"]),
                    "ci_high": float(component["ci_high"]),
                    "raw_p": float(component["raw_p"]),
                }
            )

    h2_components: list[dict[str, object]] = []
    for model_index, model in enumerate(models):
        primary_factorial = factorial_by_model[model][
            np.isclose(factorial_by_model[model]["lambda_context"].to_numpy(float), 1.0)
        ]
        component = _component(
            primary_factorial,
            left="matched_orthogonal",
            right="random_orthogonal",
            metric="A_context",
            n_bootstraps=n_bootstraps,
            n_randomizations=n_randomizations,
            seed=seed + 3000 + model_index * 101,
        )
        h2_components.append(component)
        details["H2"].append(
            {
                "model": model,
                "component": "A_matched-A_random",
                "estimate": float(component["estimate"]),
                "ci_low": float(component["ci_low"]),
                "ci_high": float(component["ci_high"]),
                "raw_p": float(component["raw_p"]),
            }
        )

    h3 = _component(
        factorial_by_model[models[1]][
            np.isclose(factorial_by_model[models[1]]["lambda_context"].to_numpy(float), 1.0)
        ],
        left="matched_orthogonal",
        right="random_orthogonal",
        metric="G_interaction",
        n_bootstraps=n_bootstraps,
        n_randomizations=n_randomizations,
        seed=seed + 4000,
    )
    details["H3"].append(
        {
            "model": models[1],
            "component": "G_matched-G_random",
            "estimate": float(h3["estimate"]),
            "ci_low": float(h3["ci_low"]),
            "ci_high": float(h3["ci_high"]),
            "raw_p": float(h3["raw_p"]),
        }
    )

    model_g: dict[str, pd.DataFrame] = {}
    for model in models:
        primary_factorial = factorial_by_model[model][
            np.isclose(factorial_by_model[model]["lambda_context"].to_numpy(float), 1.0)
        ]
        model_g[model] = paired_condition_values(
            primary_factorial,
            left="matched_orthogonal",
            right="random_orthogonal",
            metric="G_interaction",
        )
    left = model_g[models[1]].rename(columns={"effect": "large", "pair_id": "pair_large"})
    right = model_g[models[0]].rename(columns={"effect": "small", "pair_id": "pair_small"})
    cross = left.merge(right, on="base_sample_id", validate="one_to_one")
    if len(cross) != len(left) or len(cross) != len(right):
        raise RuntimeError("H4 semantic sample identities are not identical")
    if not (cross["pair_large"].astype(str) == cross["pair_small"].astype(str)).all():
        raise RuntimeError("H4 pair identities are not identical")
    cross["effect"] = cross["large"] - cross["small"]
    h4_estimate, h4_low, h4_high = pair_cluster_bootstrap_ci(
        cross["effect"], cross["pair_large"], n_draws=n_bootstraps, seed=seed + 5000
    )
    h4_p = cluster_sign_flip_pvalue(
        cross["effect"], cross["pair_large"], n_draws=n_randomizations, seed=seed + 5001
    )
    details["H4"].append(
        {
            "model": "1.7B-minus-0.6B",
            "component": "structured interaction checkpoint difference",
            "estimate": h4_estimate,
            "ci_low": h4_low,
            "ci_high": h4_high,
            "raw_p": h4_p,
        }
    )

    h1_low, h1_high = _composite_min_ci(h1_components, n_draws=n_bootstraps, seed=seed + 6000)
    h2_low, h2_high = _composite_min_ci(h2_components, n_draws=n_bootstraps, seed=seed + 7000)
    records = [
        {
            "hypothesis": "H1",
            "description": "scalar actionability and control separation in both checkpoints",
            "estimate": min(float(item["estimate"]) for item in h1_components),
            "ci_low": h1_low,
            "ci_high": h1_high,
            "raw_p": max(float(item["raw_p"]) for item in h1_components),
            "all_component_estimates_positive": all(
                float(item["estimate"]) > 0 for item in h1_components
            ),
        },
        {
            "hypothesis": "H2",
            "description": "matched additive signal exceeds random in both checkpoints",
            "estimate": min(float(item["estimate"]) for item in h2_components),
            "ci_low": h2_low,
            "ci_high": h2_high,
            "raw_p": max(float(item["raw_p"]) for item in h2_components),
            "all_component_estimates_positive": all(
                float(item["estimate"]) > 0 for item in h2_components
            ),
        },
        {
            "hypothesis": "H3",
            "description": "structured q-by-context interaction in Qwen3-1.7B",
            "estimate": float(h3["estimate"]),
            "ci_low": float(h3["ci_low"]),
            "ci_high": float(h3["ci_high"]),
            "raw_p": float(h3["raw_p"]),
            "all_component_estimates_positive": float(h3["estimate"]) > 0,
        },
        {
            "hypothesis": "H4",
            "description": "structured interaction is larger in 1.7B than 0.6B",
            "estimate": h4_estimate,
            "ci_low": h4_low,
            "ci_high": h4_high,
            "raw_p": h4_p,
            "all_component_estimates_positive": h4_estimate > 0,
        },
    ]
    frame = pd.DataFrame(records)
    adjusted = holm_adjust(dict(zip(frame["hypothesis"], frame["raw_p"])))
    frame["holm_p"] = frame["hypothesis"].map(adjusted).astype(float)
    frame["verdict"] = np.where(
        (frame["holm_p"] < 0.05) & frame["all_component_estimates_positive"], "PASS", "FAIL"
    )
    return frame, details


def confirmation_classification(primary: pd.DataFrame) -> str:
    verdict = dict(zip(primary["hypothesis"], primary["verdict"]))
    if set(verdict) != set(PRIMARY_HYPOTHESES):
        raise ValueError("classification requires H1-H4")
    if all(verdict[key] == "PASS" for key in PRIMARY_HYPOTHESES):
        return "strong"
    if all(verdict[key] == "PASS" for key in ("H1", "H2", "H3")):
        return "partial"
    return "failed"
