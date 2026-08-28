"""Reproducible locked post-processing for the completed E01 confirmation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.confirmation import pair_cluster_bootstrap_ci, paired_condition_values
from ..metrics.setpoint import summarize_grid_response
from ..reporting.tables import save_json, save_table

DISCOVERY_RUNS = {
    "Qwen/Qwen3-0.6B": {
        "scalar": "runs/E01B1/E01B1_e1169f3ffe11/intervention_rows.parquet",
        "factorial": "runs/E01B3/E01B3_84ee4bae8564/factorial_rows.parquet",
    },
    "Qwen/Qwen3-1.7B": {
        "scalar": "runs/E01B1/E01B1_5b9d70c8cffe/intervention_rows.parquet",
        "factorial": "runs/E01B3/E01B3_e7462847a558/factorial_rows.parquet",
    },
}


def _effect(
    rows: pd.DataFrame,
    *,
    left: str,
    right: str | None,
    metric: str,
    seed: int,
) -> dict[str, float | pd.DataFrame]:
    paired = paired_condition_values(rows, left=left, right=right, metric=metric)
    estimate, low, high = pair_cluster_bootstrap_ci(
        paired["effect"], paired["pair_id"], n_draws=10_000, seed=seed
    )
    return {"estimate": estimate, "ci_low": low, "ci_high": high, "rows": paired}


def _model_effects(scalar: pd.DataFrame, factorial: pd.DataFrame, *, seed: int):
    primary_factorial = factorial[np.isclose(factorial["lambda_context"], 1.0)]
    specs = {
        "Q0": (scalar, "source_free_opposite_class_median", None, "delta_margin_toward_target"),
        "A_matched": (primary_factorial, "matched_orthogonal", None, "A_context"),
        "A_random": (primary_factorial, "random_orthogonal", None, "A_context"),
        "A_structured_minus_random": (
            primary_factorial,
            "matched_orthogonal",
            "random_orthogonal",
            "A_context",
        ),
        "G_matched": (primary_factorial, "matched_orthogonal", None, "G_interaction"),
        "G_random": (primary_factorial, "random_orthogonal", None, "G_interaction"),
        "G_structured_minus_random": (
            primary_factorial,
            "matched_orthogonal",
            "random_orthogonal",
            "G_interaction",
        ),
    }
    return {
        name: _effect(
            frame,
            left=left,
            right=right,
            metric=metric,
            seed=seed + index * 101,
        )
        for index, (name, (frame, left, right, metric)) in enumerate(specs.items())
    }


def _cross_scale(effects, metric: str, *, seed: int):
    large = effects["Qwen/Qwen3-1.7B"][metric]["rows"].rename(
        columns={"effect": "large", "pair_id": "pair_large"}
    )
    small = effects["Qwen/Qwen3-0.6B"][metric]["rows"].rename(
        columns={"effect": "small", "pair_id": "pair_small"}
    )
    merged = large.merge(small, on="base_sample_id", validate="one_to_one")
    if not (merged["pair_large"].astype(str) == merged["pair_small"].astype(str)).all():
        raise RuntimeError("cross-scale confirmation comparison lost pair identity")
    difference = merged["large"] - merged["small"]
    estimate, low, high = pair_cluster_bootstrap_ci(
        difference, merged["pair_large"], n_draws=10_000, seed=seed
    )
    return {"estimate": estimate, "ci_low": low, "ci_high": high}


def analyze_confirmation(repo_root: Path, campaign_dir: Path) -> dict:
    primary = pd.read_parquet(campaign_dir / "primary_hypotheses.parquet")
    confirmation_scalar = pd.read_parquet(campaign_dir / "scalar_rows.parquet")
    confirmation_factorial = pd.read_parquet(campaign_dir / "factorial_rows.parquet")
    trace = pd.read_parquet(campaign_dir / "trace_rows.parquet")
    models = tuple(DISCOVERY_RUNS)
    discovery_effects = {}
    confirmation_effects = {}
    for index, model in enumerate(models):
        discovery_effects[model] = _model_effects(
            pd.read_parquet(repo_root / DISCOVERY_RUNS[model]["scalar"]),
            pd.read_parquet(repo_root / DISCOVERY_RUNS[model]["factorial"]),
            seed=20260831 + index * 1000,
        )
        confirmation_effects[model] = _model_effects(
            confirmation_scalar[confirmation_scalar["model_id"] == model],
            confirmation_factorial[confirmation_factorial["model_id"] == model],
            seed=20260831 + index * 1000,
        )
    discovery_cross = _cross_scale(discovery_effects, "G_structured_minus_random", seed=20268831)
    confirmation_cross = _cross_scale(
        confirmation_effects, "G_structured_minus_random", seed=20268831
    )

    comparison: list[dict] = []
    for model in models:
        for metric in discovery_effects[model]:
            discovery = discovery_effects[model][metric]
            confirmation = confirmation_effects[model][metric]
            ratio = (
                float(confirmation["estimate"]) / float(discovery["estimate"])
                if abs(float(discovery["estimate"])) > 1e-12
                else np.nan
            )
            comparison.append(
                {
                    "model_id": model,
                    "metric": metric,
                    "discovery_estimate": discovery["estimate"],
                    "discovery_ci_low": discovery["ci_low"],
                    "discovery_ci_high": discovery["ci_high"],
                    "confirmation_estimate": confirmation["estimate"],
                    "confirmation_ci_low": confirmation["ci_low"],
                    "confirmation_ci_high": confirmation["ci_high"],
                    "same_sign": bool(
                        np.sign(discovery["estimate"]) == np.sign(confirmation["estimate"])
                    ),
                    "effect_size_ratio": ratio,
                    "ci_overlap_descriptive": bool(
                        max(discovery["ci_low"], confirmation["ci_low"])
                        <= min(discovery["ci_high"], confirmation["ci_high"])
                    ),
                }
            )
    comparison.append(
        {
            "model_id": "1.7B-minus-0.6B",
            "metric": "G_structured_minus_random_cross_scale",
            "discovery_estimate": discovery_cross["estimate"],
            "discovery_ci_low": discovery_cross["ci_low"],
            "discovery_ci_high": discovery_cross["ci_high"],
            "confirmation_estimate": confirmation_cross["estimate"],
            "confirmation_ci_low": confirmation_cross["ci_low"],
            "confirmation_ci_high": confirmation_cross["ci_high"],
            "same_sign": bool(
                np.sign(discovery_cross["estimate"]) == np.sign(confirmation_cross["estimate"])
            ),
            "effect_size_ratio": confirmation_cross["estimate"] / discovery_cross["estimate"],
            "ci_overlap_descriptive": bool(
                max(discovery_cross["ci_low"], confirmation_cross["ci_low"])
                <= min(discovery_cross["ci_high"], confirmation_cross["ci_high"])
            ),
        }
    )
    comparison_frame = pd.DataFrame(comparison)
    save_table(comparison_frame, campaign_dir / "discovery_confirmation_comparison.parquet")

    relation_records = []
    primary_factorial = confirmation_factorial[
        np.isclose(confirmation_factorial["lambda_context"], 1.0)
    ]
    for index, ((model, family, condition), block) in enumerate(
        primary_factorial.groupby(["model_id", "relation_family", "condition"], sort=True)
    ):
        for metric in ("A_context", "G_interaction"):
            estimate, low, high = pair_cluster_bootstrap_ci(
                block[metric],
                block["pair_id"],
                n_draws=10_000,
                seed=20270831 + index * 31 + (0 if metric == "A_context" else 13),
            )
            relation_records.append(
                {
                    "model_id": model,
                    "relation_family": family,
                    "condition": condition,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_examples": len(block),
                    "n_pairs": block["pair_id"].nunique(),
                }
            )
    relation = pd.DataFrame(relation_records)
    save_table(relation, campaign_dir / "secondary_relation_family.parquet")

    factorial_trace = trace[
        (trace["artifact_family"] == "factorial") & np.isclose(trace["lambda_context"], 1.0)
    ]
    random_trace = factorial_trace[factorial_trace["condition"] == "random_orthogonal"]
    nonrandom_trace = factorial_trace[factorial_trace["condition"] != "random_orthogonal"]
    random_trace = random_trace.groupby(
        ["model_id", "base_sample_id", "pair_id", "condition", "trace_layer"],
        as_index=False,
    )[["A_q_z", "G_q_z", "A_margin_z", "G_margin_z"]].mean()
    trace_averaged = pd.concat([nonrandom_trace, random_trace], ignore_index=True)
    trace_metrics = trace_averaged.groupby(
        ["model_id", "condition", "trace_layer"], as_index=False
    )[["A_q_z", "G_q_z", "A_margin_z", "G_margin_z"]].mean()
    save_table(trace_metrics, campaign_dir / "secondary_trace_metrics.parquet")

    grid_records = []
    grid_summaries = {}
    for index, model in enumerate(models):
        grid = confirmation_scalar[
            (confirmation_scalar["model_id"] == model)
            & (confirmation_scalar["condition"] == "source_free_grid")
        ]
        targets, examples, summary = summarize_grid_response(
            grid,
            n_bootstraps=10_000,
            confidence_level=0.95,
            seed=20280831 + index * 1000,
        )
        targets["model_id"] = model
        examples["model_id"] = model
        grid_records.append(targets)
        save_table(examples, campaign_dir / f"secondary_grid_examples_{index}.parquet")
        grid_summaries[model] = summary
    save_table(
        pd.concat(grid_records, ignore_index=True), campaign_dir / "secondary_grid_metrics.parquet"
    )
    save_json(grid_summaries, campaign_dir / "secondary_grid_summary.json")

    classification = (
        "strong"
        if bool(primary["verdict"].eq("PASS").all())
        else "partial"
        if bool(primary.set_index("hypothesis").loc[["H1", "H2", "H3"], "verdict"].eq("PASS").all())
        else "failed"
    )
    result = {
        "classification": classification,
        "primary": primary.to_dict(orient="records"),
        "comparison": comparison_frame.to_dict(orient="records"),
        "grid": grid_summaries,
    }
    save_json(result, campaign_dir / "confirmation_analysis.json")
    return result


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    campaign = repo_root / "runs/CONFIRMATION/CONFIRMATION_46312baf5992"
    analyze_confirmation(repo_root, campaign)


if __name__ == "__main__":
    main()
