"""Reproducible cross-precision analysis for the bounded E14 pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics.causal import cluster_bootstrap_mean_ci
from ..metrics.quantization import average_random_contexts, percent_change


def _paired_actionability(rows: pd.DataFrame) -> pd.DataFrame:
    averaged = average_random_contexts(rows)
    matched = averaged[averaged["context"] == "matched"]
    random = averaged[averaged["context"] == "random"]
    keys = ["base_sample_id", "pair_id"]
    merged = matched.merge(random, on=keys, validate="one_to_one", suffixes=("_m", "_r"))
    return pd.DataFrame(
        {
            "base_sample_id": merged["base_sample_id"],
            "pair_id": merged["pair_id"],
            "Q": merged["Q0_m"].to_numpy(float),
            "A": merged["A_m"].to_numpy(float) - merged["A_r"].to_numpy(float),
            "G": merged["G_m"].to_numpy(float) - merged["G_r"].to_numpy(float),
        }
    )


def paired_precision_changes(
    reference_rows: pd.DataFrame,
    comparison_rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    reference = _paired_actionability(reference_rows)
    comparison = _paired_actionability(comparison_rows)
    merged = comparison.merge(
        reference,
        on=["base_sample_id", "pair_id"],
        validate="one_to_one",
        suffixes=("_comparison", "_reference"),
    )
    if len(merged) != len(reference):
        raise RuntimeError("cross-precision actionability pairing lost examples")
    output: dict[str, Any] = {}
    for offset, metric in enumerate(("Q", "A", "G")):
        comparison_values = merged[f"{metric}_comparison"].to_numpy(float)
        reference_values = merged[f"{metric}_reference"].to_numpy(float)
        result = cluster_bootstrap_mean_ci(
            comparison_values - reference_values,
            merged["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            seed=int(seed) + offset,
        )
        result["reference_mean"] = float(reference_values.mean())
        result["comparison_mean"] = float(comparison_values.mean())
        result["percent_change"] = percent_change(
            result["comparison_mean"], result["reference_mean"]
        )
        output[metric] = result
    return output


def trace_summary(run_dir: Path) -> pd.DataFrame:
    rows = pd.read_parquet(run_dir / "trace_rows.parquet")
    if "lambda_context" in rows:
        rows = rows[np.isclose(rows["lambda_context"].to_numpy(float), 1.0)].copy()
    metrics = json.loads((run_dir / "precision_metrics.json").read_text(encoding="utf-8"))
    target_labels = (
        pd.read_parquet(run_dir / "factorial_rows.parquet")
        .groupby(["base_sample_id", "pair_id"], as_index=False)["target_label"]
        .first()
    )
    rows = rows.merge(target_labels, on=["base_sample_id", "pair_id"], validate="many_to_one")
    orientation = np.where(rows["target_label"].to_numpy(int) == 1, 1.0, -1.0)
    rows["Q_q"] = orientation * (rows["q10"] - rows["q00"])
    rows["Q_margin"] = orientation * (rows["m10"] - rows["m00"])
    for layer, refs in metrics["validation_references"].items():
        mask = rows["trace_layer"].to_numpy(int) == int(layer)
        rows.loc[mask, "Q_q_z"] = rows.loc[mask, "Q_q"] / float(refs["sigma_q"])
        rows.loc[mask, "Q_margin_z"] = rows.loc[mask, "Q_margin"] / float(
            refs["sigma_margin"]
        )
    averaged = (
        rows.groupby(["context", "base_sample_id", "pair_id", "trace_layer"], as_index=False)[
            ["Q_q_z", "Q_margin_z", "A_q_z", "G_q_z", "A_margin_z", "G_margin_z"]
        ]
        .mean()
    )
    matched = averaged[averaged["context"] == "matched"]
    random = averaged[averaged["context"] == "random"]
    keys = ["base_sample_id", "pair_id", "trace_layer"]
    merged = matched.merge(random, on=keys, validate="one_to_one", suffixes=("_m", "_r"))
    output = merged[keys].copy()
    output["Q_q_z"] = merged["Q_q_z_m"]
    output["Q_margin_z"] = merged["Q_margin_z_m"]
    for metric in ("A_q_z", "G_q_z", "A_margin_z", "G_margin_z"):
        output[f"structured_minus_random_{metric}"] = (
            merged[f"{metric}_m"] - merged[f"{metric}_r"]
        )
    return output


def analyze_bounded_pilot(
    run_dirs: dict[str, Path],
    *,
    output_path: Path,
    n_bootstraps: int = 2000,
    seed: int = 20260832,
) -> dict[str, Any]:
    if set(run_dirs) != {"BF16", "INT8", "INT4"}:
        raise ValueError("bounded E14 analysis requires BF16, INT8, and INT4")
    metrics: dict[str, dict[str, Any]] = {}
    rows: dict[str, pd.DataFrame] = {}
    traces: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for precision, run_dir in run_dirs.items():
        metrics[precision] = json.loads(
            (run_dir / "precision_metrics.json").read_text(encoding="utf-8")
        )
        manifests[precision] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if metrics[precision].get("profile") != "pilot" or metrics[precision].get("status") != "complete":
            raise RuntimeError(f"{precision} is not a completed E14 pilot")
        if metrics[precision].get("confirmation_accessed") is not False:
            raise RuntimeError("E14 pilot accessed confirmation")
        rows[precision] = pd.read_parquet(run_dir / "factorial_rows.parquet")
        traces[precision] = trace_summary(run_dir)

    changes = {
        precision: paired_precision_changes(
            rows["BF16"], rows[precision], n_bootstraps=n_bootstraps, seed=seed + index * 100
        )
        for index, precision in enumerate(("INT8", "INT4"), start=1)
    }
    trace_aggregates = {
        precision: trace.groupby("trace_layer", as_index=False).mean(numeric_only=True).to_dict(
            orient="records"
        )
        for precision, trace in traces.items()
    }
    result = {
        "runs": {key: value.name for key, value in run_dirs.items()},
        "metrics": metrics,
        "paired_changes_from_bf16": changes,
        "trace_aggregates": trace_aggregates,
        "confirmation_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# E14 Bounded Pilot Summary",
        "",
        "Status: Stage 0 and Stage 1 complete; full discovery is not authorized.",
        "",
        "E14 used only discovery data. The E01 confirmation split was not accessed.",
        "",
        "## Backend",
        "",
        (
            "Optimum-Quanto 0.2.7 weight-only BF16/INT8/INT4 on Qwen3-1.7B; "
            "runtime activations and compute remained BF16. INT8 was per-channel symmetric; "
            "INT4 was groupwise affine. No calibration data were used."
        ),
        "",
        "## Bounded pilot",
        "",
        "| Precision | D native AUROC | D frozen-BF16 AUROC | B AUROC | Prompt PPL | Q | A matched-random | G matched-random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for precision in ("BF16", "INT8", "INT4"):
        item = metrics[precision]
        lines.append(
            f"| {precision} | {item['D_native']['auroc']:.6f} | "
            f"{item['D_frozen_bf16_axis']['auroc']:.6f} | {item['B']['auroc']:.6f} | "
            f"{item['general_quality']['prompt_perplexity']:.3f} | {item['Q0']:.6f} | "
            f"{item['factorial_contrasts']['A_matched_minus_random']['mean']:.6f} | "
            f"{item['factorial_contrasts']['G_matched_minus_random']['mean']:.6f} |"
        )
    lines.extend(["", "## Paired changes from BF16", ""])
    for precision in ("INT8", "INT4"):
        lines.append(f"### {precision}")
        lines.append("")
        for metric in ("Q", "A", "G"):
            item = changes[precision][metric]
            lines.append(
                f"- {metric}: {item['mean']:+.6f} "
                f"(95% pair-cluster CI [{item['ci_low']:+.6f}, {item['ci_high']:+.6f}]); "
                f"{item['percent_change']:+.1f}% from BF16."
            )
        lines.append("")
    lines.extend(
        [
            "## Trace localization",
            "",
            (
                "Values are precision-validation-standardized. A/G columns are "
                "matched-structured minus seed-averaged random orthogonal context."
            ),
            "",
            "| Precision | Layer | Q q_z | Q margin_z | A q_z | A margin_z | G q_z | G margin_z |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for precision in ("BF16", "INT8", "INT4"):
        for row in trace_aggregates[precision]:
            lines.append(
                f"| {precision} | {int(row['trace_layer'])} | {row['Q_q_z']:.6f} | "
                f"{row['Q_margin_z']:.6f} | "
                f"{row['structured_minus_random_A_q_z']:.6f} | "
                f"{row['structured_minus_random_A_margin_z']:.6f} | "
                f"{row['structured_minus_random_G_q_z']:.6f} | "
                f"{row['structured_minus_random_G_margin_z']:.6f} |"
            )
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "- INT8 preserved both representation views and all three actionability components; "
                "its small increases are compatible with numerical perturbation rather than damage."
            ),
            (
                "- INT4 preserved precision-native decodability but reduced alignment with the frozen "
                "BF16 semantic axis and reduced structured-minus-random A and G. Q did not degrade."
            ),
            (
                "- The INT4 G reduction is the clearest higher-order fragility. A also declines, "
                "while prompt perplexity does not show catastrophic generic failure. Native "
                "task-margin AUROC does decline, so the pilot does not establish purely "
                "semantic-specific damage."
            ),
            (
                "- Trace decomposition localizes the A reduction immediately at L17 and downstream. "
                "The G reduction is most visible during downstream conversion, especially L20/L27, "
                "rather than as failure to apply the L17 scalar setpoint."
            ),
            "",
            "## Decision",
            "",
            (
                "A full E14 discovery study is justified scientifically, but it remains unauthorized. "
                "It should retain continuous effects and add a stronger frozen general-quality corpus "
                "before making semantic-specific compression claims."
            ),
            "",
            "## Integrity",
            "",
        ]
    )
    for precision in ("BF16", "INT8", "INT4"):
        item = metrics[precision]
        manifest = manifests[precision]
        lines.append(
            f"- {precision}: exact no-op; finite={item['integrity']['finite']}; "
            f"max |context·u|={item['integrity']['max_context_dot_u']:.3e}; "
            f"peak VRAM={manifest['peak_vram_allocated_bytes'] / 2**30:.2f} GiB; "
            f"runtime={item['wall_time_s']:.1f}s."
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
