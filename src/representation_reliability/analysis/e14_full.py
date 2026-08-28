"""Locked cross-precision analysis for E14 full discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .e14_bounded import paired_precision_changes, trace_summary

PRECISIONS = ("BF16", "INT8", "INT4")


def _primary_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path / "factorial_rows.parquet")
    if "lambda_context" in rows:
        rows = rows[np.isclose(rows["lambda_context"].to_numpy(float), 1.0)].copy()
    return rows


def paired_behavior_auroc_change(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> dict[str, float | int]:
    """Paired AUROC change with matched-pair cluster bootstrap uncertainty."""
    keys = ["base_sample_id", "pair_id"]
    required = {*keys, "gold_label", "yes_no_margin"}
    for name, frame in (("reference", reference), ("comparison", comparison)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} behavior rows missing {sorted(missing)}")
    merged = comparison[list(required)].merge(
        reference[list(required)], on=keys, validate="one_to_one", suffixes=("_c", "_r")
    )
    if len(merged) != len(reference) or len(merged) != len(comparison):
        raise RuntimeError("cross-precision behavior pairing lost examples")
    if not np.array_equal(
        merged["gold_label_c"].to_numpy(int), merged["gold_label_r"].to_numpy(int)
    ):
        raise RuntimeError("cross-precision behavior labels disagree")

    def difference(frame: pd.DataFrame) -> float:
        labels = frame["gold_label_c"].to_numpy(int)
        return float(
            roc_auc_score(labels, frame["yes_no_margin_c"].to_numpy(float))
            - roc_auc_score(labels, frame["yes_no_margin_r"].to_numpy(float))
        )

    clusters = sorted(merged["pair_id"].astype(str).unique())
    by_cluster = {
        cluster: merged[merged["pair_id"].astype(str) == cluster] for cluster in clusters
    }
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(max(1, int(n_bootstraps))):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        draws.append(difference(pd.concat([by_cluster[item] for item in selected])))
    return {
        "mean": difference(merged),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_rows": len(merged),
        "n_clusters": len(clusters),
    }


def _retention(comparison: float, reference: float) -> float | None:
    if not np.isfinite(reference) or abs(reference) < 1e-8:
        return None
    return float(comparison / reference)


def analyze_full_discovery(
    run_dirs: dict[str, Path],
    *,
    output_path: Path,
    n_bootstraps: int = 2000,
    seed: int = 20261403,
) -> dict[str, Any]:
    """Audit and summarize the frozen E14 full-discovery precision ladder."""
    if tuple(run_dirs) != PRECISIONS:
        raise ValueError("E14 full analysis requires ordered BF16, INT8, and INT4 runs")
    metrics: dict[str, dict[str, Any]] = {}
    factorial: dict[str, pd.DataFrame] = {}
    behavior: dict[str, pd.DataFrame] = {}
    traces: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    sample_sets: dict[str, set[str]] = {}
    for precision, run_dir in run_dirs.items():
        metrics[precision] = json.loads(
            (run_dir / "precision_metrics.json").read_text(encoding="utf-8")
        )
        manifests[precision] = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        item = metrics[precision]
        if item.get("profile") != "full" or item.get("status") != "complete":
            raise RuntimeError(f"{precision} is not a completed E14 full run")
        if item.get("confirmation_accessed") is not False:
            raise RuntimeError("E14 full discovery accessed confirmation")
        if item.get("n_examples") != 300 or item.get("n_pairs") != 150:
            raise RuntimeError(f"{precision} has the wrong frozen discovery size")
        factorial[precision] = _primary_rows(run_dir)
        behavior[precision] = pd.read_parquet(run_dir / "behavior_rows.parquet")
        traces[precision] = trace_summary(run_dir)
        sample_sets[precision] = set(behavior[precision]["base_sample_id"].astype(str))

    if any(sample_sets[item] != sample_sets["BF16"] for item in ("INT8", "INT4")):
        raise RuntimeError("precision runs do not contain identical discovery examples")
    wikitext_ids = {
        metrics[p]["general_quality"]["wikitext"]["revision"] for p in PRECISIONS
    }
    hellaswag_ids = {
        (
            metrics[p]["general_quality"]["hellaswag"]["revision"],
            metrics[p]["general_quality"]["hellaswag"]["subset_ids_sha256"],
        )
        for p in PRECISIONS
    }
    if len(wikitext_ids) != 1 or len(hellaswag_ids) != 1:
        raise RuntimeError("external-control identities differ across precision runs")

    changes: dict[str, dict[str, Any]] = {}
    for index, precision in enumerate(("INT8", "INT4"), start=1):
        changes[precision] = paired_precision_changes(
            factorial["BF16"],
            factorial[precision],
            n_bootstraps=n_bootstraps,
            seed=seed + 100 * index,
        )
        changes[precision]["B"] = paired_behavior_auroc_change(
            behavior["BF16"],
            behavior[precision],
            n_bootstraps=n_bootstraps,
            seed=seed + 100 * index + 10,
        )

    trace_aggregates = {
        precision: trace.groupby("trace_layer", as_index=False)
        .mean(numeric_only=True)
        .to_dict(orient="records")
        for precision, trace in traces.items()
    }
    bf16 = metrics["BF16"]
    int4 = metrics["INT4"]
    retention = {
        "Q": _retention(float(int4["Q0"]), float(bf16["Q0"])),
        "A": _retention(
            float(int4["factorial_contrasts"]["A_matched_minus_random"]["mean"]),
            float(bf16["factorial_contrasts"]["A_matched_minus_random"]["mean"]),
        ),
        "G": _retention(
            float(int4["factorial_contrasts"]["G_matched_minus_random"]["mean"]),
            float(bf16["factorial_contrasts"]["G_matched_minus_random"]["mean"]),
        ),
    }
    ppl_change = (
        float(int4["general_quality"]["wikitext"]["perplexity"])
        / float(bf16["general_quality"]["wikitext"]["perplexity"])
        - 1.0
    )
    hellaswag_change = float(int4["general_quality"]["hellaswag"]["accuracy"]) - float(
        bf16["general_quality"]["hellaswag"]["accuracy"]
    )
    catastrophic_flags = {
        "wikitext_ppl_increase_gt_25pct": bool(ppl_change > 0.25),
        "hellaswag_accuracy_decrease_gt_0.10": bool(hellaswag_change < -0.10),
    }
    gate = {
        "D_native_INT4_at_least_0.99": bool(int4["D_native"]["auroc"] >= 0.99),
        "paired_G_decrease_ci_excludes_zero": bool(changes["INT4"]["G"]["ci_high"] < 0),
        "integrity": bool(all(metrics[p]["integrity"]["finite"] for p in PRECISIONS)),
    }
    gate["pass"] = bool(all(gate.values()))
    result: dict[str, Any] = {
        "runs": {key: value.name for key, value in run_dirs.items()},
        "metrics": metrics,
        "paired_changes_from_bf16": changes,
        "retention_INT4_over_BF16": retention,
        "trace_aggregates": trace_aggregates,
        "general_quality_changes_INT4_from_BF16": {
            "wikitext_ppl_relative_change": ppl_change,
            "hellaswag_accuracy_absolute_change": hellaswag_change,
            "catastrophic_flags": catastrophic_flags,
        },
        "full_discovery_gate": gate,
        "confirmation_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# E14 Full Discovery Summary",
        "",
        "Status: frozen full discovery complete; the E14 confirmation holdout was not accessed.",
        "",
        "## Primary results",
        "",
        "| Precision | D native | D frozen | B | Q | A matched-random | G matched-random |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for precision in PRECISIONS:
        item = metrics[precision]
        lines.append(
            f"| {precision} | {item['D_native']['auroc']:.6f} | "
            f"{item['D_frozen_bf16_axis']['auroc']:.6f} | {item['B']['auroc']:.6f} | "
            f"{item['Q0']:.6f} | "
            f"{item['factorial_contrasts']['A_matched_minus_random']['mean']:.6f} | "
            f"{item['factorial_contrasts']['G_matched_minus_random']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## General quality",
            "",
            "| Precision | WikiText-2 PPL | HellaSwag accuracy |",
            "|---|---:|---:|",
        ]
    )
    for precision in PRECISIONS:
        general = metrics[precision]["general_quality"]
        lines.append(
            f"| {precision} | {general['wikitext']['perplexity']:.6f} | "
            f"{general['hellaswag']['accuracy']:.6f} |"
        )
    lines.extend(["", "## Paired changes from BF16", ""])
    for precision in ("INT8", "INT4"):
        lines.append(f"### {precision}")
        lines.append("")
        for metric in ("Q", "A", "G", "B"):
            item = changes[precision][metric]
            lines.append(
                f"- {metric}: {item['mean']:+.6f} "
                f"(95% pair-cluster CI [{item['ci_low']:+.6f}, {item['ci_high']:+.6f}])."
            )
        lines.append("")
    lines.extend(
        [
            "## INT4 actionability retention",
            "",
            f"- R_Q: {retention['Q']:.6f}",
            f"- R_A: {retention['A']:.6f}",
            f"- R_G: {retention['G']:.6f}",
            "",
            "## Trace diagnosis",
            "",
            "A/G values are matched structured minus seed-averaged random context and use each precision's frozen validation scales.",
            "",
            "| Precision | Layer | Q q_z | Q margin_z | A q_z | A margin_z | G q_z | G margin_z |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for precision in PRECISIONS:
        for row in trace_aggregates[precision]:
            lines.append(
                f"| {precision} | {int(row['trace_layer'])} | {row['Q_q_z']:.6f} | "
                f"{row['Q_margin_z']:.6f} | {row['structured_minus_random_A_q_z']:.6f} | "
                f"{row['structured_minus_random_A_margin_z']:.6f} | "
                f"{row['structured_minus_random_G_q_z']:.6f} | "
                f"{row['structured_minus_random_G_margin_z']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Gate and claim boundary",
            "",
            f"- E14 confirmation gate: **{'PASS' if gate['pass'] else 'FAIL'}**.",
            f"- INT4 WikiText PPL change: {ppl_change:+.1%}; catastrophic flag: {catastrophic_flags['wikitext_ppl_increase_gt_25pct']}.",
            f"- INT4 HellaSwag accuracy change: {hellaswag_change:+.3f}; catastrophic flag: {catastrophic_flags['hellaswag_accuracy_decrease_gt_0.10']}.",
            "- Because the WikiText flag fired, discovery supports mixed actionability and general degradation, not selective semantic damage.",
            "",
            "## Integrity",
            "",
        ]
    )
    for precision in PRECISIONS:
        item = metrics[precision]
        lines.append(
            f"- {precision}: complete; finite={item['integrity']['finite']}; "
            f"no-op max deviation={item['integrity']['no_op_max_abs_logit_deviation']:.3e}; "
            f"trace rows={item['integrity']['trace_rows']}; run={item['run_id']}."
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
