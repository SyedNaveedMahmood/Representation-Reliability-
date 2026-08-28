"""One-shot locked E14 quantization confirmation campaign."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import resolve_config
from ..metrics.e14_confirmation import (
    classify_e14_confirmation,
    evaluate_e14_hypotheses,
)
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import dataset_split_hash, environment_manifest, project_git_state
from ..runtime.status import StatusFile, atomic_write_json
from .e14 import run_e14
from .e14_confirmation_support import (
    CONFIRMATION_VERSION,
    CONTEXT_STRENGTHS,
    FULL_RUNS,
    RANDOM_SEEDS,
    TRACE_LAYERS,
    materialize_e14_holdout,
    open_e14_access_record,
    validate_e14_confirmation_lock,
    validate_frozen_references,
)

logger = logging.getLogger(__name__)


def _write_summary(
    campaign_dir: Path,
    primary: pd.DataFrame,
    classification: str,
    metrics: dict[str, dict[str, Any]],
    generic_classification: str,
) -> None:
    lines = [
        "# E14 Quantization Confirmation",
        "",
        f"Classification: **{classification} confirmation**",
        f"Damage boundary: **{generic_classification}**",
        "",
        "The E14-specific holdout was accessed once under the remotely pushed locked protocol.",
        "",
        "| Hypothesis | Estimate | 95% CI | Raw p | Holm p | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.hypothesis} | {row.estimate:.6g} | "
            f"[{row.ci_low:.6g}, {row.ci_high:.6g}] | {row.raw_p:.6g} | "
            f"{row.holm_p:.6g} | {row.verdict} |"
        )
    lines.extend(
        [
            "",
            "## Precision results",
            "",
            "| Precision | D native | D frozen | B | Q | A matched-random | G matched-random | WikiText PPL | HellaSwag |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for precision in ("bf16", "int8", "int4"):
        item = metrics[precision]
        lines.append(
            f"| {precision.upper()} | {item['D_native']['auroc']:.6f} | "
            f"{item['D_frozen_bf16_axis']['auroc']:.6f} | {item['B']['auroc']:.6f} | "
            f"{item['Q0']:.6f} | "
            f"{item['factorial_contrasts']['A_matched_minus_random']['mean']:.6f} | "
            f"{item['factorial_contrasts']['G_matched_minus_random']['mean']:.6f} | "
            f"{item['general_quality']['wikitext']['perplexity']:.6f} | "
            f"{item['general_quality']['hellaswag']['accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "Primary inference is restricted to H14.1-H14.3. Lambda 0.5, INT8, Q, frozen-axis D, behavior, relation families, and traces are secondary.",
            "",
        ]
    )
    (campaign_dir / "E14_CONFIRMATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_e14_confirmation(*, protocol_commit: str) -> Path:
    """Execute BF16/INT8/INT4 on the E14 holdout as one locked campaign."""
    repo_root = Path(__file__).resolve().parents[3]
    protocol = validate_e14_confirmation_lock(repo_root, protocol_commit)
    sources = validate_frozen_references(repo_root)
    campaign_id = f"E14_CONFIRMATION_{protocol['protocol_sha256'][:12]}"
    root = repo_root / "runs" / "E14_CONFIRMATION"
    campaign_dir = root / campaign_id
    existing = StatusFile.load(campaign_dir)
    if existing is not None and existing.is_complete():
        raise RuntimeError("the single authorized E14 confirmation is already complete")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(campaign_dir, campaign_id, "E14_CONFIRMATION")
    access = open_e14_access_record(root, campaign_id=campaign_id, protocol_identity=protocol)

    # First holdout materialization occurs only after the immutable access ledger exists.
    samples, frame = materialize_e14_holdout()
    split_hash = dataset_split_hash(
        {sample_id: "confirmation" for sample_id in frame["sample_id"].astype(str)}
    )
    manifest: dict[str, Any] = {
        "version": CONFIRMATION_VERSION,
        "campaign_id": campaign_id,
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "project_git_sha": project_git_state().get("sha"),
        "environment": environment_manifest(),
        "protocol": protocol,
        "access_record": access,
        "full_discovery_runs": FULL_RUNS,
        "dataset": {
            "namespace": "e14_confirmation_v1",
            "split_hash": split_hash,
            "n_examples": len(frame),
            "n_pairs": int(frame["pair_id"].nunique()),
        },
        "model": "Qwen/Qwen3-1.7B",
        "precisions": ["bf16", "int8", "int4"],
        "lambdas": list(CONTEXT_STRENGTHS),
        "random_seeds": list(RANDOM_SEEDS),
        "trace_layers": list(TRACE_LAYERS),
        "candidate_token_ids": [7414, 2308],
    }
    atomic_write_json(campaign_dir / "manifest.json", manifest)
    try:
        behavior: dict[str, pd.DataFrame] = {}
        factorial: dict[str, pd.DataFrame] = {}
        traces: list[pd.DataFrame] = []
        metrics: dict[str, dict[str, Any]] = {}
        precision_runs: dict[str, str] = {}
        for index, precision in enumerate(("bf16", "int8", "int4"), start=1):
            status.update(
                message=f"running locked E14 confirmation {precision}",
                progress={"precision_index": index, "precisions_total": 3},
            )
            cfg, _ = resolve_config(
                base_path=repo_root / "configs/base.yaml",
                model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
                experiment_path=repo_root / "configs/experiments/E14_quantization_reliability.yaml",
                overrides=(),
            )
            if cfg.experiment.mode != "discovery":
                raise RuntimeError("E14 development configuration changed")
            run_dir = run_e14(
                repo_root / "configs/base.yaml",
                repo_root / "configs/models/qwen3_1.7b.yaml",
                repo_root / "configs/experiments/E14_quantization_reliability.yaml",
                (),
                precision=precision,
                profile="full",
                layer=17,
                trace_layers="17,20,23,27",
                _evaluation_samples=samples,
                _evaluation_frame=frame,
                _frozen_bf16_source=sources["bf16"],
                _frozen_native_source=sources[precision],
                _output_experiment="E14_CONFIRMATION",
                _confirmation_accessed=True,
                _protocol_identity=protocol,
            )
            precision_runs[precision] = run_dir.name
            behavior[precision] = pd.read_parquet(run_dir / "behavior_rows.parquet")
            block = pd.read_parquet(run_dir / "factorial_rows.parquet")
            factorial[precision] = block[
                np.isclose(block["lambda_context"].to_numpy(float), 1.0)
            ].copy()
            traces.append(pd.read_parquet(run_dir / "trace_rows.parquet"))
            metrics[precision] = json.loads(
                (run_dir / "precision_metrics.json").read_text(encoding="utf-8")
            )
            if metrics[precision]["confirmation_accessed"] is not True:
                raise RuntimeError("E14 confirmation evidence provenance is false")
            if metrics[precision]["integrity"]["native_probe_max_abs_deviation"] > 1e-10:
                raise RuntimeError("frozen native probe did not reproduce")

        scalar_all = pd.concat(behavior.values(), ignore_index=True)
        factorial_all = pd.concat(
            [
                pd.read_parquet(
                    root / precision_runs[precision] / "factorial_rows.parquet"
                )
                for precision in ("bf16", "int8", "int4")
            ],
            ignore_index=True,
        )
        trace_all = pd.concat(traces, ignore_index=True)
        save_table(scalar_all, campaign_dir / "scalar_rows.parquet")
        save_table(factorial_all, campaign_dir / "factorial_rows.parquet")
        save_table(trace_all, campaign_dir / "trace_rows.parquet")
        primary, details = evaluate_e14_hypotheses(behavior, factorial)
        classification = classify_e14_confirmation(primary)
        save_table(primary, campaign_dir / "primary_hypotheses.parquet")
        for name, detail in details.items():
            save_table(detail, campaign_dir / f"{name.replace('.', '_')}_paired_rows.parquet")
        secondary = (
            factorial_all.groupby(
                ["precision", "context", "lambda_context"], as_index=False
            )
            .agg(mean_Q=("Q0", "mean"), mean_A=("A", "mean"), mean_G=("G", "mean"))
        )
        save_table(secondary, campaign_dir / "secondary_metrics.parquet")
        bf16_general = metrics["bf16"]["general_quality"]
        int4_general = metrics["int4"]["general_quality"]
        ppl_change = (
            int4_general["wikitext"]["perplexity"]
            / bf16_general["wikitext"]["perplexity"]
            - 1.0
        )
        hella_change = (
            int4_general["hellaswag"]["accuracy"]
            - bf16_general["hellaswag"]["accuracy"]
        )
        catastrophic = bool(ppl_change > 0.25 or hella_change < -0.10)
        generic_classification = (
            "mixed semantic + general degradation" if catastrophic else "selective/higher-order fragility"
        )
        save_json(
            {
                "classification": classification,
                "generic_damage_classification": generic_classification,
                "wikitext_ppl_relative_change": ppl_change,
                "hellaswag_accuracy_absolute_change": hella_change,
                "precision_runs": precision_runs,
            },
            campaign_dir / "confirmation_metrics.json",
        )
        _write_summary(campaign_dir, primary, classification, metrics, generic_classification)
        manifest.update(
            {
                "finished_at": datetime.now(UTC).isoformat(),
                "precision_runs": precision_runs,
                "classification": classification,
                "generic_damage_classification": generic_classification,
            }
        )
        atomic_write_json(campaign_dir / "manifest.json", manifest)
        status.complete(f"locked E14 confirmation complete: {classification}")
        return campaign_dir
    except Exception as exc:
        logger.exception("locked E14 confirmation failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        raise
