"""One-shot E13 diagnostic confirmation runner.

The runner evaluates the frozen teacher, the frozen initial student (R0) and the
six frozen B-matched R2/R3 checkpoints on the untouched ``e13_confirmation_v1``
holdout, then applies exactly the preregistered hierarchical tests.

Nothing here may refit a probe, reselect a checkpoint, change a target, or
introduce a threshold.  Every locked quantity is loaded from discovery artifacts
or from the frozen registry in
:mod:`representation_reliability.runners.e13_diagnostic_confirmation_support`.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import resolve_config
from ..metrics.causal_organization import add_factorial_effect_views, matched_profiles
from ..metrics.e13_diagnostic_confirmation import (
    COMPONENTS,
    _auroc,
    behavior_noninferiority,
    classify_confirmation,
    evaluate_regime_components,
)
from ..reporting.tables import markdown_table, save_json, save_table
from ..runtime.manifest import RunManifest
from ..runtime.status import StatusFile
from .e00c import candidate_token_id_lists
from .e01a_support import extract_resid_post_layers
from .e13 import (
    CORPUS_SPECS,
    LAYER,
    SELECTOR,
    SITE,
    _evaluate_checkpoint,
    _pair_signature,
    _reference_from_model,
    build_e13_open_corpus,
)
from .e13_diagnostic_confirmation_support import (
    BASELINE_CAMPAIGN,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_REGISTRY,
    CONFIRMATION_VERSION,
    DELTA_B,
    DELTA_C,
    FAMILY_ALPHA,
    HOLDOUT_SPEC,
    TRAINING_SEEDS,
    canonical_digest,
    materialize_e13_holdout,
    open_e13_access_record,
    resolve_checkpoint,
    validate_e13_confirmation_lock,
    verify_selection_evidence,
)
from .e13_multiseed import (
    CAMPAIGN_ID,
    _checkpoint_adapter,
    _load_reference,
    _reference_paths,
    _validation_clean_metrics,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)
CAMPAIGN_DIR_NAME = "E13_DIAGNOSTIC_CONFIRMATION"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _open_corpus_signatures() -> tuple[set[tuple[str, str]], pd.DataFrame, dict[str, Any]]:
    """Rebuild the frozen open corpus and return its prompt-pair signatures."""
    samples, frame, stats = build_e13_open_corpus()
    by_id = {str(sample.sample_id): sample for sample in samples}
    signatures: set[tuple[str, str]] = set()
    for pair_id, block in frame.groupby("pair_id", sort=True):
        ids = block["sample_id"].astype(str).tolist()
        if len(ids) != 2:
            raise RuntimeError(f"open corpus pair {pair_id} is not complete")
        signatures.add(_pair_signature(by_id[ids[0]], by_id[ids[1]]))
    return signatures, frame, stats


def _student_config():
    root = _repo_root()
    cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    return cfg


def _teacher_config():
    root = _repo_root()
    cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_1.7b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    return cfg


def _profile_frame(factorial: pd.DataFrame) -> pd.DataFrame:
    """Matched-context standardized profile, one row per confirmation example."""
    profile = matched_profiles(factorial)
    required = {"base_sample_id", "pair_id", "Q_z", "A_z", "G_z"}
    missing = sorted(required - set(profile.columns))
    if missing:
        raise RuntimeError(f"confirmation profile missing columns: {missing}")
    return profile.sort_values("base_sample_id").reset_index(drop=True)


def _clean_margin_frame(factorial: pd.DataFrame, *, expected_auroc: float) -> pd.DataFrame:
    """Native Yes-minus-No clean margin plus gold label, one row per example.

    ``Y00`` is stored oriented toward the counterfactual target, so the native
    margin is recovered by undoing that orientation: the target label is
    ``1 - gold_label``, hence the stored orientation is ``+1`` when the gold
    label is ``0`` and ``-1`` when it is ``1``.  The reconstruction is verified
    against the evaluator's own B before it is used.
    """
    block = factorial[factorial["context"].astype(str) == "matched"]
    if block.empty:
        raise RuntimeError("confirmation factorial rows contain no matched context")
    frame = block.groupby(
        ["base_sample_id", "pair_id", "gold_label"], as_index=False
    )["Y00"].first()
    gold = frame["gold_label"].to_numpy(dtype=int)
    orientation = np.where(gold == 0, 1.0, -1.0)
    frame["clean_margin"] = frame["Y00"].to_numpy(dtype=np.float64) * orientation
    frame = frame.sort_values("base_sample_id").reset_index(drop=True)[
        ["base_sample_id", "pair_id", "gold_label", "clean_margin"]
    ]
    rebuilt = _auroc(
        frame["gold_label"].to_numpy(dtype=int),
        frame["clean_margin"].to_numpy(dtype=np.float64),
    )
    if not np.isclose(rebuilt, float(expected_auroc), rtol=0.0, atol=1e-9):
        raise RuntimeError(
            "reconstructed confirmation clean margins disagree with the evaluated B: "
            f"{rebuilt} vs {expected_auroc}"
        )
    return frame


def _evaluate_model(
    adapter,
    *,
    samples_by_id,
    combined_frame: pd.DataFrame,
    labels: dict[str, int],
    token_ids: list[int],
    frozen_reference: dict[str, Any],
    regime: str,
    step: int,
    output_dir: Path,
    batch_size: int,
    eval_split: str,
    confirmation_accessed: bool,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Validation-scaled confirmation evaluation for one frozen model."""
    ids = combined_frame["sample_id"].astype(str).tolist()
    activations, token_indices, token_sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=[LAYER],
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    validation, validation_rows = _validation_clean_metrics(
        adapter,
        samples_by_id,
        combined_frame,
        labels,
        token_ids,
        token_indices,
        batch_size=batch_size,
    )
    metrics, factorial = _evaluate_checkpoint(
        adapter,
        samples_by_id,
        combined_frame,
        labels,
        token_ids,
        frozen_reference,
        regime=regime,
        step=step,
        output_dir=output_dir,
        batch_size=batch_size,
        precomputed_activations=activations,
        precomputed_token_indices=token_indices,
        precomputed_token_sites=token_sites,
        eval_split=eval_split,
        confirmation_accessed=confirmation_accessed,
    )
    factorial = add_factorial_effect_views(
        factorial, sigma_margin_validation=validation["sigma_margin_validation"]
    )
    metrics["validation"] = validation
    metrics["validation_B"] = float(validation["B"]["auroc"])
    metrics["sigma_margin_validation"] = float(validation["sigma_margin_validation"])
    save_table(factorial, output_dir / "confirmation_factorial_rows.parquet")
    save_table(validation_rows, output_dir / "validation_clean_rows.parquet")
    save_json(metrics, output_dir / "metrics.json")
    return metrics, factorial, validation


def run_e13_diagnostic_confirmation(*, protocol_commit: str, dry_run: bool = False) -> Path:
    """Access the E13 holdout exactly once and apply the frozen tests.

    ``dry_run`` exercises every code path on the open discovery split instead of
    the confirmation namespace; it never opens the access ledger.
    """
    root = _repo_root()
    campaign_root = root / "runs" / CAMPAIGN_DIR_NAME
    campaign_id = f"E13DC_{CONFIRMATION_VERSION}"
    run_dir = campaign_root / ("dry_run" if dry_run else campaign_id)
    existing = StatusFile.load(run_dir)
    if existing is not None and existing.is_complete() and not dry_run:
        return run_dir / "primary_results.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(run_dir, run_dir.name, "E13")

    protocol_identity = validate_e13_confirmation_lock(root, protocol_commit)
    selection = verify_selection_evidence(root)
    save_table(selection, run_dir / "frozen_checkpoint_selection.parquet")

    signatures, open_frame, corpus_stats = _open_corpus_signatures()
    baseline_dir = root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    if BASELINE_CAMPAIGN != CAMPAIGN_ID:
        raise RuntimeError("frozen baseline campaign identity drifted")
    paths = _reference_paths(baseline_dir)
    reference_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    student_reference = _load_reference(paths)

    manifest = RunManifest(run_dir)
    manifest.set_start(
        protocol_identity["protocol_sha256"],
        {
            "identity": {
                "version": CONFIRMATION_VERSION,
                "protocol": protocol_identity,
                "holdout_spec": HOLDOUT_SPEC,
                "checkpoint_registry": CHECKPOINT_REGISTRY,
                "teacher_revisions": reference_summary["teacher_revisions"],
                "student_revisions": reference_summary["student_revisions"],
                "corpus_digest": reference_summary["corpus_digest"],
                "delta_B": DELTA_B,
                "delta_C": DELTA_C,
                "family_alpha": FAMILY_ALPHA,
                "dry_run": bool(dry_run),
            }
        },
        {},
    )

    try:
        eval_split = "discovery_test" if dry_run else "confirmation"
        if dry_run:
            eval_frame = open_frame[open_frame["split"].astype(str) == "discovery_test"].copy()
            eval_samples = None
            access_record = {
                "access_count": 0,
                "first_access_timestamp": None,
                "dry_run": True,
            }
            status.update(message="E13 confirmation dry run on discovery rows")
        else:
            access_record = open_e13_access_record(
                campaign_root, campaign_id=campaign_id, protocol_identity=protocol_identity
            )
            eval_samples, eval_frame = materialize_e13_holdout(signatures)
            status.update(message="E13 confirmation holdout materialized")

        combined = (
            open_frame.copy()
            if dry_run
            else pd.concat([open_frame, eval_frame], ignore_index=True)
        )
        if combined["sample_id"].duplicated().any():
            raise RuntimeError("confirmation and open corpora share sample identities")
        if combined["prompt"].duplicated().any():
            raise RuntimeError("confirmation and open corpora share prompts")

        open_samples, _frame, _stats = build_e13_open_corpus()
        samples_by_id = {str(sample.sample_id): sample for sample in open_samples}
        if eval_samples is not None:
            for sample in eval_samples:
                samples_by_id[str(sample.sample_id)] = sample
        labels = dict(
            zip(
                combined["sample_id"].astype(str),
                combined["target_label"].astype(int),
                strict=True,
            )
        )

        holdout_identity = {
            "n_rows": len(eval_frame),
            "n_pairs": int(eval_frame["pair_id"].nunique()),
            "split_digest": canonical_digest(
                sorted(eval_frame["sample_id"].astype(str).tolist())
            ),
            "prompt_digest": canonical_digest(
                sorted(eval_frame["prompt"].astype(str).tolist())
            ),
            "collisions_before_quota": int(eval_frame.attrs.get("collisions_before_quota", -1)),
        }
        save_json(
            {
                "holdout": holdout_identity,
                "access": access_record,
                "open_corpus_stats": corpus_stats,
            },
            run_dir / "holdout_identity.json",
        )

        cfg = _student_config()
        batch_size = int(cfg.runtime.batch_size)
        margins: dict[str, pd.DataFrame] = {}
        profiles: dict[str, pd.DataFrame] = {}
        metrics_by_model: dict[str, dict[str, Any]] = {}

        status.update(message="evaluating frozen teacher on confirmation")
        teacher_cfg = _teacher_config()
        teacher = load_adapter(teacher_cfg)
        teacher.model.eval()
        token_ids = [
            int(item[0])
            for item in candidate_token_id_lists(
                teacher, list(teacher_cfg.behavior.candidates_primary)
            )
        ]
        if [int(v) for v in reference_summary["token_ids"]] != token_ids:
            raise RuntimeError("confirmation candidate token identity drifted")
        teacher_reference = _reference_from_model(
            teacher, samples_by_id, open_frame, labels, token_ids, batch_size
        )
        teacher_metrics, teacher_factorial, _ = _evaluate_model(
            teacher,
            samples_by_id=samples_by_id,
            combined_frame=combined,
            labels=labels,
            token_ids=token_ids,
            frozen_reference=teacher_reference,
            regime="teacher",
            step=0,
            output_dir=run_dir / "teacher",
            batch_size=batch_size,
            eval_split=eval_split,
            confirmation_accessed=not dry_run,
        )
        margins["teacher"] = _clean_margin_frame(
            teacher_factorial, expected_auroc=float(teacher_metrics["B"]["auroc"])
        )
        profiles["teacher"] = _profile_frame(teacher_factorial)
        metrics_by_model["teacher"] = teacher_metrics
        del teacher
        gc.collect()
        torch.cuda.empty_cache()

        status.update(message="evaluating frozen R0 on confirmation")
        r0 = load_adapter(cfg)
        r0.model.eval()
        r0_metrics, r0_factorial, _ = _evaluate_model(
            r0,
            samples_by_id=samples_by_id,
            combined_frame=combined,
            labels=labels,
            token_ids=token_ids,
            frozen_reference=student_reference,
            regime="R0",
            step=0,
            output_dir=run_dir / "R0",
            batch_size=batch_size,
            eval_split=eval_split,
            confirmation_accessed=not dry_run,
        )
        margins["R0"] = _clean_margin_frame(
            r0_factorial, expected_auroc=float(r0_metrics["B"]["auroc"])
        )
        profiles["R0"] = _profile_frame(r0_factorial)
        metrics_by_model["R0"] = r0_metrics
        del r0
        gc.collect()
        torch.cuda.empty_cache()

        for key in sorted(CHECKPOINT_REGISTRY):
            entry = CHECKPOINT_REGISTRY[key]
            status.update(message=f"evaluating frozen {key} on confirmation")
            checkpoint = resolve_checkpoint(root, key)
            adapter = _checkpoint_adapter(cfg, checkpoint)
            adapter.model.eval()
            metrics, factorial, _ = _evaluate_model(
                adapter,
                samples_by_id=samples_by_id,
                combined_frame=combined,
                labels=labels,
                token_ids=token_ids,
                frozen_reference=student_reference,
                regime=str(entry["regime"]),
                step=int(entry["step"]),
                output_dir=run_dir / key,
                batch_size=batch_size,
                eval_split=eval_split,
                confirmation_accessed=not dry_run,
            )
            margins[key] = _clean_margin_frame(
                factorial, expected_auroc=float(metrics["B"]["auroc"])
            )
            profiles[key] = _profile_frame(factorial)
            metrics_by_model[key] = metrics
            del adapter
            gc.collect()
            torch.cuda.empty_cache()

        save_json(metrics_by_model, run_dir / "per_model_metrics.json")

        behavior: dict[str, dict[str, Any]] = {}
        components: dict[str, pd.DataFrame] = {}
        for regime in ("R2", "R3"):
            per_seed_margin = {
                int(seed): margins[f"{regime}_seed_{seed}"] for seed in TRAINING_SEEDS
            }
            behavior[regime] = behavior_noninferiority(
                per_seed_margin,
                teacher=margins["teacher"],
                delta_b=DELTA_B,
                n_draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED,
            )
            teacher_profile = profiles["teacher"]
            gaps: dict[str, dict[int, pd.DataFrame]] = {}
            for component in COMPONENTS:
                column = f"{component}_z"
                by_seed: dict[int, pd.DataFrame] = {}
                for seed in TRAINING_SEEDS:
                    student_profile = profiles[f"{regime}_seed_{seed}"]
                    if (
                        student_profile["base_sample_id"].tolist()
                        != teacher_profile["base_sample_id"].tolist()
                    ):
                        raise RuntimeError("teacher/student confirmation rows are not aligned")
                    frame = pd.DataFrame(
                        {
                            "base_sample_id": student_profile["base_sample_id"],
                            "pair_id": student_profile["pair_id"].astype(str),
                            "gap": student_profile[column].to_numpy(float)
                            - teacher_profile[column].to_numpy(float),
                        }
                    )
                    by_seed[int(seed)] = frame
                gaps[component] = by_seed
            components[regime] = evaluate_regime_components(
                gaps,
                delta_c=DELTA_C,
                n_draws=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED + 13,
                alpha=FAMILY_ALPHA,
            )
            components[regime].insert(0, "regime", regime)

        primary_rows = []
        for regime in ("R2", "R3"):
            result = behavior[regime]
            primary_rows.append(
                {
                    "hypothesis": "H13-C1" if regime == "R2" else "H13-C2",
                    "regime": regime,
                    "stage": "A_behavioral_noninferiority",
                    "estimate": float(result["aggregate_delta_B"]),
                    "ci_low": float(result["aggregate_ci_low"]),
                    "ci_high": float(result["aggregate_ci_high"]),
                    "threshold": -DELTA_B,
                    "seeds_supporting": int(result["seeds_noninferior"]),
                    "verdict": str(result["verdict"]),
                }
            )
        classification = classify_confirmation(behavior, components)
        for regime in ("R2", "R3"):
            verdict = classification["regimes"][regime]
            frame = components[regime]
            best = frame.loc[frame["mismatch"]] if frame["mismatch"].any() else frame
            leading = best.iloc[int(np.argmax(np.abs(best["aggregate_gap"].to_numpy(float))))]
            primary_rows.append(
                {
                    "hypothesis": "H13-C3" if regime == "R2" else "H13-C4",
                    "regime": regime,
                    "stage": "B_causal_organization_mismatch",
                    "estimate": float(leading["aggregate_gap"]),
                    "ci_low": float(leading["ci_low"]),
                    "ci_high": float(leading["ci_high"]),
                    "threshold": DELTA_C,
                    "seeds_supporting": int(leading["seeds_beyond_sesoi_same_direction"]),
                    "verdict": str(verdict["causal_mismatch"]),
                }
            )
        primary = pd.DataFrame(primary_rows)
        component_table = pd.concat(
            [components["R2"], components["R3"]], ignore_index=True
        ).drop(columns=["per_seed"])
        save_table(primary, run_dir / "primary_results.parquet")
        save_table(component_table, run_dir / "component_results.parquet")
        save_json(
            {
                "behavior": behavior,
                "classification": classification,
                "holdout": holdout_identity,
                "access": access_record,
                "delta_B": DELTA_B,
                "delta_C": DELTA_C,
                "family_alpha": FAMILY_ALPHA,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "confirmation_accessed": not dry_run,
            },
            run_dir / "confirmation_verdict.json",
        )
        _write_summary(
            root,
            run_dir=run_dir,
            primary=primary,
            component_table=component_table,
            behavior=behavior,
            classification=classification,
            holdout=holdout_identity,
            access=access_record,
            metrics_by_model=metrics_by_model,
            protocol_identity=protocol_identity,
            dry_run=dry_run,
        )
        status.complete(message="E13 diagnostic confirmation complete")
        manifest.finish()
        return run_dir / "primary_results.parquet"
    except Exception as exc:
        status.fail(message=f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise


def _write_summary(
    root: Path,
    *,
    run_dir: Path,
    primary: pd.DataFrame,
    component_table: pd.DataFrame,
    behavior: dict[str, Any],
    classification: dict[str, Any],
    holdout: dict[str, Any],
    access: dict[str, Any],
    metrics_by_model: dict[str, Any],
    protocol_identity: dict[str, str],
    dry_run: bool,
) -> Path:
    target = run_dir / "summary.md" if dry_run else root / "E13_DIAGNOSTIC_CONFIRMATION_SUMMARY.md"
    status_line = "DRY RUN on discovery rows" if dry_run else "one-shot confirmation complete"
    seed_rows = []
    for regime in ("R2", "R3"):
        for seed, item in behavior[regime]["per_seed"].items():
            seed_rows.append(
                {
                    "regime": regime,
                    "seed": int(seed),
                    "student_B": float(item["student_B"]),
                    "teacher_B": float(behavior[regime]["teacher_B"]),
                    "delta_B": float(item["delta_B"]),
                    "ci_low": float(item["ci_low"]),
                    "ci_high": float(item["ci_high"]),
                    "noninferior": bool(item["noninferior"]),
                }
            )
    lines = [
        "# E13 Diagnostic Confirmation",
        "",
        f"Status: {status_line}.",
        (
            f"Protocol commit `{protocol_identity['protocol_commit']}`, "
            f"SHA-256 `{protocol_identity['protocol_sha256']}`."
        ),
        (
            f"Confirmation access count: `{access.get('access_count')}`; "
            f"first access `{access.get('first_access_timestamp')}`."
        ),
        (
            f"Holdout rows `{holdout['n_rows']}`, pairs `{holdout['n_pairs']}`, "
            f"split digest `{holdout['split_digest']}`."
        ),
        "",
        "## Primary hierarchical results",
        "",
        markdown_table(primary, float_fmt="{:.6f}"),
        "",
        "## Per-seed behavioral non-inferiority",
        "",
        markdown_table(pd.DataFrame(seed_rows), float_fmt="{:.6f}"),
        "",
        "## Component mismatch (Holm within regime)",
        "",
        markdown_table(component_table, float_fmt="{:.6f}"),
        "",
        f"Classification: **{classification['classification']}**.",
        f"Cross-family entry gate: `{classification['entry_gate_for_cross_family']}`.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(metrics_by_model, run_dir / "per_model_metrics.json")
    return target


__all__ = ["CORPUS_SPECS", "LAYER", "SELECTOR", "SITE", "run_e13_diagnostic_confirmation"]
