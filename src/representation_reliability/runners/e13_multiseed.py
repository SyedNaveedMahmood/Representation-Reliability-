"""Frozen E13 multi-seed causal-organization transfer discovery campaign."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import resolve_config
from ..metrics.causal_organization import (
    add_factorial_effect_views,
    causal_organization_distance,
    controlled_profiles,
    immutable_run_identity,
    matched_profiles,
    representation_similarity,
    select_b_matched_checkpoint,
    validation_margin_statistics,
)
from ..metrics.decoding import classification_metrics
from ..metrics.general_quality import score_hellaswag, score_wikitext
from ..probes.linear import fit_probe, transform_features
from ..reporting.tables import save_json, save_table
from ..runtime.checkpoint import latest_complete_checkpoint
from ..runtime.manifest import RunManifest, dataset_split_hash
from ..runtime.status import StatusFile
from .e00c import candidate_token_id_lists
from .e01a import _selected_margin
from .e01a_support import (
    extract_resid_post_layers,
    run_unintervened_batches,
)
from .e13 import (
    CHECKPOINT_STEPS,
    CORPUS_SPECS,
    LAYER,
    MAX_STEPS,
    SELECTOR,
    SITE,
    HiddenStateProjector,
    _evaluate_checkpoint,
    _reference_from_model,
    _train_regime,
    build_e13_open_corpus,
    rms_normalize_hidden,
)
from .extract import load_adapter
from .probe import text_baseline_metrics

logger = logging.getLogger(__name__)

PROTOCOL_SHA256 = "04daa7fcc66cc1c93f8077de23962dfec9861c9412c44367d83603ed0ccb7cac"
E13_MULTI_VERSION = "e13-multiseed-causal-transfer-v1"
TRAINING_SEEDS = (20261305, 20261315, 20261325)
REGIMES = ("R1", "R2", "R3")
CAMPAIGN_ID = f"E13MS_{PROTOCOL_SHA256[:12]}"


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def dataframe_to_markdown(frame: pd.DataFrame, *, digits: int = 6) -> str:
    """Render a compact pipe table without Pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{int(digits)}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _column in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _corpus_bundle():
    samples, frame, stats = build_e13_open_corpus(CORPUS_SPECS)
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    split_digest = dataset_split_hash(dict(zip(frame["sample_id"], frame["split"])))
    corpus_digest = _sha256_json(
        frame[["sample_id", "pair_id", "split", "prompt", "target_label"]].to_dict(
            orient="records"
        )
    )
    return samples, frame, stats, labels, samples_by_id, split_digest, corpus_digest


def _validation_clean_metrics(
    adapter,
    samples_by_id,
    frame: pd.DataFrame,
    labels: dict[str, int],
    token_ids: list[int],
    token_indices: dict[str, int],
    *,
    batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    validation_ids = frame.loc[
        frame["split"].eq("validation"), "sample_id"
    ].astype(str).tolist()
    logits = run_unintervened_batches(
        adapter,
        samples_by_id,
        validation_ids,
        token_indices=token_indices,
        output_token_ids=token_ids,
        batch_size=batch_size,
    )
    margins = np.asarray([_selected_margin(logits[sid]) for sid in validation_ids])
    targets = np.asarray([labels[sid] for sid in validation_ids], dtype=int)
    stats = validation_margin_statistics(margins, targets)
    stats["B"] = classification_metrics(targets, margins)
    rows = pd.DataFrame(
        {
            "sample_id": validation_ids,
            "pair_id": [str(samples_by_id[sid].pair_id) for sid in validation_ids],
            "target_label": targets,
            "yes_logit": [float(np.asarray(logits[sid])[0]) for sid in validation_ids],
            "no_logit": [float(np.asarray(logits[sid])[1]) for sid in validation_ids],
            "yes_no_margin": margins,
            "source_split": "validation",
            "confirmation_accessed": False,
        }
    )
    return stats, rows


def _primary_summary(rows: pd.DataFrame) -> dict[str, Any]:
    matched = rows.loc[rows["context"].eq("matched")]
    random_rows = rows.loc[rows["context"].eq("random")]
    summary: dict[str, Any] = {}
    for name in (
        "Q_raw", "A_raw", "G_raw", "Q_z", "A_z", "G_z",
        "Q_prob", "A_prob", "G_prob", "q_target_flip",
        "context_target_flip", "joint_target_flip",
    ):
        summary[name] = float(matched[name].mean())
    for name in (
        "A_raw", "G_raw", "A_z", "G_z", "A_prob", "G_prob",
        "context_target_flip", "joint_target_flip",
    ):
        summary[f"{name}_matched_minus_random"] = float(
            matched[name].mean() - random_rows[name].mean()
        )
    return summary


def _probe_controls(
    activations: dict[int, dict[str, np.ndarray]],
    frame: pd.DataFrame,
    labels: dict[str, int],
) -> dict[str, Any]:
    blocks = {}
    texts = {}
    label_blocks = {}
    for split in ("train", "validation", "discovery_test"):
        subset = frame.loc[frame["split"].eq(split)]
        ids = subset["sample_id"].astype(str).tolist()
        blocks[split] = np.stack([activations[LAYER][sid] for sid in ids])
        texts[split] = subset["prompt"].astype(str).tolist()
        label_blocks[split] = np.asarray([labels[sid] for sid in ids], dtype=int)
    rng = np.random.default_rng(20261306)
    randomized = {name: rng.permutation(values) for name, values in label_blocks.items()}
    fit = fit_probe(
        blocks["train"],
        randomized["train"],
        blocks["validation"],
        randomized["validation"],
        c_grid=(0.01, 0.1, 1.0, 10.0),
        seed=20261306,
    )
    random_scores = fit["classifier"].decision_function(
        transform_features(fit, blocks["discovery_test"])
    )
    majority_label = int(np.mean(label_blocks["train"]) >= 0.5)
    majority_scores = np.full(len(label_blocks["discovery_test"]), majority_label)
    return {
        "majority": classification_metrics(
            label_blocks["discovery_test"], majority_scores
        ),
        "text_surface": text_baseline_metrics(
            texts,
            {name: values.tolist() for name, values in label_blocks.items()},
            c_grid=(0.01, 0.1, 1.0, 10.0),
            seed=20261306,
        ),
        "random_label": classification_metrics(
            randomized["discovery_test"], random_scores
        ),
        "random_label_seed": 20261306,
        "random_label_validation_auroc": float(fit["validation_auroc_best"]),
    }


def _project_numpy(projector, values: np.ndarray, device) -> np.ndarray:
    outputs = []
    projector.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 64):
            tensor = torch.as_tensor(values[start : start + 64], device=device)
            tensor = tensor.to(next(projector.parameters()).dtype)
            outputs.append(projector(rms_normalize_hidden(tensor)).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _enhanced_evaluate(
    adapter,
    samples_by_id,
    frame,
    labels,
    token_ids,
    frozen_reference,
    teacher_primary_profile,
    teacher_controlled_profile,
    *,
    regime: str,
    seed: int | None,
    step: int,
    output_dir: Path,
    batch_size: int,
    projector=None,
    teacher_activations: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ids = frame["sample_id"].astype(str).tolist()
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
        frame,
        labels,
        token_ids,
        token_indices,
        batch_size=batch_size,
    )
    metrics, factorial = _evaluate_checkpoint(
        adapter,
        samples_by_id,
        frame,
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
    )
    factorial = add_factorial_effect_views(
        factorial,
        sigma_margin_validation=validation["sigma_margin_validation"],
    )
    save_table(factorial, output_dir / "factorial_rows.parquet")
    save_table(validation_rows, output_dir / "validation_clean_rows.parquet")
    primary = matched_profiles(factorial)
    controlled = controlled_profiles(factorial)
    metrics.update(
        {
            "seed": seed,
            "validation": validation,
            "validation_B": float(validation["B"]["auroc"]),
            "selection_split": "validation",
            "causal_effects": _primary_summary(factorial),
            "probe_controls": _probe_controls(activations, frame, labels),
            "COD": (
                0.0
                if teacher_primary_profile is None
                else causal_organization_distance(teacher_primary_profile, primary)
            ),
            "COD_controlled": (
                0.0
                if teacher_controlled_profile is None
                else causal_organization_distance(teacher_controlled_profile, controlled)
            ),
            "confirmation_accessed": False,
        }
    )
    if projector is not None:
        if teacher_activations is None:
            raise ValueError("R3 diagnostics require frozen teacher activations")
        diagnostics = {}
        for split in ("validation", "discovery_test"):
            split_ids = frame.loc[frame["split"].eq(split), "sample_id"].astype(str).tolist()
            student_matrix = np.stack([activations[LAYER][sid] for sid in split_ids])
            teacher_matrix = np.asarray(teacher_activations[split])
            if teacher_matrix.shape[0] != len(split_ids):
                raise RuntimeError("teacher/student representation rows do not align")
            projected = _project_numpy(projector, student_matrix, adapter.device)
            teacher_normalized = teacher_matrix / np.sqrt(
                np.mean(teacher_matrix.astype(np.float64) ** 2, axis=1, keepdims=True)
                + 1e-8
            )
            diagnostics[split] = representation_similarity(
                student_matrix,
                teacher_normalized,
                projected,
                cka_teacher=teacher_matrix,
            )
        metrics["representation_similarity"] = diagnostics
    save_json(metrics, output_dir / "metrics.json")
    return metrics, factorial


def _reference_paths(campaign_dir: Path) -> dict[str, Path]:
    reference = campaign_dir / "reference"
    return {
        "root": reference,
        "status": reference / "status.json",
        "student_npz": reference / "initial_student_reference.npz",
        "student_targets": reference / "initial_student_targets.json",
        "student_tokens": reference / "initial_student_token_sites.json",
        "teacher_rows": reference / "teacher" / "factorial_rows.parquet",
        "teacher_activations": reference / "teacher_activations.safetensors",
        "summary": reference / "reference_summary.json",
    }


def _save_reference(reference: dict[str, Any], paths: dict[str, Path]) -> None:
    np.savez(
        paths["student_npz"],
        **reference["probe"],
        direction=reference["direction"],
    )
    save_json(reference["targets"], paths["student_targets"])
    save_json(
        {
            "token_indices": reference["token_indices"],
            "token_sites": reference["token_sites"],
        },
        paths["student_tokens"],
    )


def _load_reference(paths: dict[str, Path]) -> dict[str, Any]:
    arrays = np.load(paths["student_npz"])
    with paths["student_targets"].open("r", encoding="utf-8") as handle:
        targets = json.load(handle)
    with paths["student_tokens"].open("r", encoding="utf-8") as handle:
        tokens = json.load(handle)
    return {
        "probe": {name: arrays[name] for name in ("coef", "intercept", "mean", "scale")},
        "direction": arrays["direction"],
        "targets": targets,
        "token_indices": {str(k): int(v) for k, v in tokens["token_indices"].items()},
        "token_sites": tokens["token_sites"],
    }


def _quality(adapter, output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Frozen E14 scorer omits one transition at each 512-token block boundary;
    # 10,021 input tokens therefore guarantee at least 10,000 scored tokens.
    wikitext = score_wikitext(adapter, token_budget=10_021)
    if int(wikitext["scored_tokens"]) < 10_000:
        raise RuntimeError("E13 WikiText control scored fewer than 10,000 tokens")
    hellaswag, rows = score_hellaswag(adapter, n_examples=500, batch_size=batch_size)
    save_table(rows, output_dir / "hellaswag_rows.parquet")
    result = {"wikitext": wikitext, "hellaswag": hellaswag}
    save_json(result, output_dir / "general_quality.json")
    return result


def prepare_e13_multiseed_reference(campaign_dir: Path | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    campaign_dir = campaign_dir or repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    paths = _reference_paths(campaign_dir)
    existing = StatusFile.load(paths["root"])
    if existing is not None and existing.is_complete():
        return paths["root"]
    paths["root"].mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(paths["root"], f"{CAMPAIGN_ID}-reference", "E13")
    cfg, _ = resolve_config(
        base_path=repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    _samples, frame, corpus_stats, labels, samples_by_id, split_hash, corpus_digest = (
        _corpus_bundle()
    )
    save_table(frame, paths["root"] / "corpus_rows.parquet")
    save_json(corpus_stats, paths["root"] / "corpus_stats.json")
    batch_size = int(cfg.runtime.batch_size)
    try:
        status.update(message="evaluating frozen teacher reference")
        teacher_cfg, _ = resolve_config(
            base_path=repo_root / "configs/base.yaml",
            model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
            experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
            overrides=(),
        )
        teacher = load_adapter(teacher_cfg)
        teacher.model.eval()
        teacher_revisions = teacher.resolved_revisions()
        token_ids = [
            int(item[0])
            for item in candidate_token_id_lists(
                teacher, list(teacher_cfg.behavior.candidates_primary)
            )
        ]
        teacher_reference = _reference_from_model(
            teacher, samples_by_id, frame, labels, token_ids, batch_size
        )
        teacher_metrics, teacher_rows = _enhanced_evaluate(
            teacher,
            samples_by_id,
            frame,
            labels,
            token_ids,
            teacher_reference,
            None,
            None,
            regime="teacher",
            seed=None,
            step=0,
            output_dir=paths["root"] / "teacher",
            batch_size=batch_size,
        )
        teacher_primary = matched_profiles(teacher_rows)
        teacher_controlled = controlled_profiles(teacher_rows)
        all_ids = frame["sample_id"].astype(str).tolist()
        teacher_acts, _indices, _sites = extract_resid_post_layers(
            teacher,
            [samples_by_id[sid] for sid in all_ids],
            layers=[LAYER],
            token_selector=SELECTOR,
            batch_size=batch_size,
        )
        from safetensors.numpy import save_file as save_numpy_safetensors

        activation_tensors = {}
        for split in ("validation", "discovery_test"):
            split_ids = frame.loc[frame["split"].eq(split), "sample_id"].astype(str).tolist()
            activation_tensors[split] = np.stack(
                [teacher_acts[LAYER][sid] for sid in split_ids]
            ).astype(np.float32)
        save_numpy_safetensors(activation_tensors, paths["teacher_activations"])
        teacher_quality = _quality(teacher, paths["root"] / "teacher" / "quality", batch_size=16)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()

        status.update(message="evaluating frozen student reference")
        _seed_everything(TRAINING_SEEDS[0])
        student = load_adapter(cfg)
        student.model.eval()
        student_revisions = student.resolved_revisions()
        student_token_ids = [
            int(item[0])
            for item in candidate_token_id_lists(
                student, list(cfg.behavior.candidates_primary)
            )
        ]
        if student_token_ids != token_ids:
            raise RuntimeError("teacher/student candidate token IDs differ")
        student_reference = _reference_from_model(
            student, samples_by_id, frame, labels, token_ids, batch_size
        )
        _save_reference(student_reference, paths)
        baseline_metrics, _baseline_rows = _enhanced_evaluate(
            student,
            samples_by_id,
            frame,
            labels,
            token_ids,
            student_reference,
            teacher_primary,
            teacher_controlled,
            regime="R0",
            seed=None,
            step=0,
            output_dir=paths["root"] / "R0",
            batch_size=batch_size,
        )
        baseline_quality = _quality(student, paths["root"] / "R0" / "quality", batch_size=16)
        summary = {
            "status": "complete",
            "protocol_sha256": PROTOCOL_SHA256,
            "version": E13_MULTI_VERSION,
            "split_hash": split_hash,
            "corpus_digest": corpus_digest,
            "token_ids": token_ids,
            "teacher_revisions": teacher_revisions,
            "student_revisions": student_revisions,
            "teacher": teacher_metrics,
            "R0": baseline_metrics,
            "teacher_quality": teacher_quality,
            "R0_quality": baseline_quality,
            "confirmation_accessed": False,
        }
        save_json(summary, paths["summary"])
        status.complete("E13 multi-seed reference complete")
        del student
        gc.collect()
        torch.cuda.empty_cache()
        return paths["root"]
    except Exception as exc:
        status.fail(f"{type(exc).__name__}: {exc}")
        raise


def run_e13_multiseed_smoke() -> Path:
    """Forty-row GPU contract for enhanced metrics, row alignment, and artifacts."""
    repo_root = Path(__file__).resolve().parents[3]
    output = repo_root / "runs" / "E13_MULTI_SEED_SMOKE"
    cfg, _ = resolve_config(
        base_path=repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    samples, frame, _stats = build_e13_open_corpus(
        (("train", 20, 20261301), ("validation", 10, 20261302), ("discovery_test", 10, 20261303))
    )
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    teacher_cfg, _ = resolve_config(
        base_path=repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
        experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    teacher = load_adapter(teacher_cfg)
    token_ids = [
        int(item[0])
        for item in candidate_token_id_lists(
            teacher, list(teacher_cfg.behavior.candidates_primary)
        )
    ]
    teacher_reference = _reference_from_model(
        teacher, samples_by_id, frame, labels, token_ids, 2
    )
    teacher_metrics, teacher_rows = _enhanced_evaluate(
        teacher,
        samples_by_id,
        frame,
        labels,
        token_ids,
        teacher_reference,
        None,
        None,
        regime="teacher",
        seed=None,
        step=0,
        output_dir=output / "teacher",
        batch_size=2,
    )
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    student = load_adapter(cfg)
    student_reference = _reference_from_model(
        student, samples_by_id, frame, labels, token_ids, 2
    )
    student_metrics, student_rows = _enhanced_evaluate(
        student,
        samples_by_id,
        frame,
        labels,
        token_ids,
        student_reference,
        matched_profiles(teacher_rows),
        controlled_profiles(teacher_rows),
        regime="R0",
        seed=None,
        step=0,
        output_dir=output / "R0",
        batch_size=2,
    )
    required = [
        output / "teacher" / "factorial_rows.parquet",
        output / "teacher" / "validation_clean_rows.parquet",
        output / "R0" / "factorial_rows.parquet",
        output / "R0" / "metrics.json",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("E13 enhanced GPU smoke omitted required evidence")
    if len(matched_profiles(student_rows)) != 10 or not np.isfinite(
        student_metrics["COD"]["COD"]
    ):
        raise RuntimeError("E13 enhanced GPU smoke failed row/COD identity")
    result = {
        "status": "complete",
        "n_total": len(frame),
        "teacher_validation_scale": teacher_metrics["validation"][
            "sigma_margin_validation"
        ],
        "student_validation_scale": student_metrics["validation"][
            "sigma_margin_validation"
        ],
        "COD": student_metrics["COD"]["COD"],
        "required_files": [str(path) for path in required],
        "confirmation_accessed": False,
    }
    save_json(result, output / "smoke_summary.json")
    return output


def _load_projector(path: Path, student_width: int, teacher_width: int, device, dtype):
    from safetensors.torch import load_file

    projector = HiddenStateProjector(student_width, teacher_width).to(device=device, dtype=dtype)
    projector.load_state_dict(load_file(path, device=str(device)))
    return projector


def _checkpoint_adapter(cfg, checkpoint: Path):
    local_model = cfg.model.model_copy(update={"id": str(checkpoint / "model"), "revision": None})
    local_cfg = cfg.model_copy(update={"model": local_model})
    return load_adapter(local_cfg)


def run_e13_multiseed_job(regime: str, seed: int, campaign_dir: Path | None = None) -> Path:
    if regime not in REGIMES or int(seed) not in TRAINING_SEEDS:
        raise ValueError("job regime/seed is outside the frozen E13 design")
    repo_root = Path(__file__).resolve().parents[3]
    campaign_dir = campaign_dir or repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    prepare_e13_multiseed_reference(campaign_dir)
    paths = _reference_paths(campaign_dir)
    with paths["summary"].open("r", encoding="utf-8") as handle:
        reference_summary = json.load(handle)
    identity_payload = {
        "version": E13_MULTI_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "regime": regime,
        "seed": int(seed),
        "corpus_digest": reference_summary["corpus_digest"],
        "teacher_revisions": reference_summary["teacher_revisions"],
        "student_revisions": reference_summary["student_revisions"],
        "checkpoint_steps": CHECKPOINT_STEPS,
        "max_steps": MAX_STEPS,
        "confirmation_accessed": False,
    }
    identity = immutable_run_identity(identity_payload)
    run_dir = campaign_dir / "jobs" / f"{regime}_seed_{int(seed)}_{identity[:12]}"
    existing = StatusFile.load(run_dir)
    if existing is not None and existing.is_complete():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(run_dir, run_dir.name, "E13")
    manifest = RunManifest(run_dir)
    manifest.set_start(identity, {"identity": identity_payload}, {"training": int(seed)})
    cfg, _ = resolve_config(
        base_path=repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    _samples, frame, _stats, labels, samples_by_id, split_hash, corpus_digest = (
        _corpus_bundle()
    )
    if corpus_digest != reference_summary["corpus_digest"]:
        raise RuntimeError("E13 corpus digest mismatch")
    token_ids = list(map(int, reference_summary["token_ids"]))
    frozen_reference = _load_reference(paths)
    teacher_rows = pd.read_parquet(paths["teacher_rows"])
    teacher_primary = matched_profiles(teacher_rows)
    teacher_controlled = controlled_profiles(teacher_rows)
    from safetensors.numpy import load_file as load_numpy_safetensors

    teacher_activations = load_numpy_safetensors(paths["teacher_activations"])
    batch_size = int(cfg.runtime.batch_size)
    _seed_everything(int(seed))
    try:
        status.update(message=f"training {regime} seed {seed}")
        checkpoint_root = run_dir / "checkpoints"
        latest = latest_complete_checkpoint(checkpoint_root, identity=identity)
        resume_checkpoint = latest[0] if latest is not None else None
        student = (
            _checkpoint_adapter(cfg, resume_checkpoint)
            if resume_checkpoint is not None
            else load_adapter(cfg)
        )
        teacher = None
        if regime in {"R2", "R3"}:
            teacher_cfg, _ = resolve_config(
                base_path=repo_root / "configs/base.yaml",
                model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
                experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
                overrides=(),
            )
            teacher = load_adapter(teacher_cfg)
            teacher.model.eval()
            for parameter in teacher.model.parameters():
                parameter.requires_grad_(False)
        projector = None
        if regime == "R3" and resume_checkpoint is not None:
            projector = _load_projector(
                resume_checkpoint / "projector.safetensors",
                student.hidden_size,
                teacher.hidden_size,
                student.device,
                student.torch_dtype,
            )
        baseline = dict(reference_summary["R0"])
        step_zero = {
            **baseline,
            "regime": regime,
            "seed": int(seed),
            "step": 0,
            "reused_from": "R0",
        }

        def evaluate(name, step, directory, projector):
            return _enhanced_evaluate(
                student,
                samples_by_id,
                frame,
                labels,
                token_ids,
                frozen_reference,
                teacher_primary,
                teacher_controlled,
                regime=name,
                seed=int(seed),
                step=step,
                output_dir=directory,
                batch_size=batch_size,
                projector=projector,
                teacher_activations=teacher_activations if projector is not None else None,
            )

        train_ids = frame.loc[frame["split"].eq("train"), "sample_id"].astype(str).tolist()
        loss_rows, trained_metrics = _train_regime(
            regime,
            student,
            teacher,
            samples_by_id,
            train_ids,
            labels,
            token_ids,
            evaluate,
            checkpoint_root,
            training_seed=int(seed),
            run_identity=identity,
            resume_checkpoint=resume_checkpoint,
            projector=projector,
        )
        all_metrics = [step_zero, *trained_metrics]
        metric_frame = pd.json_normalize(all_metrics, sep=".")
        save_table(metric_frame, run_dir / "checkpoint_metrics.parquet")
        save_table(pd.DataFrame(loss_rows), run_dir / "training_losses.parquet")
        selection_rows = pd.DataFrame(
            [
                {
                    "step": row["step"],
                    "validation_B": row["validation_B"],
                    "selection_split": "validation",
                }
                for row in all_metrics
            ]
        )
        selection = select_b_matched_checkpoint(
            selection_rows,
            float(reference_summary["teacher"]["validation_B"]),
        )
        save_json(selection, run_dir / "b_matched_selection.json")
        quality = {"step_000": reference_summary["R0_quality"]}
        quality_steps = sorted({int(selection["selected_step"]), MAX_STEPS} - {0})
        for quality_step in quality_steps:
            checkpoint = run_dir / "checkpoints" / f"step_{quality_step:03d}"
            quality_adapter = _checkpoint_adapter(cfg, checkpoint)
            quality[f"step_{quality_step:03d}"] = _quality(
                quality_adapter,
                run_dir / "quality" / f"step_{quality_step:03d}",
                batch_size=16,
            )
            del quality_adapter
            gc.collect()
            torch.cuda.empty_cache()
        result = {
            "status": "complete",
            "identity": identity_payload,
            "run_identity_sha256": identity,
            "regime": regime,
            "seed": int(seed),
            "checkpoints": all_metrics,
            "b_matched": selection,
            "general_quality": quality,
            "split_hash": split_hash,
            "confirmation_accessed": False,
        }
        save_json(result, run_dir / "job_summary.json")
        manifest.update_dataset_info(
            split_hash=split_hash,
            corpus_digest=corpus_digest,
            confirmation_accessed=False,
        )
        manifest.update_model_info(
            teacher="Qwen/Qwen3-1.7B",
            student="Qwen/Qwen3-0.6B",
            layer=LAYER,
            site=SITE,
            token_selector=SELECTOR,
            candidate_token_ids=token_ids,
        )
        manifest.finish([{"regime": regime, "seed": int(seed), "status": "complete"}])
        status.complete(f"{regime} seed {seed} complete")
        return run_dir
    except Exception as exc:
        logger.exception("E13 multi-seed job failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.manifest.setdefault("errors", []).append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        manifest.finish()
        raise


def run_e13_multiseed_campaign() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    campaign_dir = repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    start = time.monotonic()
    prepare_e13_multiseed_reference(campaign_dir)
    completed = []
    for regime in REGIMES:
        for seed in TRAINING_SEEDS:
            completed.append(str(run_e13_multiseed_job(regime, seed, campaign_dir)))
    save_json(
        {
            "status": "baseline_complete",
            "protocol_sha256": PROTOCOL_SHA256,
            "jobs": completed,
            "wall_time_s": time.monotonic() - start,
            "confirmation_accessed": False,
        },
        campaign_dir / "campaign_summary.json",
    )
    return campaign_dir


def evaluate_method_gate(
    reference_summary: dict[str, Any], job_summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen A-E method gate without discovery-dependent choices."""
    baseline = reference_summary["R0"]
    gate_a = (
        float(baseline["D_native"]["auroc"]) >= 0.98
        and float(baseline["D_frozen_initial_axis"]["auroc"]) >= 0.98
    )
    baseline_ppl = float(reference_summary["R0_quality"]["wikitext"]["perplexity"])
    baseline_hellaswag = float(
        reference_summary["R0_quality"]["hellaswag"]["accuracy"]
    )
    seed_rows = []
    for job in job_summaries:
        if job["regime"] not in {"R2", "R3"}:
            continue
        selected_step = int(job["b_matched"]["selected_step"])
        selected = next(
            row for row in job["checkpoints"] if int(row["step"]) == selected_step
        )
        cod = selected["COD"]
        b_pass = float(job["b_matched"]["absolute_B_gap"]) <= 0.03
        component_floor = max(
            float(cod[f"mean_abs_{name}_gap"]) for name in ("Q_z", "A_z", "G_z")
        )
        c_pass = float(cod["COD"]) > 1e-8 and component_floor >= 0.20
        quality = job["general_quality"][f"step_{selected_step:03d}"]
        ppl = float(quality["wikitext"]["perplexity"])
        hellaswag = float(quality["hellaswag"]["accuracy"])
        e_pass = (
            np.isfinite([ppl, hellaswag]).all()
            and ppl < 10.0 * baseline_ppl
            and hellaswag >= baseline_hellaswag - 0.20
        )
        seed_rows.append(
            {
                "regime": job["regime"],
                "seed": int(job["seed"]),
                "selected_step": selected_step,
                "absolute_validation_B_gap": float(job["b_matched"]["absolute_B_gap"]),
                "COD": float(cod["COD"]),
                "max_mean_abs_component_gap": component_floor,
                "wikitext_perplexity": ppl,
                "hellaswag_accuracy": hellaswag,
                "Gate_B": bool(b_pass),
                "Gate_C": bool(c_pass),
                "Gate_E": bool(e_pass),
                "joint_B_C_E": bool(b_pass and c_pass and e_pass),
            }
        )
    gate_frame = pd.DataFrame(seed_rows)
    counts = (
        gate_frame.groupby("regime")["joint_B_C_E"].sum().astype(int).to_dict()
        if not gate_frame.empty
        else {}
    )
    gate_b = bool(gate_frame["Gate_B"].any()) if not gate_frame.empty else False
    gate_c = bool((gate_frame["Gate_B"] & gate_frame["Gate_C"]).any()) if not gate_frame.empty else False
    gate_d = bool(any(count >= 2 for count in counts.values()))
    gate_e = bool(
        any(
            int(group["joint_B_C_E"].sum()) >= 2
            for _regime, group in gate_frame.groupby("regime")
        )
    ) if not gate_frame.empty else False
    authorized = bool(gate_a and gate_b and gate_c and gate_d and gate_e)
    return {
        "Gate_A": gate_a,
        "Gate_B": gate_b,
        "Gate_C": gate_c,
        "Gate_D": gate_d,
        "Gate_E": gate_e,
        "joint_pass_counts_by_regime": counts,
        "conversion_response_authorized": authorized,
        "seed_rows": seed_rows,
        "confirmation_accessed": False,
    }


def analyze_e13_multiseed_campaign(campaign_dir: Path | None = None) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    campaign_dir = campaign_dir or repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    paths = _reference_paths(campaign_dir)
    with paths["summary"].open("r", encoding="utf-8") as handle:
        reference = json.load(handle)
    jobs = []
    for regime in REGIMES:
        for seed in TRAINING_SEEDS:
            candidates = list((campaign_dir / "jobs").glob(f"{regime}_seed_{seed}_*/job_summary.json"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one complete summary for {regime} seed {seed}")
            with candidates[0].open("r", encoding="utf-8") as handle:
                job = json.load(handle)
            if job.get("status") != "complete" or job.get("confirmation_accessed") is not False:
                raise RuntimeError(f"invalid job completion identity for {regime} seed {seed}")
            jobs.append(job)
    gate = evaluate_method_gate(reference, jobs)
    trajectory_rows = []
    b_matched_rows = []
    for job in jobs:
        selected_step = int(job["b_matched"]["selected_step"])
        for row in job["checkpoints"]:
            effects = row["causal_effects"]
            trajectory_rows.append(
                {
                    "regime": job["regime"],
                    "seed": job["seed"],
                    "step": int(row["step"]),
                    "B": float(row["B"]["auroc"]),
                    "validation_B": float(row["validation_B"]),
                    "D_native": float(row["D_native"]["auroc"]),
                    "D_frozen": float(row["D_frozen_initial_axis"]["auroc"]),
                    "Q_z": float(effects["Q_z"]),
                    "A_z": float(effects["A_z"]),
                    "G_z": float(effects["G_z"]),
                    "Q_raw": float(effects["Q_raw"]),
                    "A_raw": float(effects["A_raw"]),
                    "G_raw": float(effects["G_raw"]),
                    "Q_prob": float(effects["Q_prob"]),
                    "A_prob": float(effects["A_prob"]),
                    "G_prob": float(effects["G_prob"]),
                    "COD": float(row["COD"]["COD"]),
                    "is_b_matched": int(row["step"]) == selected_step,
                }
            )
        selected = next(row for row in job["checkpoints"] if int(row["step"]) == selected_step)
        b_matched_rows.append(
            {
                "regime": job["regime"],
                "seed": job["seed"],
                "selected_step": selected_step,
                "B_student": float(selected["B"]["auroc"]),
                "validation_B_student": float(selected["validation_B"]),
                "validation_B_teacher": float(reference["teacher"]["validation_B"]),
                "absolute_validation_B_gap": float(job["b_matched"]["absolute_B_gap"]),
                "Q_z": float(selected["causal_effects"]["Q_z"]),
                "A_z": float(selected["causal_effects"]["A_z"]),
                "G_z": float(selected["causal_effects"]["G_z"]),
                "COD": float(selected["COD"]["COD"]),
            }
        )
    trajectory = pd.DataFrame(trajectory_rows)
    b_matched = pd.DataFrame(b_matched_rows)
    save_table(trajectory, campaign_dir / "causal_organization_trajectory.parquet")
    save_table(b_matched, campaign_dir / "b_matched_results.parquet")
    save_table(pd.DataFrame(gate["seed_rows"]), campaign_dir / "method_gate_rows.parquet")
    save_json(gate, campaign_dir / "method_gate.json")
    aggregate = (
        b_matched.groupby("regime")[["B_student", "Q_z", "A_z", "G_z", "COD"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    save_table(aggregate, campaign_dir / "b_matched_seed_aggregate.parquet")
    report = repo_root / "E13_MULTI_SEED_DISCOVERY_SUMMARY.md"
    teacher = reference["teacher"]
    baseline = reference["R0"]
    lines = [
        "# E13 Multi-Seed Causal-Organization Transfer Discovery",
        "",
        "Status date: 2026-08-28. Full open discovery only; E13 confirmation was not accessed.",
        "",
        f"Protocol SHA-256: `{PROTOCOL_SHA256}`  ",
        f"Campaign: `{campaign_dir.relative_to(repo_root)}`",
        "",
        "## Frozen references",
        "",
        "| Model | B | D native | D frozen | Qz | Az | Gz | COD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Teacher | {teacher['B']['auroc']:.6f} | {teacher['D_native']['auroc']:.6f} | "
            f"{teacher['D_frozen_initial_axis']['auroc']:.6f} | "
            f"{teacher['causal_effects']['Q_z']:.6f} | {teacher['causal_effects']['A_z']:.6f} | "
            f"{teacher['causal_effects']['G_z']:.6f} | 0.000000 |"
        ),
        (
            f"| R0 | {baseline['B']['auroc']:.6f} | {baseline['D_native']['auroc']:.6f} | "
            f"{baseline['D_frozen_initial_axis']['auroc']:.6f} | "
            f"{baseline['causal_effects']['Q_z']:.6f} | {baseline['causal_effects']['A_z']:.6f} | "
            f"{baseline['causal_effects']['G_z']:.6f} | {baseline['COD']['COD']:.6f} |"
        ),
        "",
        "## Behavior-matched discovery",
        "",
        dataframe_to_markdown(b_matched, digits=6),
        "",
        "## Frozen method gate",
        "",
        f"- Gate A: {'PASS' if gate['Gate_A'] else 'FAIL'}",
        f"- Gate B: {'PASS' if gate['Gate_B'] else 'FAIL'}",
        f"- Gate C: {'PASS' if gate['Gate_C'] else 'FAIL'}",
        f"- Gate D: {'PASS' if gate['Gate_D'] else 'FAIL'}",
        f"- Gate E: {'PASS' if gate['Gate_E'] else 'FAIL'}",
        "",
        (
            "**CONVERSION-RESPONSE METHOD AUTHORIZED**"
            if gate["conversion_response_authorized"]
            else "**STOP AFTER TASK 4: conversion-response method is not authorized.**"
        ),
        "",
        "Raw, validation-z, bounded-probability, strict-flip, per-example COD, representation-similarity, and quality evidence are retained in the campaign directory. Claims remain discovery-only and model/task/site specific.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
