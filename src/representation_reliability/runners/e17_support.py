"""Frozen-teacher cache, reference evaluation, and training for E17.

The selected OLMo-2 pair cannot hold a 13.6 GiB teacher and a 1.48 B student
being trained on one 16 GiB device. Because the teacher is frozen and runs under
``inference_mode``, its last-prompt logits and layer* hidden state depend only on
the prompt and are precomputed once. Caching them changes no term of any
objective, no learning rate, no step count, and no batch size; it only removes
the teacher from the GPU while the student trains.
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

from ..metrics.causal_organization import (
    add_factorial_effect_views,
    immutable_run_identity,
    matched_profiles,
    representation_similarity,
    select_b_matched_checkpoint,
)
from ..reporting.tables import save_json, save_table
from ..runtime.checkpoint import latest_complete_checkpoint
from ..runtime.manifest import RunManifest
from ..runtime.status import StatusFile
from .e00c import candidate_token_id_lists
from .e01a_support import extract_resid_post_layers
from .e13 import (
    MAX_STEPS,
    MICROBATCH,
    SELECTOR,
    _evaluate_checkpoint,
    _reference_from_model,
    _train_regime,
)
from .e13_multiseed import (
    _primary_summary,
    _quality,
    _seed_everything,
    _validation_clean_metrics,
)
from .e17 import (
    E17_VERSION,
    RELATIVE_DEPTH,
    TRAINING_SEEDS,
    _config,
    _corpus_bundle,
    _selection,
    campaign_dir,
    relative_layer,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)


def reference_paths() -> dict[str, Path]:
    root = campaign_dir() / "reference"
    return {
        "root": root,
        "summary": root / "reference_summary.json",
        "teacher_probe": root / "teacher_reference.npz",
        "teacher_targets": root / "teacher_targets.json",
        "student_probe": root / "initial_student_reference.npz",
        "student_targets": root / "initial_student_targets.json",
        "kd_cache": root / "teacher_kd_cache.safetensors",
        "kd_index": root / "teacher_kd_index.json",
    }


def _save_reference_npz(reference: dict[str, Any], npz_path: Path, targets_path: Path) -> None:
    np.savez(npz_path, **reference["probe"], direction=reference["direction"])
    save_json(reference["targets"], targets_path)


def load_reference_npz(npz_path: Path, targets_path: Path) -> dict[str, Any]:
    payload = np.load(npz_path)
    probe = {key: payload[key] for key in payload.files if key != "direction"}
    return {
        "probe": probe,
        "direction": payload["direction"],
        "targets": json.loads(targets_path.read_text(encoding="utf-8")),
    }


def build_teacher_kd_cache(teacher, samples_by_id, train_ids, *, layer: int, microbatch: int):
    """Precompute the frozen teacher's last-prompt logits and layer* hidden state."""
    from .e13 import _last_positions

    logits_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    for start in range(0, len(train_ids), microbatch):
        chunk = train_ids[start : start + microbatch]
        encoded = teacher.tokenize([samples_by_id[sid].prompt for sid in chunk])
        input_ids = encoded["input_ids"].to(teacher.device)
        attention = encoded["attention_mask"].to(teacher.device)
        positions = _last_positions(attention)
        index = torch.arange(len(chunk), device=teacher.device)
        with torch.inference_mode():
            out = teacher.model(
                input_ids=input_ids, attention_mask=attention, output_hidden_states=True
            )
            logits_rows.append(out.logits[index, positions].to(torch.float16).cpu().numpy())
            hidden_rows.append(
                out.hidden_states[layer + 1][index, positions].to(torch.float16).cpu().numpy()
            )
    return np.concatenate(logits_rows, axis=0), np.concatenate(hidden_rows, axis=0)


def make_teacher_lookup(index_ids, logits, hidden, device, dtype):
    position = {str(sid): i for i, sid in enumerate(index_ids)}

    def lookup(batch_ids):
        rows = [position[str(sid)] for sid in batch_ids]
        return {
            "logits": torch.as_tensor(np.asarray(logits[rows])).to(device=device, dtype=dtype),
            "hidden": torch.as_tensor(np.asarray(hidden[rows])).to(device=device, dtype=dtype),
        }

    return lookup


def evaluate_model(
    adapter,
    samples_by_id,
    frame,
    labels,
    token_ids,
    reference,
    *,
    regime: str,
    step: int,
    output_dir: Path,
    batch_size: int,
    layer: int,
):
    """Validation-scaled causal evaluation of one model at its own relative site."""
    ids = frame["sample_id"].astype(str).tolist()
    activations, token_indices, token_sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=[layer],
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    validation, validation_rows = _validation_clean_metrics(
        adapter, samples_by_id, frame, labels, token_ids, token_indices, batch_size=batch_size
    )
    metrics, rows = _evaluate_checkpoint(
        adapter,
        samples_by_id,
        frame,
        labels,
        token_ids,
        reference,
        regime=regime,
        step=step,
        output_dir=output_dir,
        batch_size=batch_size,
        precomputed_activations=activations,
        precomputed_token_indices=token_indices,
        precomputed_token_sites=token_sites,
        layer=layer,
    )
    rows = add_factorial_effect_views(
        rows, sigma_margin_validation=validation["sigma_margin_validation"]
    )
    metrics["validation"] = validation
    metrics["validation_B"] = float(validation["B"]["auroc"])
    metrics["selection_split"] = "validation"
    metrics["site_layer"] = int(layer)
    metrics["causal_effects"] = _primary_summary(rows)
    save_table(rows, output_dir / "factorial_rows.parquet")
    save_table(validation_rows, output_dir / "validation_clean_rows.parquet")
    save_json(metrics, output_dir / "metrics.json")
    return metrics, rows, validation, activations, layer


def prepare_e17_reference() -> Path:
    """Evaluate the frozen teacher and R0, and build the frozen-teacher KD cache."""
    from safetensors.numpy import save_file as save_numpy_safetensors

    selected = _selection()
    paths = reference_paths()
    existing = StatusFile.load(paths["root"])
    if existing is not None and existing.is_complete():
        return paths["summary"]
    paths["root"].mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(paths["root"], "E17-reference", "E17")

    _samples, frame, corpus_stats, labels, samples_by_id = _corpus_bundle()
    save_table(frame, paths["root"] / "corpus_rows.parquet")
    save_json(corpus_stats, paths["root"] / "corpus_stats.json")
    train_ids = frame.loc[frame["split"].eq("train"), "sample_id"].astype(str).tolist()
    all_ids = frame["sample_id"].astype(str).tolist()

    teacher_cfg = _config(selected["teacher_config"])
    student_cfg = _config(selected["student_config"])
    batch_size = int(teacher_cfg.runtime.batch_size)
    summary: dict[str, Any] = {
        "version": E17_VERSION,
        "selected_pair": {
            key: selected[key] for key in ("candidate", "teacher_id", "student_id")
        },
        "relative_depth": RELATIVE_DEPTH,
        "confirmation_accessed": False,
    }
    try:
        status.update(message="evaluating frozen E17 teacher")
        _seed_everything(20261700)
        teacher = load_adapter(teacher_cfg)
        teacher.model.eval()
        teacher_layer = relative_layer(teacher.num_layers)
        token_ids = [
            int(item[0])
            for item in candidate_token_id_lists(
                teacher, list(teacher_cfg.behavior.candidates_primary)
            )
        ]
        teacher_reference = _reference_from_model(
            teacher, samples_by_id, frame, labels, token_ids, batch_size, layer=teacher_layer
        )
        _save_reference_npz(teacher_reference, paths["teacher_probe"], paths["teacher_targets"])
        teacher_metrics, teacher_rows, _tval, teacher_acts, _l = evaluate_model(
            teacher,
            samples_by_id,
            frame,
            labels,
            token_ids,
            teacher_reference,
            regime="teacher",
            step=0,
            output_dir=paths["root"] / "teacher",
            batch_size=batch_size,
            layer=teacher_layer,
        )
        save_table(matched_profiles(teacher_rows), paths["root"] / "teacher" / "profile.parquet")
        teacher_quality = _quality(
            teacher, paths["root"] / "teacher" / "quality", batch_size=batch_size
        )
        teacher_matrix = np.stack([teacher_acts[teacher_layer][sid] for sid in all_ids])

        status.update(message="building frozen-teacher KD cache")
        kd_logits, kd_hidden = build_teacher_kd_cache(
            teacher, samples_by_id, train_ids, layer=teacher_layer, microbatch=MICROBATCH
        )
        save_numpy_safetensors(
            {
                "logits": kd_logits,
                "hidden": kd_hidden,
                "teacher_activations": teacher_matrix.astype(np.float16),
            },
            str(paths["kd_cache"]),
        )
        save_json(
            {
                "train_ids": train_ids,
                "all_ids": all_ids,
                "teacher_layer": int(teacher_layer),
                "vocab": int(kd_logits.shape[1]),
                "teacher_width": int(kd_hidden.shape[1]),
                "dtype": "float16",
            },
            paths["kd_index"],
        )
        teacher_probe_ids = teacher.tokenize(
            [samples_by_id[sid].prompt for sid in all_ids[:8]]
        )["input_ids"].cpu().numpy()
        teacher_revisions = teacher.resolved_revisions()
        teacher_width = int(teacher.hidden_size)
        del teacher, teacher_acts
        gc.collect()
        torch.cuda.empty_cache()

        status.update(message="evaluating frozen E17 R0 student")
        student = load_adapter(student_cfg)
        student.model.eval()
        student_layer = relative_layer(student.num_layers)
        student_token_ids = [
            int(item[0])
            for item in candidate_token_id_lists(
                student, list(student_cfg.behavior.candidates_primary)
            )
        ]
        if student_token_ids != token_ids:
            raise RuntimeError("E17 teacher/student candidate token identity mismatch")
        student_probe_ids = student.tokenize(
            [samples_by_id[sid].prompt for sid in all_ids[:8]]
        )["input_ids"].cpu().numpy()
        if not np.array_equal(teacher_probe_ids, student_probe_ids):
            raise RuntimeError("E17 teacher/student tokenization differs; KD cache is invalid")
        student_reference = _reference_from_model(
            student,
            samples_by_id,
            frame,
            labels,
            student_token_ids,
            batch_size,
            layer=student_layer,
        )
        _save_reference_npz(student_reference, paths["student_probe"], paths["student_targets"])
        r0_metrics, r0_rows, _r0val, r0_acts, _l0 = evaluate_model(
            student,
            samples_by_id,
            frame,
            labels,
            student_token_ids,
            student_reference,
            regime="R0",
            step=0,
            output_dir=paths["root"] / "R0",
            batch_size=batch_size,
            layer=student_layer,
        )
        save_table(matched_profiles(r0_rows), paths["root"] / "R0" / "profile.parquet")
        r0_quality = _quality(student, paths["root"] / "R0" / "quality", batch_size=batch_size)
        student_revisions = student.resolved_revisions()
        student_width = int(student.hidden_size)
        del student, r0_acts
        gc.collect()
        torch.cuda.empty_cache()

        summary.update(
            {
                "teacher_layer": int(teacher_layer),
                "student_layer": int(student_layer),
                "teacher_width": teacher_width,
                "student_width": student_width,
                "teacher_revisions": teacher_revisions,
                "student_revisions": student_revisions,
                "token_ids": token_ids,
                "teacher": teacher_metrics,
                "teacher_quality": teacher_quality,
                "R0": r0_metrics,
                "R0_quality": r0_quality,
                "status": "complete",
            }
        )
        save_json(summary, paths["summary"])
        status.complete(message="E17 reference complete")
        return paths["summary"]
    except Exception as exc:
        status.fail(message=f"{type(exc).__name__}: {exc}")
        raise


def run_e17_job(regime: str, seed: int) -> Path:
    """Train and evaluate one frozen E17 regime/seed with the cached teacher."""
    from safetensors.numpy import load_file as load_numpy_safetensors

    if regime not in ("R1", "R2", "R3"):
        raise ValueError("E17 trains only R1, R2 and R3")
    if int(seed) not in TRAINING_SEEDS:
        raise ValueError("E17 seed is outside the frozen design")
    selected = _selection()
    paths = reference_paths()
    if not paths["summary"].exists():
        raise RuntimeError("E17 reference must be prepared before training")
    reference_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    identity_payload = {
        "version": E17_VERSION,
        "regime": regime,
        "seed": int(seed),
        "teacher_id": selected["teacher_id"],
        "student_id": selected["student_id"],
        "teacher_revisions": reference_summary["teacher_revisions"],
        "student_revisions": reference_summary["student_revisions"],
        "teacher_layer": reference_summary["teacher_layer"],
        "student_layer": reference_summary["student_layer"],
        "relative_depth": RELATIVE_DEPTH,
        "max_steps": MAX_STEPS,
        "confirmation_accessed": False,
    }
    identity = immutable_run_identity(identity_payload)
    run_dir = campaign_dir() / "jobs" / f"{regime}_seed_{seed}_{identity[:12]}"
    existing = StatusFile.load(run_dir)
    if existing is not None and existing.is_complete():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(run_dir, run_dir.name, "E17")
    manifest = RunManifest(run_dir)
    manifest.set_start(identity, {"identity": identity_payload}, {"training": int(seed)})

    try:
        _samples, frame, _stats, labels, samples_by_id = _corpus_bundle()
        train_ids = frame.loc[frame["split"].eq("train"), "sample_id"].astype(str).tolist()
        student_cfg = _config(selected["student_config"])
        batch_size = int(student_cfg.runtime.batch_size)
        student_layer = int(reference_summary["student_layer"])
        teacher_layer = int(reference_summary["teacher_layer"])
        token_ids = [int(v) for v in reference_summary["token_ids"]]
        student_reference = load_reference_npz(paths["student_probe"], paths["student_targets"])

        cache = load_numpy_safetensors(str(paths["kd_cache"]))
        index = json.loads(paths["kd_index"].read_text(encoding="utf-8"))
        if list(index["train_ids"]) != train_ids:
            raise RuntimeError("E17 KD cache row identity mismatch")
        teacher_matrix = np.asarray(cache["teacher_activations"], dtype=np.float32)

        _seed_everything(int(seed))
        student = load_adapter(student_cfg)
        checkpoint_root = run_dir / "checkpoints"
        resume = latest_complete_checkpoint(checkpoint_root, identity=identity)
        resume_checkpoint = resume[0] if resume is not None else None

        lookup = (
            None
            if regime == "R1"
            else make_teacher_lookup(
                index["train_ids"],
                cache["logits"],
                cache["hidden"],
                student.device,
                student.torch_dtype,
            )
        )

        def evaluate(regime_name, step, output_dir, projector=None):
            adapter = student
            adapter.model.eval()
            metrics, rows, _validation, activations, _l = evaluate_model(
                adapter,
                samples_by_id,
                frame,
                labels,
                token_ids,
                student_reference,
                regime=str(regime_name),
                step=step,
                output_dir=output_dir,
                batch_size=batch_size,
                layer=student_layer,
            )
            metrics["seed"] = int(seed)
            if projector is not None:
                diagnostics = {}
                for split in ("validation", "discovery_test"):
                    split_ids = (
                        frame.loc[frame["split"].eq(split), "sample_id"].astype(str).tolist()
                    )
                    positions = [index["all_ids"].index(sid) for sid in split_ids]
                    student_matrix = np.stack([activations[student_layer][s] for s in split_ids])
                    teacher_block = teacher_matrix[positions]
                    teacher_normalized = teacher_block / np.sqrt(
                        np.mean(teacher_block.astype(np.float64) ** 2, axis=1, keepdims=True)
                        + 1e-8
                    )
                    with torch.no_grad():
                        tensor = torch.from_numpy(student_matrix).to(
                            adapter.device, torch.float32
                        )
                        normalized = tensor / torch.sqrt(
                            torch.mean(tensor**2, dim=1, keepdim=True) + 1e-8
                        )
                        projected = projector(normalized).float().cpu().numpy()
                    diagnostics[split] = representation_similarity(
                        student_matrix,
                        teacher_normalized,
                        projected,
                        cka_teacher=teacher_block,
                    )
                metrics["representation_similarity"] = diagnostics
            save_json(metrics, output_dir / "metrics.json")
            adapter.model.train()
            return metrics, rows

        status.update(message=f"training E17 {regime} seed {seed}")
        losses, metrics = _train_regime(
            regime,
            student,
            None,
            samples_by_id,
            train_ids,
            labels,
            token_ids,
            evaluate,
            checkpoint_root,
            training_seed=int(seed),
            run_identity=identity,
            resume_checkpoint=resume_checkpoint,
            layer=student_layer,
            teacher_layer=teacher_layer,
            teacher_batch_lookup=lookup,
        )
        baseline_metrics = dict(reference_summary["R0"])
        baseline_metrics["seed"] = int(seed)
        all_metrics = [baseline_metrics, *metrics]
        save_table(pd.json_normalize(all_metrics, sep="."), run_dir / "checkpoint_metrics.parquet")
        save_table(pd.DataFrame(losses), run_dir / "training_losses.parquet")

        teacher_validation_b = float(reference_summary["teacher"]["validation_B"])
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
        selection = select_b_matched_checkpoint(selection_rows, teacher_validation_b)
        save_json(selection, run_dir / "b_matched_selection.json")

        quality: dict[str, Any] = {}
        selected_step = int(selection["selected_step"])
        for step in sorted({selected_step, MAX_STEPS}):
            if step == 0:
                quality[f"step_{step:03d}"] = reference_summary["R0_quality"]
                continue
            checkpoint = checkpoint_root / f"step_{step:03d}"
            local = student_cfg.model.model_copy(
                update={"id": str(checkpoint / "model"), "revision": None}
            )
            adapter = load_adapter(student_cfg.model_copy(update={"model": local}))
            adapter.model.eval()
            quality[f"step_{step:03d}"] = _quality(
                adapter, run_dir / "quality" / f"step_{step:03d}", batch_size=batch_size
            )
            del adapter
            gc.collect()
            torch.cuda.empty_cache()

        summary = {
            "regime": regime,
            "seed": int(seed),
            "identity": identity_payload,
            "run_identity_sha256": identity,
            "b_matched": selection,
            "checkpoints": all_metrics,
            "general_quality": quality,
            "status": "complete",
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "job_summary.json")
        status.complete(message=f"E17 {regime} seed {seed} complete")
        manifest.finish()
        return run_dir
    except Exception as exc:
        status.fail(message=f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise


__all__ = [
    "build_teacher_kd_cache",
    "evaluate_model",
    "load_reference_npz",
    "make_teacher_lookup",
    "prepare_e17_reference",
    "reference_paths",
    "run_e17_job",
]
