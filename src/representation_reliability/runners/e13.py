"""E13 one-seed bounded distillation-reliability diagnostic."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.base import samples_to_dataframe
from ..data.synthetic import RELATION_FAMILIES, generate_synthetic_relations
from ..interventions.orthogonal_context import (
    orthogonal_component,
    standardize_orthogonal_context,
)
from ..interventions.setpoint import source_free_setpoint_delta
from ..interventions.truth_coordinate import coordinate_value, random_unit_direction
from ..metrics.decoding import classification_metrics
from ..metrics.quantization import factorial_components, summarize_factorial
from ..metrics.setpoint import validation_setpoint_targets
from ..probes.linear import raw_probe_direction, transform_features
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .e00c import candidate_token_id_lists
from .e01a import _selected_margin
from .e01a_support import (
    extract_resid_post_layers,
    intervention_base_activations,
    run_intervention_batches,
    run_unintervened_batches,
)
from .e14 import _fit_layer_probe, _frozen_scores, _unit
from .extract import load_adapter

logger = logging.getLogger(__name__)

E13_VERSION = "e13-bounded-distillation-v1"
LAYER = 17
SITE = "resid_post"
SELECTOR = "last_prompt"
FAMILIES = tuple(RELATION_FAMILIES)
CORPUS_SPECS = (
    ("train", 4000, 20261301),
    ("validation", 500, 20261302),
    ("discovery_test", 300, 20261303),
)
TRAINING_SEED = 20261305
CHECKPOINT_STEPS = (0, 10, 25, 50, 100)
RANDOM_CONTEXT_SEEDS = (2130, 2131, 2132)
BOOTSTRAP_DRAWS = 500
MAX_STEPS = 100
MICROBATCH = 2
GRAD_ACCUMULATION = 4
PEAK_LR = 2e-5
KD_TEMPERATURE = 2.0


def _pair_signature(first, second) -> tuple[str, str]:
    return str(first.prompt), str(second.prompt)


def _rename_pair(first, second, split_name: str, pair_index: int):
    namespace = f"e13-{split_name}-v1"
    pair_id = f"{namespace}-pair-{pair_index:06d}"
    first_id = f"{namespace}-sample-{2 * pair_index:06d}"
    second_id = f"{namespace}-sample-{2 * pair_index + 1:06d}"
    first_meta = {**dict(first.metadata), "pair_id": pair_id}
    second_meta = {**dict(second.metadata), "pair_id": pair_id}
    return (
        replace(
            first,
            sample_id=first_id,
            pair_id=pair_id,
            counterfactual_id=second_id,
            metadata=first_meta,
        ),
        replace(
            second,
            sample_id=second_id,
            pair_id=pair_id,
            counterfactual_id=first_id,
            metadata=second_meta,
        ),
    )


def build_e13_open_corpus(
    specs: tuple[tuple[str, int, int], ...] = CORPUS_SPECS,
) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    """Generate pair-complete, globally prompt-deduplicated open E13 splits."""
    seen: set[tuple[str, str]] = set()
    selected: list[Any] = []
    statistics: dict[str, Any] = {}
    for split_name, directed_count, seed in specs:
        if directed_count % 2:
            raise ValueError("E13 split sizes must contain whole pairs")
        needed_pairs = directed_count // 2
        candidate_samples = max(directed_count * 4, directed_count + 200)
        candidates = generate_synthetic_relations(
            candidate_samples,
            seed,
            n_entities=42,
            families=FAMILIES,
        )
        kept = 0
        collisions = 0
        split_rows: list[Any] = []
        for offset in range(0, len(candidates), 2):
            first, second = candidates[offset : offset + 2]
            signature = _pair_signature(first, second)
            if signature in seen:
                collisions += 1
                continue
            seen.add(signature)
            renamed = _rename_pair(first, second, split_name, kept)
            split_rows.extend(renamed)
            kept += 1
            if kept == needed_pairs:
                break
        if kept != needed_pairs:
            raise RuntimeError(f"nonduplicative E13 quota unavailable for {split_name}")
        selected.extend(split_rows)
        statistics[split_name] = {
            "seed": seed,
            "candidate_directed": candidate_samples,
            "selected_directed": len(split_rows),
            "selected_pairs": kept,
            "collisions_before_quota": collisions,
        }
    frame = samples_to_dataframe(selected)
    split_by_prefix = {
        f"e13-{split_name}-v1": split_name for split_name, _count, _seed in specs
    }
    frame["split"] = [
        next(value for prefix, value in split_by_prefix.items() if str(sid).startswith(prefix))
        for sid in frame["sample_id"]
    ]
    if frame["sample_id"].duplicated().any() or frame["prompt"].duplicated().any():
        raise RuntimeError("E13 open corpus contains duplicate identities or prompts")
    pair_sizes = frame.groupby("pair_id")["sample_id"].size()
    if not bool(pair_sizes.eq(2).all()):
        raise RuntimeError("E13 corpus split a counterfactual pair")
    statistics["confirmation_accessed"] = False
    return selected, frame, statistics


def distillation_loss(
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    teacher_logits: torch.Tensor | None,
    regime: str,
    temperature: float = KD_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Frozen R1/R2 next-token objectives over the full vocabulary."""
    ce = F.cross_entropy(student_logits.float(), target_ids)
    if regime == "R1":
        return ce, {"ce": float(ce.detach()), "kd": 0.0}
    if regime != "R2" or teacher_logits is None:
        raise ValueError("R2 requires frozen teacher logits")
    temp = float(temperature)
    kd = F.kl_div(
        F.log_softmax(student_logits.float() / temp, dim=-1),
        F.softmax(teacher_logits.float() / temp, dim=-1),
        reduction="batchmean",
    )
    loss = 0.5 * ce + 0.5 * temp * temp * kd
    return loss, {"ce": float(ce.detach()), "kd": float(kd.detach())}


def learning_rate_for_step(step: int) -> float:
    if step <= 0 or step > MAX_STEPS:
        raise ValueError("E13 optimizer step is outside the frozen schedule")
    if step <= 10:
        scale = step / 10.0
    else:
        scale = 0.5 * (1.0 + math.cos(math.pi * (step - 10) / 90.0))
    return PEAK_LR * scale


def teacher_gap_closure(value: float, baseline: float, teacher: float) -> float | None:
    denominator = float(teacher) - float(baseline)
    if abs(denominator) < 1e-8:
        return None
    return float((float(value) - float(baseline)) / denominator)


def _last_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return torch.where(attention_mask.bool(), positions[None, :], -1).max(dim=1).values


def _reference_from_model(adapter, samples_by_id, frame, labels, token_ids, batch_size):
    ids = frame["sample_id"].astype(str).tolist()
    activations, token_indices, token_sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=[LAYER],
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    fit, _ = _fit_layer_probe(
        activations,
        frame,
        labels,
        LAYER,
        c_grid=(0.01, 0.1, 1.0, 10.0),
        seed=20260829,
    )
    direction = _unit(raw_probe_direction(fit))
    validation_ids = frame.loc[frame["split"] == "validation", "sample_id"].astype(str).tolist()
    clean = run_unintervened_batches(
        adapter,
        samples_by_id,
        validation_ids,
        token_indices=token_indices,
        output_token_ids=token_ids,
        batch_size=batch_size,
    )
    validation_margins = np.asarray([_selected_margin(clean[sid]) for sid in validation_ids])
    validation_vectors = np.stack([activations[LAYER][sid] for sid in validation_ids])
    targets = validation_setpoint_targets(
        validation_vectors @ direction,
        np.asarray([labels[sid] for sid in validation_ids], dtype=int),
        validation_margins,
    )
    return {
        "probe": {
            "coef": np.asarray(fit["classifier"].coef_[0], dtype=np.float64),
            "intercept": np.asarray(fit["classifier"].intercept_, dtype=np.float64),
            "mean": np.asarray(fit["scaler_mean"], dtype=np.float64),
            "scale": np.asarray(fit["scaler_scale"], dtype=np.float64),
        },
        "direction": direction,
        "targets": targets,
        "token_indices": token_indices,
        "token_sites": token_sites,
    }


def _evaluate_checkpoint(
    adapter,
    samples_by_id,
    frame,
    labels,
    token_ids,
    frozen_reference,
    *,
    regime: str,
    step: int,
    output_dir: Path,
    batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ids = frame["sample_id"].astype(str).tolist()
    activations, token_indices, _token_sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=[LAYER],
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    fit, _ = _fit_layer_probe(
        activations,
        frame,
        labels,
        LAYER,
        c_grid=(0.01, 0.1, 1.0, 10.0),
        seed=20260829,
    )
    eval_ids = frame.loc[frame["split"] == "discovery_test", "sample_id"].astype(str).tolist()
    eval_vectors = np.stack([activations[LAYER][sid] for sid in eval_ids])
    y_eval = np.asarray([labels[sid] for sid in eval_ids], dtype=int)
    native_scores = fit["classifier"].decision_function(transform_features(fit, eval_vectors))
    frozen_scores = _frozen_scores(frozen_reference["probe"], eval_vectors)
    hidden_dim = int(adapter.hidden_size)
    zeros = {sid: np.zeros(hidden_dim, dtype=np.float64) for sid in eval_ids}
    clean = run_intervention_batches(
        adapter,
        samples_by_id,
        eval_ids,
        layer=LAYER,
        token_indices=token_indices,
        deltas_by_id=zeros,
        output_token_ids=token_ids,
        capture_layers=[LAYER],
        batch_size=batch_size,
    )
    unhooked = run_unintervened_batches(
        adapter,
        samples_by_id,
        eval_ids,
        token_indices=token_indices,
        output_token_ids=token_ids,
        batch_size=batch_size,
    )
    no_op = max(
        float(np.max(np.abs(clean[sid]["selected_logits"] - unhooked[sid])))
        for sid in eval_ids
    )
    bases = intervention_base_activations(clean, eval_ids, layer=LAYER)
    direction = frozen_reference["direction"]
    targets = frozen_reference["targets"]
    q_targets = {
        sid: float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
        for sid in eval_ids
    }
    semantic = {
        sid: source_free_setpoint_delta(bases[sid], direction, q_targets[sid])
        for sid in eval_ids
    }
    y10 = run_intervention_batches(
        adapter,
        samples_by_id,
        eval_ids,
        layer=LAYER,
        token_indices=token_indices,
        deltas_by_id=semantic,
        output_token_ids=token_ids,
        capture_layers=[LAYER],
        batch_size=batch_size,
    )
    contexts: list[tuple[str, int | None, dict[str, np.ndarray]]] = []
    matched: dict[str, np.ndarray] = {}
    reference_norms: dict[str, float] = {}
    for sid in eval_ids:
        source_id = str(samples_by_id[sid].counterfactual_id)
        raw = orthogonal_component(activations[LAYER][source_id], bases[sid], direction)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-8:
            raise RuntimeError("E13 matched orthogonal context is degenerate")
        vector, _ = standardize_orthogonal_context(raw, direction, norm, epsilon=1e-8)
        matched[sid] = vector
        reference_norms[sid] = norm
    contexts.append(("matched", None, matched))
    for seed in RANDOM_CONTEXT_SEEDS:
        unit = random_unit_direction(hidden_dim, seed, orthogonal_to=direction)
        by_id = {
            sid: standardize_orthogonal_context(
                unit, direction, reference_norms[sid], epsilon=1e-8
            )[0]
            for sid in eval_ids
        }
        contexts.append(("random", seed, by_id))
    clean_margins = {sid: _selected_margin(clean[sid]["selected_logits"]) for sid in eval_ids}
    y10_margins = {sid: _selected_margin(y10[sid]["selected_logits"]) for sid in eval_ids}
    rows: list[dict[str, Any]] = []
    max_q_error = 0.0
    max_context_dot = 0.0
    for context, direction_seed, vectors in contexts:
        y01 = run_intervention_batches(
            adapter,
            samples_by_id,
            eval_ids,
            layer=LAYER,
            token_indices=token_indices,
            deltas_by_id=vectors,
            output_token_ids=token_ids,
            capture_layers=[LAYER],
            batch_size=batch_size,
        )
        y11 = run_intervention_batches(
            adapter,
            samples_by_id,
            eval_ids,
            layer=LAYER,
            token_indices=token_indices,
            deltas_by_id={sid: semantic[sid] + vectors[sid] for sid in eval_ids},
            output_token_ids=token_ids,
            capture_layers=[LAYER],
            batch_size=batch_size,
        )
        for sid in eval_ids:
            target_label = 1 - int(labels[sid])
            orientation = 1.0 if target_label == 1 else -1.0
            estimates = factorial_components(
                orientation * clean_margins[sid],
                orientation * y10_margins[sid],
                orientation * _selected_margin(y01[sid]["selected_logits"]),
                orientation * _selected_margin(y11[sid]["selected_logits"]),
            )
            q_after = coordinate_value(y11[sid]["captured"][LAYER], direction)
            max_q_error = max(max_q_error, abs(q_after - q_targets[sid]))
            max_context_dot = max(max_context_dot, abs(float(np.dot(vectors[sid], direction))))
            rows.append(
                {
                    "regime": regime,
                    "step": step,
                    "base_sample_id": sid,
                    "pair_id": str(samples_by_id[sid].pair_id),
                    "relation_family": str(samples_by_id[sid].metadata["relation"]),
                    "gold_label": int(labels[sid]),
                    "context": context,
                    "direction_seed": direction_seed,
                    "Y00": orientation * clean_margins[sid],
                    "Y10": orientation * y10_margins[sid],
                    "Y01": orientation * _selected_margin(y01[sid]["selected_logits"]),
                    "Y11": orientation * _selected_margin(y11[sid]["selected_logits"]),
                    "Q0": float(estimates["Q0"]),
                    "A": float(estimates["A"]),
                    "Q_context": float(estimates["Q_context"]),
                    "G": float(estimates["G"]),
                    "confirmation_accessed": False,
                }
            )
    factorial = pd.DataFrame(rows)
    aggregate, contrasts = summarize_factorial(
        factorial,
        n_bootstraps=BOOTSTRAP_DRAWS,
        confidence_level=0.95,
        seed=20260830 + step,
    )
    clean_margin_values = np.asarray([clean_margins[sid] for sid in eval_ids])
    b = classification_metrics(y_eval, clean_margin_values)
    q = float(factorial.groupby("base_sample_id")["Q0"].first().mean())
    metrics = {
        "regime": regime,
        "step": step,
        "progress": step / MAX_STEPS if regime != "teacher" else None,
        "B": b,
        "D_native": classification_metrics(y_eval, native_scores),
        "D_frozen_initial_axis": classification_metrics(y_eval, frozen_scores),
        "Q": q,
        "A": contrasts["A_matched_minus_random"]["mean"],
        "A_ci_low": contrasts["A_matched_minus_random"]["ci_low"],
        "A_ci_high": contrasts["A_matched_minus_random"]["ci_high"],
        "G": contrasts["G_matched_minus_random"]["mean"],
        "G_ci_low": contrasts["G_matched_minus_random"]["ci_low"],
        "G_ci_high": contrasts["G_matched_minus_random"]["ci_high"],
        "integrity": {
            "no_op_max_abs_logit_deviation": no_op,
            "max_setpoint_deviation": max_q_error,
            "max_context_dot_u": max_context_dot,
            "finite": bool(np.isfinite(factorial.select_dtypes(include=[np.number])).all().all()),
            "n_eval": len(eval_ids),
        },
        "confirmation_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_table(factorial, output_dir / "factorial_rows.parquet")
    save_table(aggregate, output_dir / "factorial_metrics.parquet")
    save_json(metrics, output_dir / "metrics.json")
    if no_op > 1e-6 or max_context_dot > 1e-10 or not metrics["integrity"]["finite"]:
        raise RuntimeError(f"E13 checkpoint integrity failed for {regime} step {step}")
    return metrics, factorial


def _target_ids(labels: list[int], token_ids: list[int], device) -> torch.Tensor:
    return torch.tensor(
        [token_ids[0] if int(label) == 1 else token_ids[1] for label in labels],
        dtype=torch.long,
        device=device,
    )


def _train_regime(
    regime: str,
    student,
    teacher,
    samples_by_id,
    train_ids,
    labels,
    token_ids,
    evaluate,
    regime_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if regime not in {"R1", "R2"}:
        raise ValueError("trainable E13 regime must be R1 or R2")
    model = student.model
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    rng = np.random.default_rng(TRAINING_SEED)
    order = rng.permutation(train_ids).tolist()
    cursor = 0
    loss_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, MAX_STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_for_step(step)
        micro_losses = []
        micro_ce = []
        micro_kd = []
        for _micro in range(GRAD_ACCUMULATION):
            if cursor + MICROBATCH > len(order):
                order = rng.permutation(train_ids).tolist()
                cursor = 0
            batch_ids = order[cursor : cursor + MICROBATCH]
            cursor += MICROBATCH
            prompts = [samples_by_id[sid].prompt for sid in batch_ids]
            encoded = student.tokenize(prompts)
            input_ids = encoded["input_ids"].to(student.device)
            attention = encoded["attention_mask"].to(student.device)
            positions = _last_positions(attention)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            selected = logits[torch.arange(len(batch_ids), device=student.device), positions]
            teacher_selected = None
            if regime == "R2":
                with torch.inference_mode():
                    teacher_logits = teacher.model(
                        input_ids=input_ids, attention_mask=attention
                    ).logits
                    teacher_selected = teacher_logits[
                        torch.arange(len(batch_ids), device=student.device), positions
                    ]
            targets = _target_ids([labels[sid] for sid in batch_ids], token_ids, student.device)
            loss, parts = distillation_loss(
                selected,
                targets,
                teacher_logits=teacher_selected,
                regime=regime,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite E13 {regime} loss")
            (loss / GRAD_ACCUMULATION).backward()
            micro_losses.append(float(loss.detach()))
            micro_ce.append(parts["ce"])
            micro_kd.append(parts["kd"])
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite E13 {regime} gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_rows.append(
            {
                "regime": regime,
                "step": step,
                "loss": float(np.mean(micro_losses)),
                "ce": float(np.mean(micro_ce)),
                "kd": float(np.mean(micro_kd)),
                "learning_rate": learning_rate_for_step(step),
                "gradient_norm": grad_norm,
            }
        )
        if step in CHECKPOINT_STEPS[1:]:
            model.eval()
            checkpoint_dir = regime_dir / f"step_{step:03d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(
                checkpoint_dir / "model", safe_serialization=True, max_shard_size="2GB"
            )
            metrics, _ = evaluate(regime, step, checkpoint_dir)
            metric_rows.append(metrics)
            model.train()
    return loss_rows, metric_rows


def run_e13_training_smoke() -> Path:
    """Tiny GPU contract: finite gradients, falling losses, deterministic teacher, live hooks."""
    repo_root = Path(__file__).resolve().parents[3]
    samples, _frame, _stats = build_e13_open_corpus((("train", 8, 20261301),))
    batch = samples[:2]
    results: dict[str, Any] = {"confirmation_accessed": False}
    for regime in ("R1", "R2"):
        student_cfg, _ = resolve_config(
            base_path=repo_root / "configs/base.yaml",
            model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
            experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
            overrides=(),
        )
        student = load_adapter(student_cfg)
        teacher = None
        if regime == "R2":
            teacher_cfg, _ = resolve_config(
                base_path=repo_root / "configs/base.yaml",
                model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
                experiment_path=repo_root
                / "configs/experiments/E13_distillation_reliability.yaml",
                overrides=(),
            )
            teacher = load_adapter(teacher_cfg)
            teacher.model.eval()
            for parameter in teacher.model.parameters():
                parameter.requires_grad_(False)
        candidates = candidate_token_id_lists(
            student, list(student_cfg.behavior.candidates_primary)
        )
        token_ids = [int(item[0]) for item in candidates]
        encoded = student.tokenize([sample.prompt for sample in batch])
        input_ids = encoded["input_ids"].to(student.device)
        attention = encoded["attention_mask"].to(student.device)
        positions = _last_positions(attention)
        targets = _target_ids(
            [int(sample.target_label) for sample in batch], token_ids, student.device
        )
        teacher_selected = None
        if teacher is not None:
            with torch.inference_mode():
                first = teacher.model(input_ids=input_ids, attention_mask=attention).logits
                second = teacher.model(input_ids=input_ids, attention_mask=attention).logits
            teacher_determinism = float(torch.max(torch.abs(first - second)))
            teacher_selected = first[
                torch.arange(len(batch), device=student.device), positions
            ]
        else:
            teacher_determinism = 0.0
        optimizer = torch.optim.AdamW(student.model.parameters(), lr=PEAK_LR)
        losses = []
        gradients = []
        student.model.train()
        student.model.config.use_cache = False
        for _step in range(5):
            optimizer.zero_grad(set_to_none=True)
            logits = student.model(input_ids=input_ids, attention_mask=attention).logits
            selected = logits[torch.arange(len(batch), device=student.device), positions]
            loss, _ = distillation_loss(
                selected,
                targets,
                teacher_logits=teacher_selected,
                regime=regime,
            )
            loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0)))
            optimizer.step()
            losses.append(float(loss.detach()))
        student.model.eval()
        activations, _indices, _sites = extract_resid_post_layers(
            student,
            batch,
            layers=[LAYER],
            token_selector=SELECTOR,
            batch_size=2,
        )
        finite_hook = bool(
            np.isfinite(np.stack(list(activations[LAYER].values()))).all()
        )
        results[regime] = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "gradients_finite": bool(np.isfinite(gradients).all()),
            "teacher_logits_deterministic": teacher_determinism == 0.0,
            "evaluation_hook_finite": finite_hook,
        }
        if not all(
            (
                results[regime]["loss_decreased"],
                results[regime]["gradients_finite"],
                results[regime]["teacher_logits_deterministic"],
                results[regime]["evaluation_hook_finite"],
            )
        ):
            raise RuntimeError(f"E13 {regime} GPU training smoke failed")
        del student
        if teacher is not None:
            del teacher
        gc.collect()
        torch.cuda.empty_cache()
    output = repo_root / "runs" / "E13_SMOKE" / "training_contract.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, output)
    return output


def run_e13_bounded(
    base_path=None,
    experiment_path=None,
) -> Path:
    start = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    cfg, provenance = resolve_config(
        base_path=base_path or repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=experiment_path
        or repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    if cfg.experiment.id != "E13" or cfg.experiment.mode != "discovery":
        raise RuntimeError("E13 requires its frozen discovery configuration")
    samples, frame, corpus_stats = build_e13_open_corpus()
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    split_hash = dataset_split_hash(dict(zip(frame["sample_id"], frame["split"])))
    shape = {
        "version": E13_VERSION,
        "regimes": ["R0", "R1", "R2"],
        "checkpoint_steps": CHECKPOINT_STEPS,
        "training_seed": TRAINING_SEED,
        "confirmation_accessed": False,
    }
    resolved_hash = hashlib.sha256(
        (config_hash(cfg) + json.dumps(shape, sort_keys=True)).encode()
    ).hexdigest()
    run_id = make_run_id("E13", resolved_hash, TRAINING_SEED, "qwen3-distillation", split_hash)
    canonical = repo_root / cfg.project.output_root / "E13" / run_id
    existing = StatusFile.load(canonical)
    if existing is not None and existing.is_complete():
        return canonical
    run_dir = allocate_run_dir(repo_root / cfg.project.output_root, "E13", run_id)
    status = StatusFile.create(run_dir, run_id, "E13")
    save_resolved_config(cfg, run_dir / "config.resolved.yaml", {**provenance, "shape": shape})
    save_table(frame, run_dir / "corpus_rows.parquet")
    save_json(corpus_stats, run_dir / "corpus_stats.json")
    manifest = RunManifest(run_dir)
    manifest.set_start(resolved_hash, {**provenance, "shape": shape}, cfg.effective_seeds())
    manifest.update_dataset_info(
        split_hash=split_hash,
        prompt_hash_sample=prompt_hash(str(frame.iloc[0]["prompt"])),
        confirmation_accessed=False,
        corpus_stats=corpus_stats,
    )
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    all_metrics: list[dict[str, Any]] = []
    try:
        # Teacher reference on the same fresh, open corpus.
        status.update(message="evaluating frozen E13 teacher")
        teacher_cfg, _ = resolve_config(
            base_path=repo_root / "configs/base.yaml",
            model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
            experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
            overrides=(),
        )
        teacher = load_adapter(teacher_cfg)
        teacher.model.eval()
        teacher_candidates = candidate_token_id_lists(
            teacher, list(teacher_cfg.behavior.candidates_primary)
        )
        if any(len(item) != 1 for item in teacher_candidates):
            raise RuntimeError("E13 teacher Yes/No candidates are not single-token")
        token_ids = [int(item[0]) for item in teacher_candidates]
        teacher_reference = _reference_from_model(
            teacher,
            samples_by_id,
            frame,
            labels,
            token_ids,
            int(cfg.runtime.batch_size),
        )
        teacher_metrics, _ = _evaluate_checkpoint(
            teacher,
            samples_by_id,
            frame,
            labels,
            token_ids,
            teacher_reference,
            regime="teacher",
            step=0,
            output_dir=run_dir / "teacher",
            batch_size=int(cfg.runtime.batch_size),
        )
        all_metrics.append(teacher_metrics)
        del teacher
        gc.collect()
        torch.cuda.empty_cache()

        # Initial student defines the frozen student intervention axis/targets.
        status.update(message="evaluating frozen E13 student baseline")
        student = load_adapter(cfg)
        student.model.eval()
        student_candidates = candidate_token_id_lists(
            student, list(cfg.behavior.candidates_primary)
        )
        student_token_ids = [int(item[0]) for item in student_candidates]
        if student_token_ids != token_ids:
            raise RuntimeError("E13 teacher/student candidate token IDs differ")
        student_reference = _reference_from_model(
            student,
            samples_by_id,
            frame,
            labels,
            token_ids,
            int(cfg.runtime.batch_size),
        )
        np.savez(
            run_dir / "initial_student_reference.npz",
            **student_reference["probe"],
            direction=student_reference["direction"],
        )
        save_json(student_reference["targets"], run_dir / "initial_student_targets.json")

        def evaluate(active_adapter, regime, step, directory):
            return _evaluate_checkpoint(
                active_adapter,
                samples_by_id,
                frame,
                labels,
                token_ids,
                student_reference,
                regime=regime,
                step=step,
                output_dir=directory,
                batch_size=int(cfg.runtime.batch_size),
            )

        baseline_metrics, _ = evaluate(student, "R0", 0, run_dir / "R0" / "step_000")
        all_metrics.append(baseline_metrics)
        del student
        gc.collect()
        torch.cuda.empty_cache()

        loss_tables = []
        train_ids = frame.loc[frame["split"] == "train", "sample_id"].astype(str).tolist()
        for regime in ("R1", "R2"):
            status.update(message=f"training bounded E13 {regime}")
            student = load_adapter(cfg)
            teacher = None
            if regime == "R2":
                teacher = load_adapter(teacher_cfg)
                teacher.model.eval()
                for parameter in teacher.model.parameters():
                    parameter.requires_grad_(False)
            regime_dir = run_dir / regime
            regime_dir.mkdir(parents=True, exist_ok=True)
            step_zero = {**baseline_metrics, "regime": regime, "reused_from": "R0"}
            all_metrics.append(step_zero)

            def callback(name, step, directory, active_student=student):
                return evaluate(active_student, name, step, directory)

            loss_rows, metric_rows = _train_regime(
                regime,
                student,
                teacher,
                samples_by_id,
                train_ids,
                labels,
                token_ids,
                callback,
                regime_dir,
            )
            loss_tables.extend(loss_rows)
            all_metrics.extend(metric_rows)
            del student
            if teacher is not None:
                del teacher
            gc.collect()
            torch.cuda.empty_cache()
        metrics_frame = pd.json_normalize(all_metrics, sep=".")
        save_table(metrics_frame, run_dir / "checkpoint_metrics.parquet")
        save_table(pd.DataFrame(loss_tables), run_dir / "training_losses.parquet")
        teacher_row = next(item for item in all_metrics if item["regime"] == "teacher")
        baseline_row = next(item for item in all_metrics if item["regime"] == "R0")
        gap_rows = []
        for item in all_metrics:
            if item["regime"] not in {"R1", "R2"}:
                continue
            gap_rows.append(
                {
                    "regime": item["regime"],
                    "step": item["step"],
                    **{
                        f"T_{metric}": teacher_gap_closure(
                            item[metric], baseline_row[metric], teacher_row[metric]
                        )
                        for metric in ("Q", "A", "G")
                    },
                    "T_B": teacher_gap_closure(
                        item["B"]["auroc"],
                        baseline_row["B"]["auroc"],
                        teacher_row["B"]["auroc"],
                    ),
                }
            )
        save_table(pd.DataFrame(gap_rows), run_dir / "teacher_gap_closure.parquet")
        summary = {
            "status": "complete",
            "run_id": run_id,
            "teacher": teacher_row,
            "baseline": baseline_row,
            "checkpoints": all_metrics,
            "corpus_stats": corpus_stats,
            "wall_time_s": time.monotonic() - start,
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "e13_metrics.json")
        manifest.update_model_info(
            teacher="Qwen/Qwen3-1.7B",
            student="Qwen/Qwen3-0.6B",
            candidate_token_ids=token_ids,
            layer=LAYER,
            site=SITE,
            token_selector=SELECTOR,
        )
        manifest.finish([{"regimes": ["R0", "R1", "R2"], "status": "complete"}])
        status.complete("E13 one-seed bounded diagnostic complete")
        return run_dir
    except Exception as exc:
        logger.exception("E13 bounded diagnostic failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.manifest.setdefault("errors", []).append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        manifest.finish()
        raise
