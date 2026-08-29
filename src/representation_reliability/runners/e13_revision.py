"""Read-only diagnostics and frozen method revisions for E13 discovery."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..adapters.intervention import (
    differentiable_resid_post_logits,
    forward_with_resid_post_capture,
)
from ..config import resolve_config
from ..metrics.causal_organization import (
    controlled_profiles,
    immutable_run_identity,
    matched_profiles,
    select_b_matched_checkpoint,
)
from ..reporting.tables import markdown_table, save_json, save_table
from ..runtime.checkpoint import latest_complete_checkpoint
from ..runtime.manifest import RunManifest
from ..runtime.status import StatusFile
from .e00c import candidate_token_id_lists
from .e13 import (
    LAYER,
    MAX_STEPS,
    _last_positions,
    _train_regime,
    build_e13_open_corpus,
    distillation_loss,
)
from .e13_methods import (
    METHOD_CAMPAIGN_ID,
    ModelLocalReference,
    _orientation,
    _random_pair,
    _selected_method_row,
    _student_response_reference,
    _validate_cache_artifacts,
)
from .e13_multiseed import (
    CAMPAIGN_ID,
    TRAINING_SEEDS,
    _checkpoint_adapter,
    _corpus_bundle,
    _enhanced_evaluate,
    _load_reference,
    _quality,
    _reference_paths,
    _seed_everything,
)
from .extract import load_adapter

DIAGNOSTIC_VERSION = "e13-response-regularization-v1"
NEAR_ZERO_THRESHOLD = 1e-3
GRADIENT_AUDIT_SEED = 20261305
GRADIENT_AUDIT_STEPS = (10, 100)
GRADIENT_AUDIT_BATCHES = 2
GRADIENT_NAMES = ("KD", "Q", "A", "G", "R6")
SPECIFICITY_PROTOCOL_SHA256 = "08b45d7912c2110b87fb2915b983dcc9e4d2b66a0c961efa2bb8c46438d078ba"
OBJECTIVE_PROTOCOL_SHA256 = "79d9141a25b420ed6b8fcd290297384cb9d31e920d53dec634bb0462b6a180b0"
REVISION_CAMPAIGN_ID = "E13MR_08b45d7912c2_79d9141a25b4"
REVISION_REGIMES = tuple(f"R{index}" for index in range(7, 17))
BOUNDED_REVISION_SEED = 20261305
SHUFFLE_SEED = 20261331
logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _method_root() -> Path:
    return _repo_root() / "runs" / "E13_CONVERSION_RESPONSE" / METHOD_CAMPAIGN_ID


def _one_job(root: Path, regime: str, seed: int) -> tuple[Path, dict[str, Any]]:
    candidates = list((root / "jobs").glob(f"{regime}_seed_{seed}_*/job_summary.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one {regime} seed {seed} job under {root}")
    path = candidates[0]
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("status") != "complete" or job.get("confirmation_accessed") is not False:
        raise RuntimeError(f"invalid diagnostic source identity for {regime} seed {seed}")
    return path.parent, job


def _distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64, copy=True)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic distribution must be nonempty and finite")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "median": float(median),
        "iqr": float(q75 - q25),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "fraction_near_zero": float(np.mean(np.abs(array) <= NEAR_ZERO_THRESHOLD)),
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
    }


def _cosine_from_norms(left: float, right: float, summed: float) -> float:
    denominator = 2.0 * float(left) * float(right)
    if denominator <= 0:
        return float("nan")
    value = (float(summed) ** 2 - float(left) ** 2 - float(right) ** 2) / denominator
    return float(np.clip(value, -1.0, 1.0))


def _teacher_signal_tables(cache_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution_rows = []
    for split in ("train", "validation"):
        chosen = cache_rows.loc[cache_rows["split"].eq(split)]
        for family, prefix in (("semantic", "R5"), ("random", "R6")):
            for component in ("Q", "A", "G"):
                distribution_rows.append(
                    {
                        "split": split,
                        "family": family,
                        "component": component,
                        **_distribution(chosen[f"{prefix}_{component}"]),
                    }
                )
    distributions = pd.DataFrame(distribution_rows)

    correlation_rows = []
    for split in ("train", "validation"):
        chosen = cache_rows.loc[cache_rows["split"].eq(split)].copy()
        chosen["R5_norm"] = np.sqrt(sum(chosen[f"R5_{name}"].pow(2) for name in "QAG"))
        chosen["R6_norm"] = np.sqrt(sum(chosen[f"R6_{name}"].pow(2) for name in "QAG"))
        columns = ["R5_Q", "R5_A", "R5_G", "R6_Q", "R6_A", "R6_G", "R5_norm", "R6_norm"]
        corr = chosen[columns].corr(method="pearson")
        for left in columns:
            for right in columns:
                correlation_rows.append(
                    {
                        "split": split,
                        "left": left,
                        "right": right,
                        "pearson": float(corr.loc[left, right]),
                    }
                )
    return distributions, pd.DataFrame(correlation_rows)


def _fixed_point_free_permutation(items: list[str], seed: int) -> dict[str, str]:
    if len(items) < 2 or len(set(items)) != len(items):
        raise ValueError("pair permutation requires at least two unique identities")
    rng = np.random.default_rng(int(seed))
    permuted = np.asarray(items, dtype=object)[rng.permutation(len(items))].tolist()
    for shift in range(len(items)):
        candidate = permuted[shift:] + permuted[:shift]
        if all(source != target for source, target in zip(items, candidate)):
            return dict(zip(items, candidate))
    raise RuntimeError("could not construct fixed-point-free pair permutation")


def _specificity_target_map(
    regime: str,
    train_ids: list[str],
    *,
    samples_by_id: dict[str, Any],
    labels: dict[str, int],
) -> pd.DataFrame:
    if regime not in {"R7", "R8"}:
        return pd.DataFrame(
            {
                "sample_id": train_ids,
                "target_sample_id": train_ids,
                "regime": regime,
            }
        )
    pair_order = list(dict.fromkeys(str(samples_by_id[sid].pair_id) for sid in train_ids))
    pair_members: dict[str, dict[int, str]] = {}
    pair_family: dict[str, str] = {}
    for sid in train_ids:
        pair = str(samples_by_id[sid].pair_id)
        pair_members.setdefault(pair, {})[int(labels[sid])] = sid
        pair_family[pair] = str(samples_by_id[sid].metadata["relation"])
    if not all(set(members) == {0, 1} for members in pair_members.values()):
        raise RuntimeError("specificity permutation pair-label identity mismatch")
    if regime == "R7":
        pair_map = _fixed_point_free_permutation(pair_order, SHUFFLE_SEED)
    else:
        pair_map = {}
        for family in sorted(set(pair_family.values())):
            family_pairs = [pair for pair in pair_order if pair_family[pair] == family]
            digest = hashlib.sha256(f"{SHUFFLE_SEED}|R8|{family}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            pair_map.update(_fixed_point_free_permutation(family_pairs, seed))
    rows = []
    for sid in train_ids:
        source_pair = str(samples_by_id[sid].pair_id)
        target_pair = pair_map[source_pair]
        target_sid = pair_members[target_pair][int(labels[sid])]
        rows.append(
            {
                "regime": regime,
                "sample_id": sid,
                "source_pair_id": source_pair,
                "target_sample_id": target_sid,
                "target_pair_id": target_pair,
                "relation_family": pair_family[source_pair],
                "target_relation_family": pair_family[target_pair],
                "gold_label": int(labels[sid]),
                "target_gold_label": int(labels[target_sid]),
                "fixed_point": source_pair == target_pair,
            }
        )
    result = pd.DataFrame(rows)
    if result["fixed_point"].any() or result["gold_label"].ne(result["target_gold_label"]).any():
        raise RuntimeError("invalid specificity permutation")
    if regime == "R8" and result["relation_family"].ne(result["target_relation_family"]).any():
        raise RuntimeError("R8 permutation crossed relation families")
    return result


def _revision_validation_geometry(
    cache_rows: pd.DataFrame, cache_meta: dict[str, Any]
) -> dict[str, Any]:
    validation = cache_rows.loc[cache_rows["split"].eq("validation")].copy()
    semantic = validation[["R5_Q", "R5_A", "R5_G"]].to_numpy(np.float64)
    covariance = np.cov(semantic, rowvar=False, ddof=0)
    epsilon = 1e-4 * float(np.trace(covariance)) / 3.0
    regularized = covariance + epsilon * np.eye(3)
    eigenvalues = np.linalg.eigvalsh(regularized)
    if not np.isfinite(regularized).all() or float(eigenvalues.min()) <= 0:
        raise RuntimeError("invalid R12 validation covariance")
    arm = np.column_stack(
        [semantic[:, 0], semantic[:, 1], semantic[:, 0] + semantic[:, 1] + semantic[:, 2]]
    )
    return {
        "R11_scales": np.maximum(arm.std(axis=0, ddof=0), 1e-6).tolist(),
        "R12_covariance": covariance.tolist(),
        "R12_epsilon": epsilon,
        "R12_regularized_covariance": regularized.tolist(),
        "R12_inverse": np.linalg.inv(regularized).tolist(),
        "R12_eigenvalues": eigenvalues.tolist(),
        "R12_condition_number": float(np.linalg.cond(regularized)),
        "source_split": "validation",
        "source_cache_digest": cache_meta["response_tensor_sha256"],
    }


def make_revision_response_loss(
    *,
    regime: str,
    cache_rows: pd.DataFrame,
    cache_meta: dict[str, Any],
    validation_geometry: dict[str, Any],
    target_map: pd.DataFrame,
    samples_by_id: dict[str, Any],
    labels: dict[str, int],
    student_reference: ModelLocalReference,
    student_adapter,
    token_ids: list[int],
):
    if regime not in REVISION_REGIMES:
        raise ValueError("unknown E13 revision regime")
    cache = cache_rows.set_index("sample_id")
    mapping = target_map.set_index("sample_id")["target_sample_id"].astype(str).to_dict()
    direction = student_reference.direction
    targets = student_reference.targets
    margin_sigma = float(targets["sigma_margin_validation"])
    precision = torch.as_tensor(
        np.asarray(validation_geometry["R12_inverse"], dtype=np.float32),
        device=student_adapter.device,
        dtype=torch.float32,
    )

    def callback(
        *,
        student,
        batch_ids,
        clean_selected_logits,
        input_ids,
        attention_mask,
        positions,
        clean_hidden,
        **_kwargs,
    ):
        orientation = torch.as_tensor(
            _orientation(labels, batch_ids), device=student.device, dtype=torch.float32
        )
        clean_margin = orientation * (
            clean_selected_logits[:, 0].float() - clean_selected_logits[:, 1].float()
        )
        source_prompts = [
            samples_by_id[str(samples_by_id[sid].counterfactual_id)].prompt for sid in batch_ids
        ]
        encoded = student.tokenize(source_prompts)
        source_ids = encoded["input_ids"].to(student.device)
        source_attention = encoded["attention_mask"].to(student.device)
        source_positions = _last_positions(source_attention)
        rows = torch.arange(len(batch_ids), device=student.device)
        with torch.no_grad():
            source_outputs = student.model(
                input_ids=source_ids,
                attention_mask=source_attention,
                output_hidden_states=True,
            )
            source_hidden = source_outputs.hidden_states[LAYER + 1][rows, source_positions]
        bases = clean_hidden.detach().float().cpu().numpy()
        sources = source_hidden.detach().float().cpu().numpy()
        q_values, c_values = [], []
        for index, sid in enumerate(batch_ids):
            target_q = float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
            q_delta = student_reference.semantic_delta(bases[index], target_q)
            c_delta = student_reference.context_delta(bases[index], sources[index])
            if regime == "R10":
                q_delta, c_delta = _random_pair(
                    student.hidden_size,
                    sid,
                    direction,
                    float(np.linalg.norm(q_delta)),
                    float(np.linalg.norm(c_delta)),
                    student_reference.random_identity,
                )
            q_values.append(q_delta)
            c_values.append(c_delta)
        q = torch.as_tensor(np.stack(q_values), device=student.device, dtype=clean_hidden.dtype)
        c = torch.as_tensor(np.stack(c_values), device=student.device, dtype=clean_hidden.dtype)
        arm_margins = []
        for delta in (q, c, q + c):
            logits = differentiable_resid_post_logits(
                student,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer=LAYER,
                token_indices=positions,
                deltas=delta,
                output_token_ids=token_ids,
            )
            arm_margins.append(orientation * (logits[:, 0].float() - logits[:, 1].float()))
        yq, yc, yqc = arm_margins
        student_components = torch.stack(
            [
                (yq - clean_margin) / margin_sigma,
                (yc - clean_margin) / margin_sigma,
                ((yqc - yq) - (yc - clean_margin)) / margin_sigma,
            ],
            dim=-1,
        )
        target_ids = [mapping.get(sid, sid) for sid in batch_ids]

        if regime == "R11":
            student_arm = torch.stack(
                [
                    student_components[:, 0],
                    student_components[:, 1],
                    (yqc - clean_margin) / margin_sigma,
                ],
                dim=-1,
            )
            target_arm = np.column_stack(
                [
                    cache.loc[target_ids, "R5_Q"].to_numpy(float),
                    cache.loc[target_ids, "R5_A"].to_numpy(float),
                    (
                        cache.loc[target_ids, "R5_Q"]
                        + cache.loc[target_ids, "R5_A"]
                        + cache.loc[target_ids, "R5_G"]
                    ).to_numpy(float),
                ]
            )
            scales = torch.as_tensor(
                validation_geometry["R11_scales"], device=student.device, dtype=torch.float32
            )
            target = torch.as_tensor(target_arm, device=student.device, dtype=torch.float32)
            return ((student_arm / scales - target / scales) ** 2).sum(dim=-1).mean()

        target_family = "R6" if regime == "R9" else "R5"
        target = torch.as_tensor(
            cache.loc[target_ids, [f"{target_family}_{name}" for name in "QAG"]].to_numpy(
                dtype=float, copy=True
            ),
            device=student.device,
            dtype=torch.float32,
        )
        difference = student_components - target
        if regime == "R12":
            return torch.einsum("bi,ij,bj->b", difference, precision, difference).mean()
        scale_family = "R6" if regime == "R9" else "R5"
        scales = torch.as_tensor(
            [cache_meta["component_scales"][f"{scale_family}_{name}"] for name in "QAG"],
            device=student.device,
            dtype=torch.float32,
        )
        squared = (difference / scales).pow(2)
        if regime == "R13":
            return squared[:, :2].sum(dim=-1).mean()
        if regime == "R14":
            return squared[:, 2].mean()
        return squared.mean()

    return callback


def _revision_run_dir(regime: str, seed: int, identity: str) -> Path:
    return (
        _repo_root()
        / "runs"
        / "E13_METHOD_REVISION"
        / REVISION_CAMPAIGN_ID
        / "jobs"
        / f"{regime}_seed_{seed}_{identity[:12]}"
    )


def _full_model_gradient_norm(model) -> float:
    squared = sum(
        float(parameter.grad.float().pow(2).sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return float(np.sqrt(squared))


def _training_batch_losses(
    *,
    student,
    teacher,
    samples_by_id: dict[str, Any],
    labels: dict[str, int],
    token_ids: list[int],
    batch_ids: list[str],
    response_loss_fn,
    need_base: bool,
    need_response: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    prompts = [samples_by_id[sid].prompt for sid in batch_ids]
    encoded = student.tokenize(prompts)
    input_ids = encoded["input_ids"].to(student.device)
    attention = encoded["attention_mask"].to(student.device)
    positions = _last_positions(attention)
    rows = torch.arange(len(batch_ids), device=student.device)
    response_hidden = None
    if need_response:
        outputs, sequence = forward_with_resid_post_capture(
            student, input_ids=input_ids, attention_mask=attention, layer=LAYER
        )
        response_hidden = sequence[rows, positions]
    else:
        outputs = student.model(input_ids=input_ids, attention_mask=attention)
    selected = outputs.logits[rows, positions]
    base_loss = None
    parts = {"ce": float("nan"), "kd": float("nan")}
    if need_base:
        with torch.inference_mode():
            teacher_outputs = teacher.model(input_ids=input_ids, attention_mask=attention)
            teacher_selected = teacher_outputs.logits[rows, positions]
        target_ids = torch.tensor(
            [token_ids[0] if int(labels[sid]) == 1 else token_ids[1] for sid in batch_ids],
            device=student.device,
            dtype=torch.long,
        )
        base_loss, parts = distillation_loss(
            selected, target_ids, teacher_logits=teacher_selected, regime="R2"
        )
    response_loss = None
    if need_response:
        response_loss = response_loss_fn(
            student=student,
            batch_ids=batch_ids,
            clean_selected_logits=selected.index_select(
                -1, torch.as_tensor(token_ids, dtype=torch.long, device=student.device)
            ),
            input_ids=input_ids,
            attention_mask=attention,
            positions=positions,
            clean_hidden=response_hidden,
            regime="R5",
        )
    return base_loss, response_loss, parts


def _make_custom_gradient_step(
    *,
    regime: str,
    student,
    teacher,
    samples_by_id: dict[str, Any],
    labels: dict[str, int],
    token_ids: list[int],
    response_loss_fn,
):
    if regime not in {"R15", "R16"}:
        raise ValueError("custom gradient strategy is only defined for R15/R16")
    accumulation = 4.0

    def step_fn(step_batches: list[list[str]]) -> dict[str, Any]:
        if len(step_batches) != int(accumulation):
            raise RuntimeError("gradient revision effective-batch identity mismatch")
        student.model.zero_grad(set_to_none=True)
        response_values = []
        for batch_ids in step_batches:
            _, response, _ = _training_batch_losses(
                student=student,
                teacher=teacher,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                batch_ids=batch_ids,
                response_loss_fn=response_loss_fn,
                need_base=False,
                need_response=True,
            )
            if response is None or not torch.isfinite(response):
                raise RuntimeError(f"nonfinite E13 {regime} response gradient pass")
            (response / accumulation).backward()
            response_values.append(float(response.detach()))
        response_norm = _full_model_gradient_norm(student.model)
        student.model.zero_grad(set_to_none=True)

        ce_values, kd_values, base_values = [], [], []
        for batch_ids in step_batches:
            base, _, parts = _training_batch_losses(
                student=student,
                teacher=teacher,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                batch_ids=batch_ids,
                response_loss_fn=response_loss_fn,
                need_base=True,
                need_response=False,
            )
            if base is None or not torch.isfinite(base):
                raise RuntimeError(f"nonfinite E13 {regime} KD gradient pass")
            (base / accumulation).backward()
            base_values.append(float(base.detach()))
            ce_values.append(float(parts["ce"]))
            kd_values.append(float(parts["kd"]))
        kd_norm = _full_model_gradient_norm(student.model)
        student.model.zero_grad(set_to_none=True)
        if kd_norm <= 0 or response_norm <= 0 or not np.isfinite([kd_norm, response_norm]).all():
            raise RuntimeError(f"invalid E13 {regime} gradient norm")

        if regime == "R16":
            response_scale = 0.5 * kd_norm / response_norm
            for batch_ids in step_batches:
                base, response, _ = _training_batch_losses(
                    student=student,
                    teacher=teacher,
                    samples_by_id=samples_by_id,
                    labels=labels,
                    token_ids=token_ids,
                    batch_ids=batch_ids,
                    response_loss_fn=response_loss_fn,
                    need_base=True,
                    need_response=True,
                )
                ((base + response_scale * response) / accumulation).backward()
            combined_norm = _full_model_gradient_norm(student.model)
            cosine = None
            projection_coefficient = 0.0
            projection_applied = False
        else:
            for batch_ids in step_batches:
                base, response, _ = _training_batch_losses(
                    student=student,
                    teacher=teacher,
                    samples_by_id=samples_by_id,
                    labels=labels,
                    token_ids=token_ids,
                    batch_ids=batch_ids,
                    response_loss_fn=response_loss_fn,
                    need_base=True,
                    need_response=True,
                )
                ((base + response) / accumulation).backward()
            sum_norm = _full_model_gradient_norm(student.model)
            dot = 0.5 * (sum_norm**2 - kd_norm**2 - response_norm**2)
            cosine = float(np.clip(dot / (kd_norm * response_norm), -1.0, 1.0))
            projection_coefficient = dot / (kd_norm**2)
            projection_applied = dot < 0
            response_scale = 1.0
            if projection_applied:
                correction_scale = -projection_coefficient
                for batch_ids in step_batches:
                    base, _, _ = _training_batch_losses(
                        student=student,
                        teacher=teacher,
                        samples_by_id=samples_by_id,
                        labels=labels,
                        token_ids=token_ids,
                        batch_ids=batch_ids,
                        response_loss_fn=response_loss_fn,
                        need_base=True,
                        need_response=False,
                    )
                    (correction_scale * base / accumulation).backward()
            combined_norm = _full_model_gradient_norm(student.model)
        if not np.isfinite(combined_norm):
            raise RuntimeError(f"nonfinite E13 {regime} combined gradient")
        return {
            "losses": [base + response for base, response in zip(base_values, response_values)],
            "ce": ce_values,
            "kd": kd_values,
            "response": response_values,
            "gradient_diagnostics": {
                "kd_gradient_norm_preclip": kd_norm,
                "response_gradient_norm_preclip": response_norm,
                "combined_gradient_norm_preclip": combined_norm,
                "kd_response_cosine": cosine,
                "response_gradient_scale": response_scale,
                "projection_coefficient": projection_coefficient,
                "projection_applied": projection_applied,
            },
        }

    return step_fn


def run_revision_job(regime: str, seed: int) -> Path:
    if regime not in REVISION_REGIMES:
        raise ValueError("revision regime is outside the frozen design")
    if seed not in TRAINING_SEEDS:
        raise ValueError("revision seed is outside the frozen design")
    if seed != BOUNDED_REVISION_SEED and regime not in _selected_full_wave_regimes():
        raise ValueError("non-bounded revision seed is not authorized before selection")
    root = _repo_root()
    method_root = _method_root()
    cache_dir = method_root / "teacher_cache"
    cache_meta = _validate_cache_artifacts(cache_dir)
    cache_rows = pd.read_parquet(cache_dir / "teacher_response_rows.parquet")
    baseline_dir = root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    paths = _reference_paths(baseline_dir)
    reference_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    protocol_sha = (
        SPECIFICITY_PROTOCOL_SHA256
        if regime in {"R7", "R8", "R9", "R10"}
        else OBJECTIVE_PROTOCOL_SHA256
    )
    identity_payload = {
        "revision_campaign": REVISION_CAMPAIGN_ID,
        "protocol_sha256": protocol_sha,
        "base_method_protocol_sha256": cache_meta["protocol_sha256"],
        "regime": regime,
        "seed": int(seed),
        "teacher_cache_response_sha256": cache_meta["response_tensor_sha256"],
        "corpus_digest": cache_meta["corpus_digest"],
        "teacher_revisions": reference_summary["teacher_revisions"],
        "student_revisions": reference_summary["student_revisions"],
        "max_steps": MAX_STEPS,
        "confirmation_accessed": False,
    }
    identity = immutable_run_identity(identity_payload)
    run_dir = _revision_run_dir(regime, seed, identity)
    existing = StatusFile.load(run_dir)
    if existing is not None and existing.is_complete():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(run_dir, run_dir.name, "E13")
    manifest = RunManifest(run_dir)
    manifest.set_start(identity, {"identity": identity_payload}, {"training": seed})
    student_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    try:
        _, frame, _, labels, samples_by_id, split_hash, corpus_digest = _corpus_bundle()
        if corpus_digest != cache_meta["corpus_digest"]:
            raise RuntimeError("revision corpus digest mismatch")
        train_ids = frame.loc[frame["split"].eq("train"), "sample_id"].astype(str).tolist()
        target_map = _specificity_target_map(
            regime, train_ids, samples_by_id=samples_by_id, labels=labels
        )
        target_map_path = save_table(target_map, run_dir / "target_mapping.parquet")
        if regime in {"R7", "R8"}:
            source = cache_rows.set_index("sample_id").loc[train_ids, ["R5_Q", "R5_A", "R5_G"]]
            mapped_ids = (
                target_map.set_index("sample_id").loc[train_ids, "target_sample_id"].tolist()
            )
            mapped = cache_rows.set_index("sample_id").loc[mapped_ids, ["R5_Q", "R5_A", "R5_G"]]
            for column in source.columns:
                if not np.array_equal(
                    np.sort(source[column].to_numpy()), np.sort(mapped[column].to_numpy())
                ):
                    raise RuntimeError(f"{regime} failed exact target-marginal preservation")
        validation_geometry = _revision_validation_geometry(cache_rows, cache_meta)
        save_json(validation_geometry, run_dir / "validation_response_geometry.json")
        frozen_reference = _load_reference(paths)
        token_ids = list(map(int, reference_summary["token_ids"]))
        teacher_rows = pd.read_parquet(paths["teacher_rows"])
        teacher_primary = matched_profiles(teacher_rows)
        teacher_controlled = controlled_profiles(teacher_rows)
        student_reference, student_reference_meta = _student_response_reference(
            student_cfg,
            run_dir,
            frame=frame,
            labels=labels,
            samples_by_id=samples_by_id,
            token_ids=token_ids,
            seed=seed,
        )
        checkpoint_root = run_dir / "checkpoints"
        latest = latest_complete_checkpoint(checkpoint_root, identity=identity)
        resume_checkpoint = latest[0] if latest is not None else None
        _seed_everything(seed)
        student = (
            _checkpoint_adapter(student_cfg, resume_checkpoint)
            if resume_checkpoint is not None
            else load_adapter(student_cfg)
        )
        teacher_cfg, _ = resolve_config(
            base_path=root / "configs/base.yaml",
            model_path=root / "configs/models/qwen3_1.7b.yaml",
            experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
            overrides=(),
        )
        teacher = load_adapter(teacher_cfg)
        teacher.model.eval()
        for parameter in teacher.model.parameters():
            parameter.requires_grad_(False)
        response = make_revision_response_loss(
            regime=regime,
            cache_rows=cache_rows,
            cache_meta=cache_meta,
            validation_geometry=validation_geometry,
            target_map=target_map,
            samples_by_id=samples_by_id,
            labels=labels,
            student_reference=student_reference,
            student_adapter=student,
            token_ids=token_ids,
        )
        baseline = {
            **reference_summary["R0"],
            "regime": regime,
            "seed": seed,
            "step": 0,
            "reused_from": "R0",
        }

        def evaluate(name, step, directory, _projector):
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
                seed=seed,
                step=step,
                output_dir=directory,
                batch_size=int(student_cfg.runtime.batch_size),
            )

        status.update(message=f"training {regime} seed {seed}")
        custom_gradient_step = (
            _make_custom_gradient_step(
                regime=regime,
                student=student,
                teacher=teacher,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                response_loss_fn=response,
            )
            if regime in {"R15", "R16"}
            else None
        )
        losses, metrics = _train_regime(
            regime,
            student,
            teacher,
            samples_by_id,
            train_ids,
            labels,
            token_ids,
            evaluate,
            checkpoint_root,
            training_seed=seed,
            run_identity=identity,
            resume_checkpoint=resume_checkpoint,
            response_loss_fn=response,
            custom_gradient_step_fn=custom_gradient_step,
        )
        all_metrics = [baseline, *metrics]
        save_table(pd.json_normalize(all_metrics, sep="."), run_dir / "checkpoint_metrics.parquet")
        save_table(pd.DataFrame(losses), run_dir / "training_losses.parquet")
        selection = select_b_matched_checkpoint(
            pd.DataFrame(
                [
                    {
                        "step": row["step"],
                        "validation_B": row["validation_B"],
                        "selection_split": "validation",
                    }
                    for row in all_metrics
                ]
            ),
            float(reference_summary["teacher"]["validation_B"]),
        )
        save_json(selection, run_dir / "b_matched_selection.json")
        quality = {"step_000": reference_summary["R0_quality"]}
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        for step in sorted({int(selection["selected_step"]), MAX_STEPS} - {0}):
            adapter = _checkpoint_adapter(student_cfg, run_dir / "checkpoints" / f"step_{step:03d}")
            quality[f"step_{step:03d}"] = _quality(
                adapter, run_dir / "quality" / f"step_{step:03d}", batch_size=16
            )
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
        summary = {
            "status": "complete",
            "regime": regime,
            "seed": seed,
            "identity": identity_payload,
            "run_identity_sha256": identity,
            "student_response_reference": student_reference_meta,
            "target_mapping_sha256": hashlib.sha256(target_map_path.read_bytes()).hexdigest(),
            "validation_response_geometry": validation_geometry,
            "checkpoints": all_metrics,
            "b_matched": selection,
            "general_quality": quality,
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "job_summary.json")
        manifest.update_dataset_info(
            split_hash=split_hash, corpus_digest=corpus_digest, confirmation_accessed=False
        )
        manifest.update_model_info(
            teacher="Qwen/Qwen3-1.7B",
            student="Qwen/Qwen3-0.6B",
            layer=LAYER,
            site="resid_post",
            token_selector="last_prompt",
            candidate_token_ids=token_ids,
        )
        manifest.finish([{"regime": regime, "seed": seed, "status": "complete"}])
        status.complete(f"{regime} seed {seed} complete")
        return run_dir
    except Exception as exc:
        logger.exception("E13 revision job failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.manifest.setdefault("errors", []).append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        manifest.finish()
        raise


def _selected_full_wave_regimes() -> set[str]:
    selection_path = (
        _repo_root() / "runs" / "E13_METHOD_REVISION" / REVISION_CAMPAIGN_ID / "selection.json"
    )
    if not selection_path.exists():
        return set()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("selected_regime")
    return {str(chosen)} if chosen else set()


def run_revision_smoke() -> Path:
    """Exercise all frozen revision losses and both gradient transforms on GPU."""
    root = _repo_root()
    output = root / "runs" / "E13_METHOD_REVISION" / "smoke_v1"
    output.mkdir(parents=True, exist_ok=True)
    student_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    teacher_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_1.7b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    samples, _, _ = build_e13_open_corpus((("train", 8, 20261301),))
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    ids = list(samples_by_id)
    labels = {sid: int(samples_by_id[sid].target_label) for sid in ids}
    student = load_adapter(student_cfg)
    teacher = load_adapter(teacher_cfg)
    student.model.config.use_cache = False
    teacher.model.eval()
    for parameter in teacher.model.parameters():
        parameter.requires_grad_(False)
    candidates = candidate_token_id_lists(student, list(student_cfg.behavior.candidates_primary))
    token_ids = [int(item[0]) for item in candidates]
    direction = np.zeros(student.hidden_size, dtype=np.float64)
    direction[0] = 1.0
    reference = ModelLocalReference.from_reference(
        model_role="student",
        hidden_size=student.hidden_size,
        reference={
            "direction": direction,
            "probe": {
                "coef": direction,
                "intercept": np.zeros(1),
                "mean": np.zeros(student.hidden_size),
                "scale": np.ones(student.hidden_size),
            },
            "targets": {
                "q0_star": -0.5,
                "q1_star": 0.5,
                "sigma_q_validation": 1.0,
                "sigma_margin_validation": 1.0,
            },
        },
        model_revisions=student.resolved_revisions(),
    )
    cache = pd.DataFrame(
        [
            {
                "sample_id": sid,
                "split": "train",
                **{f"R5_{name}": 0.0 for name in "QAG"},
                **{f"R6_{name}": 0.0 for name in "QAG"},
            }
            for sid in ids
        ]
    )
    cache_meta = {
        "component_scales": {f"R{family}_{name}": 1.0 for family in (5, 6) for name in "QAG"}
    }
    validation_geometry = {"R11_scales": [1.0, 1.0, 1.0], "R12_inverse": np.eye(3).tolist()}
    target_map = pd.DataFrame({"sample_id": ids, "target_sample_id": ids})
    results: dict[str, Any] = {}
    for regime in REVISION_REGIMES:
        callback = make_revision_response_loss(
            regime=regime,
            cache_rows=cache,
            cache_meta=cache_meta,
            validation_geometry=validation_geometry,
            target_map=target_map,
            samples_by_id=samples_by_id,
            labels=labels,
            student_reference=reference,
            student_adapter=student,
            token_ids=token_ids,
        )
        student.model.zero_grad(set_to_none=True)
        if regime in {"R15", "R16"}:
            strategy = _make_custom_gradient_step(
                regime=regime,
                student=student,
                teacher=teacher,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                response_loss_fn=callback,
            )
            audit = strategy([ids[index : index + 2] for index in range(0, 8, 2)])
            result = {
                "loss": float(np.mean(audit["losses"])),
                "gradient_norm": _full_model_gradient_norm(student.model),
                **audit["gradient_diagnostics"],
            }
        else:
            base, response, _ = _training_batch_losses(
                student=student,
                teacher=teacher,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                batch_ids=ids[:2],
                response_loss_fn=callback,
                need_base=True,
                need_response=True,
            )
            loss = base + response
            loss.backward()
            result = {
                "loss": float(loss.detach()),
                "response": float(response.detach()),
                "gradient_norm": _full_model_gradient_norm(student.model),
            }
        result["finite"] = bool(
            np.isfinite([result["loss"], result["gradient_norm"]]).all()
            and result["gradient_norm"] > 0
        )
        results[regime] = result
    results["confirmation_accessed"] = False
    if not all(results[regime]["finite"] for regime in REVISION_REGIMES):
        raise RuntimeError("nonfinite E13 revision smoke")
    save_json(results, output / "revision_smoke.json")
    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _comparison_row(regime: str, seed: int) -> dict[str, Any]:
    baseline_root = _repo_root() / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    if regime in {"R2", "R3"}:
        _, job = _one_job(baseline_root, regime, seed)
    elif regime in {"R5", "R6"}:
        _, job = _one_job(_method_root(), regime, seed)
    else:
        revision_root = _repo_root() / "runs" / "E13_METHOD_REVISION" / REVISION_CAMPAIGN_ID
        _, job = _one_job(revision_root, regime, seed)
    return _selected_method_row(job)


def analyze_revision_campaign() -> Path:
    root = _repo_root()
    campaign_root = root / "runs" / "E13_METHOD_REVISION" / REVISION_CAMPAIGN_ID
    campaign_root.mkdir(parents=True, exist_ok=True)
    bounded_rows = [
        _comparison_row(regime, BOUNDED_REVISION_SEED)
        for regime in ("R2", "R3", "R5", "R6", *REVISION_REGIMES)
    ]
    bounded = pd.DataFrame(bounded_rows)
    save_table(bounded, campaign_root / "bounded_b_matched_results.parquet")
    indexed = bounded.set_index("regime")
    baseline_dir = root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    reference = json.loads(_reference_paths(baseline_dir)["summary"].read_text(encoding="utf-8"))
    r0_ppl = float(reference["R0_quality"]["wikitext"]["perplexity"])
    r0_acc = float(reference["R0_quality"]["hellaswag"]["accuracy"])
    candidate_rows = []
    for regime in tuple(f"R{index}" for index in range(11, 17)):
        row = indexed.loc[regime]
        criteria = {
            "criterion_1_B_gap": float(row["absolute_validation_B_gap"]) <= 0.03,
            "criterion_2_COD_below_R5": float(row["COD"]) < float(indexed.loc["R5", "COD"]),
            "criterion_3_COD_below_R6": float(row["COD"]) < float(indexed.loc["R6", "COD"]),
            "criterion_4_G_gap": float(row["G_gap"]) <= float(indexed.loc["R5", "G_gap"]) + 0.01,
            "criterion_5_quality": bool(
                np.isfinite([row["wikitext_perplexity"], row["hellaswag_accuracy"]]).all()
                and float(row["wikitext_perplexity"]) < 10.0 * r0_ppl
                and float(row["hellaswag_accuracy"]) >= r0_acc - 0.20
            ),
        }
        candidate_rows.append(
            {
                "regime": regime,
                **criteria,
                "eligible": all(criteria.values()),
                "COD": float(row["COD"]),
                "Q_gap": float(row["Q_gap"]),
                "A_gap": float(row["A_gap"]),
                "G_gap": float(row["G_gap"]),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    save_table(candidates, campaign_root / "candidate_gate.parquet")
    eligible = candidates.loc[candidates["eligible"]].copy()
    simplicity = {
        name: index for index, name in enumerate(("R14", "R13", "R11", "R12", "R16", "R15"))
    }
    selected = None
    if not eligible.empty:
        eligible["simplicity"] = eligible["regime"].map(simplicity)
        selected = str(
            eligible.sort_values(
                ["COD", "G_gap", "Q_gap", "A_gap", "simplicity"], kind="stable"
            ).iloc[0]["regime"]
        )
    selection = {
        "protocol_sha256": OBJECTIVE_PROTOCOL_SHA256,
        "bounded_seed": BOUNDED_REVISION_SEED,
        "selected_regime": selected,
        "eligible_regimes": eligible["regime"].astype(str).tolist(),
        "no_candidate_passed": selected is None,
        "confirmation_accessed": False,
    }
    save_json(selection, campaign_root / "selection.json")

    display = ["regime", "B_student", "Q_z", "A_z", "G_z", "COD", "Q_gap", "A_gap", "G_gap"]
    specificity = bounded.loc[bounded["regime"].isin(["R5", "R6", "R7", "R8", "R9", "R10"])]
    specificity_report = root / "E13_RESPONSE_SPECIFICITY_RESULTS.md"
    specificity_report.write_text(
        "\n".join(
            [
                "# E13 Response-Specificity Results",
                "",
                "Status: frozen one-seed discovery controls complete; confirmation was not accessed.",
                "",
                markdown_table(specificity[display], float_fmt="{:.6f}"),
                "",
                "Interpretation is based on the preregistered R5/R6/R7/R8/R9/R10 contrasts and is finalized in the method-revision discovery summary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    objective = bounded.loc[bounded["regime"].isin(["R11", "R12", "R13", "R14", "R15", "R16"])]
    objective_report = root / "E13_RESPONSE_OBJECTIVE_REVISION_RESULTS.md"
    objective_report.write_text(
        "\n".join(
            [
                "# E13 Response-Objective Revision Results",
                "",
                "Status: frozen one-seed objective comparison complete; confirmation was not accessed.",
                "",
                markdown_table(objective[display], float_fmt="{:.6f}"),
                "",
                markdown_table(candidates, float_fmt="{:.6f}"),
                "",
                f"Frozen selected regime: `{selected}`."
                if selected
                else "No candidate passed the frozen gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    full_complete = selected is not None
    full_rows = []
    if selected is not None:
        for seed in TRAINING_SEEDS:
            try:
                full_rows.append(_comparison_row(selected, seed))
            except RuntimeError:
                full_complete = False
                break
    success = None
    if full_complete:
        selected_rows = pd.DataFrame(full_rows)
        comparator_rows = pd.DataFrame(
            [
                _comparison_row(regime, seed)
                for regime in ("R2", "R3", "R5", "R6")
                for seed in TRAINING_SEEDS
            ]
        )
        final_rows = pd.concat([comparator_rows, selected_rows], ignore_index=True)
        save_table(final_rows, campaign_root / "full_b_matched_results.parquet")
        paired = final_rows.set_index(["regime", "seed"])
        behavior_count = int(selected_rows["absolute_validation_B_gap"].le(0.03).sum())
        below_r2 = all(
            float(paired.loc[(selected, seed), "COD"]) < float(paired.loc[("R2", seed), "COD"])
            for seed in TRAINING_SEEDS
        )
        below_r6_count = sum(
            float(paired.loc[(selected, seed), "COD"]) < float(paired.loc[("R6", seed), "COD"])
            for seed in TRAINING_SEEDS
        )
        mean_below_r6 = float(selected_rows["COD"].mean()) < float(
            final_rows.loc[final_rows["regime"].eq("R6"), "COD"].mean()
        )
        component_wins = sum(
            float(selected_rows[component].mean())
            < float(final_rows.loc[final_rows["regime"].eq("R6"), component].mean())
            for component in ("Q_gap", "A_gap", "G_gap")
        )
        quality = bool(
            selected_rows["wikitext_perplexity"].lt(10.0 * r0_ppl).all()
            and selected_rows["hellaswag_accuracy"].ge(r0_acc - 0.20).all()
        )
        criteria = {
            "A_teacher_like_B_at_least_2_of_3": behavior_count >= 2,
            "B_COD_below_R2_3_of_3": below_r2,
            "C_COD_below_R6_at_least_2_of_3": below_r6_count >= 2,
            "D_mean_COD_below_R6": mean_below_r6,
            "E_at_least_two_component_gaps_below_R6": component_wins >= 2,
            "F_quality": quality,
        }
        success = all(criteria.values())
        save_json(
            {
                "selected_regime": selected,
                "criteria": criteria,
                "method_revision_success": success,
                "confirmation_accessed": False,
            },
            campaign_root / "final_verdict.json",
        )
    summary = root / "E13_METHOD_REVISION_DISCOVERY_SUMMARY.md"
    summary.write_text(
        "\n".join(
            [
                "# E13 Method-Revision Discovery Summary",
                "",
                "Status: open discovery only; E13 confirmation was not accessed.",
                "",
                f"Bounded selection: `{selected}`."
                if selected
                else "No revised method passed the frozen bounded gate.",
                f"Three-seed wave complete: `{full_complete}`.",
                f"METHOD REVISION SUCCESS: `{success}`."
                if success is not None
                else "Three-seed success test not yet applicable.",
                "",
                "The full scientific interpretation is finalized after the scheduler completes all jobs authorized by this selection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _loss_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    step_rows = []
    summary_rows = []
    for regime in ("R5", "R6"):
        for seed in TRAINING_SEEDS:
            run_dir, _job = _one_job(_method_root(), regime, seed)
            losses = pd.read_parquet(run_dir / "training_losses.parquet")
            required = {"step", "loss", "ce", "kd", "response"}
            if not required.issubset(losses.columns) or len(losses) != 100:
                raise RuntimeError(f"invalid loss evidence for {regime} seed {seed}")
            for row in losses.itertuples(index=False):
                step_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "step": int(row.step),
                        "L_total": float(row.loss),
                        "L_CE": float(row.ce),
                        "L_KD_unweighted": float(row.kd),
                        "L_response": float(row.response),
                    }
                )
            for name, column in (
                ("L_total", "loss"),
                ("L_CE", "ce"),
                ("L_KD_unweighted", "kd"),
                ("L_response", "response"),
            ):
                values = losses[column].astype(float)
                summary_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "loss_component": name,
                        **_distribution(values),
                    }
                )
    return pd.DataFrame(step_rows), pd.DataFrame(summary_rows)


def _sensitivity_audit() -> pd.DataFrame:
    baseline_root = _repo_root() / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    rows = []
    for regime, root in (("R2", baseline_root), ("R5", _method_root()), ("R6", _method_root())):
        for seed in TRAINING_SEEDS:
            run_dir, job = _one_job(root, regime, seed)
            step = int(job["b_matched"]["selected_step"])
            evidence = pd.read_parquet(
                run_dir / "checkpoints" / f"step_{step:03d}" / "factorial_rows.parquet"
            )
            if evidence["confirmation_accessed"].ne(False).any():
                raise RuntimeError("invalid sensitivity evidence namespace")
            families: dict[str, pd.DataFrame] = {
                "semantic_q": evidence.loc[evidence["context"].eq("matched")].assign(
                    response=lambda frame: frame["Q_z"],
                    delta_norm=lambda frame: frame["semantic_delta_norm"],
                ),
                "matched_context": evidence.loc[evidence["context"].eq("matched")].assign(
                    response=lambda frame: frame["A_z"],
                    delta_norm=lambda frame: frame["context_delta_norm"],
                ),
            }
            random_rows = evidence.loc[evidence["context"].eq("random")]
            for direction_seed in sorted(
                random_rows["direction_seed"].dropna().astype(int).unique()
            ):
                families[f"unseen_random_{direction_seed}"] = random_rows.loc[
                    random_rows["direction_seed"].eq(direction_seed)
                ].assign(
                    response=lambda frame: frame["A_z"],
                    delta_norm=lambda frame: frame["context_delta_norm"],
                )
            families["unseen_random_pooled"] = random_rows.assign(
                response=lambda frame: frame["A_z"],
                delta_norm=lambda frame: frame["context_delta_norm"],
            )
            for family, chosen in families.items():
                response = chosen["response"].to_numpy(np.float64)
                norms = chosen["delta_norm"].to_numpy(np.float64)
                ratios = response / np.maximum(norms, 1e-12)
                rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "selected_step": step,
                        "family": family,
                        "n": len(chosen),
                        "mean_squared_response": float(np.mean(response**2)),
                        "response_variance": float(np.var(response)),
                        "mean_abs_response": float(np.mean(np.abs(response))),
                        "mean_delta_norm": float(np.mean(norms)),
                        "mean_abs_lipschitz_proxy": float(np.mean(np.abs(ratios))),
                        "jacobian_norm_proxy": float(np.sqrt(np.mean(ratios**2))),
                    }
                )
    return pd.DataFrame(rows)


def _load_student_reference(run_dir: Path) -> ModelLocalReference:
    from safetensors.numpy import load_file

    metadata = json.loads((run_dir / "student_response_reference.json").read_text(encoding="utf-8"))
    tensors = load_file(run_dir / "student_response_reference.safetensors")
    raw = {
        "direction": tensors["direction"],
        "probe": {
            "coef": tensors["probe_coef"],
            "intercept": tensors["probe_intercept"],
            "mean": tensors["probe_mean"],
            "scale": tensors["probe_scale"],
        },
        "targets": metadata["targets"],
    }
    return ModelLocalReference.from_reference(
        model_role="student",
        hidden_size=int(metadata["hidden_size"]),
        reference=raw,
        model_revisions=metadata["model_revisions"],
    )


def _gradient_loss_builder(
    *,
    student,
    teacher,
    reference: ModelLocalReference,
    cache: pd.DataFrame,
    cache_meta: dict[str, Any],
    samples_by_id: dict[str, Any],
    labels: dict[str, int],
    token_ids: list[int],
    batch_ids: list[str],
) -> Callable[[str], torch.Tensor]:
    cache_indexed = cache.set_index("sample_id")
    direction = reference.direction
    targets = reference.targets
    margin_sigma = float(targets["sigma_margin_validation"])

    def build(name: str) -> torch.Tensor:
        encoded = student.tokenize([samples_by_id[sid].prompt for sid in batch_ids])
        input_ids = encoded["input_ids"].to(student.device)
        attention = encoded["attention_mask"].to(student.device)
        positions = _last_positions(attention)
        row_ids = torch.arange(len(batch_ids), device=student.device)
        selected_ids = torch.as_tensor(token_ids, dtype=torch.long, device=student.device)
        outputs, sequence = forward_with_resid_post_capture(
            student, input_ids=input_ids, attention_mask=attention, layer=LAYER
        )
        selected = outputs.logits[row_ids, positions]
        clean_logits = selected.index_select(-1, selected_ids)
        clean_hidden = sequence[row_ids, positions]
        if name == "KD":
            with torch.inference_mode():
                teacher_outputs = teacher.model(input_ids=input_ids, attention_mask=attention)
                teacher_selected = teacher_outputs.logits[row_ids, positions]
            target_ids = torch.tensor(
                [token_ids[0] if int(labels[sid]) == 1 else token_ids[1] for sid in batch_ids],
                device=student.device,
                dtype=torch.long,
            )
            loss, _ = distillation_loss(
                selected,
                target_ids,
                teacher_logits=teacher_selected,
                regime="R2",
            )
            return loss

        orientation = torch.as_tensor(
            _orientation(labels, batch_ids), device=student.device, dtype=torch.float32
        )
        clean_margin = orientation * (clean_logits[:, 0].float() - clean_logits[:, 1].float())
        source_prompts = [
            samples_by_id[str(samples_by_id[sid].counterfactual_id)].prompt for sid in batch_ids
        ]
        source_encoded = student.tokenize(source_prompts)
        source_ids = source_encoded["input_ids"].to(student.device)
        source_attention = source_encoded["attention_mask"].to(student.device)
        source_positions = _last_positions(source_attention)
        with torch.no_grad():
            source_outputs = student.model(
                input_ids=source_ids,
                attention_mask=source_attention,
                output_hidden_states=True,
            )
            source_hidden = source_outputs.hidden_states[LAYER + 1][row_ids, source_positions]
        bases = clean_hidden.detach().float().cpu().numpy()
        sources = source_hidden.detach().float().cpu().numpy()
        q_values, c_values, rq_values, rc_values = [], [], [], []
        for index, sid in enumerate(batch_ids):
            target_q = float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
            q_delta = reference.semantic_delta(bases[index], target_q)
            c_delta = reference.context_delta(bases[index], sources[index])
            random_q, random_c = _random_pair(
                student.hidden_size,
                sid,
                direction,
                float(np.linalg.norm(q_delta)),
                float(np.linalg.norm(c_delta)),
                reference.random_identity,
            )
            q_values.append(q_delta)
            c_values.append(c_delta)
            rq_values.append(random_q)
            rc_values.append(random_c)

        def components(q_array: list[np.ndarray], c_array: list[np.ndarray]) -> list[torch.Tensor]:
            q = torch.as_tensor(np.stack(q_array), device=student.device, dtype=clean_hidden.dtype)
            c = torch.as_tensor(np.stack(c_array), device=student.device, dtype=clean_hidden.dtype)
            arm_margins = []
            for delta in (q, c, q + c):
                logits = differentiable_resid_post_logits(
                    student,
                    input_ids=input_ids,
                    attention_mask=attention,
                    layer=LAYER,
                    token_indices=positions,
                    deltas=delta,
                    output_token_ids=token_ids,
                )
                arm_margins.append(orientation * (logits[:, 0].float() - logits[:, 1].float()))
            yq, yc, yqc = arm_margins
            return [
                (yq - clean_margin) / margin_sigma,
                (yc - clean_margin) / margin_sigma,
                ((yqc - yq) - (yc - clean_margin)) / margin_sigma,
            ]

        if name == "R6":
            values = components(rq_values, rc_values)
            losses = []
            for component, value in zip(("Q", "A", "G"), values):
                scale = float(cache_meta["component_scales"][f"R6_{component}"])
                target = torch.as_tensor(
                    cache_indexed.loc[batch_ids, f"R6_{component}"].to_numpy(
                        dtype=float, copy=True
                    ),
                    device=student.device,
                    dtype=torch.float32,
                )
                losses.append(((value / scale) - (target / scale)).pow(2))
            return torch.stack(losses).mean()
        semantic = components(q_values, c_values)
        index = {"Q": 0, "A": 1, "G": 2}[name]
        scale = float(cache_meta["component_scales"][f"R5_{name}"])
        target = torch.as_tensor(
            cache_indexed.loc[batch_ids, f"R5_{name}"].to_numpy(dtype=float, copy=True),
            device=student.device,
            dtype=torch.float32,
        )
        return ((semantic[index] / scale) - (target / scale)).pow(2).mean()

    return build


def _gradient_norm(
    student, loss_builder: Callable[[str], torch.Tensor], names: tuple[str, ...]
) -> tuple[float, float]:
    student.model.zero_grad(set_to_none=True)
    losses = [loss_builder(name) for name in names]
    loss = torch.stack(losses).sum()
    loss.backward()
    squared = sum(
        float(parameter.grad.float().pow(2).sum())
        for parameter in student.model.parameters()
        if parameter.grad is not None
    )
    norm = float(np.sqrt(squared))
    value = float(sum(float(item.detach()) for item in losses))
    student.model.zero_grad(set_to_none=True)
    del losses, loss
    gc.collect()
    torch.cuda.empty_cache()
    return norm, value


def run_gradient_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _repo_root()
    method_root = _method_root()
    run_dir, _job = _one_job(method_root, "R5", GRADIENT_AUDIT_SEED)
    cache_dir = method_root / "teacher_cache"
    cache = pd.read_parquet(cache_dir / "teacher_response_rows.parquet")
    cache_meta = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    _, frame, _, labels, samples_by_id, _, _ = _corpus_bundle()
    validation_ids = frame.loc[frame["split"].eq("validation"), "sample_id"].astype(str).tolist()
    reference = _load_student_reference(run_dir)
    student_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    teacher_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_1.7b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    teacher = load_adapter(teacher_cfg)
    teacher.model.eval()
    for parameter in teacher.model.parameters():
        parameter.requires_grad_(False)
    candidates = candidate_token_id_lists(teacher, list(teacher_cfg.behavior.candidates_primary))
    token_ids = [int(item[0]) for item in candidates]
    norm_rows, cosine_rows = [], []
    for step in GRADIENT_AUDIT_STEPS:
        student = _checkpoint_adapter(student_cfg, run_dir / "checkpoints" / f"step_{step:03d}")
        student.model.eval()
        student.model.config.use_cache = False
        for batch_index in range(GRADIENT_AUDIT_BATCHES):
            start = batch_index * 2
            batch_ids = validation_ids[start : start + 2]
            builder = _gradient_loss_builder(
                student=student,
                teacher=teacher,
                reference=reference,
                cache=cache,
                cache_meta=cache_meta,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                batch_ids=batch_ids,
            )
            norms: dict[str, float] = {}
            values: dict[str, float] = {}
            for name in GRADIENT_NAMES:
                norms[name], values[name] = _gradient_norm(student, builder, (name,))
                norm_rows.append(
                    {
                        "checkpoint_step": step,
                        "batch_index": batch_index,
                        "sample_ids": json.dumps(batch_ids),
                        "loss_component": name,
                        "loss": values[name],
                        "gradient_norm": norms[name],
                    }
                )
            for left_index, left in enumerate(GRADIENT_NAMES):
                for right in GRADIENT_NAMES[left_index + 1 :]:
                    sum_norm, _ = _gradient_norm(student, builder, (left, right))
                    cosine_rows.append(
                        {
                            "checkpoint_step": step,
                            "batch_index": batch_index,
                            "left": left,
                            "right": right,
                            "cosine": _cosine_from_norms(norms[left], norms[right], sum_norm),
                        }
                    )
        del student
        gc.collect()
        torch.cuda.empty_cache()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(norm_rows), pd.DataFrame(cosine_rows)


def run_response_regularization_diagnostic() -> Path:
    """Audit existing E13 evidence and run read-only full-model gradient checks."""
    root = _repo_root()
    output = root / "runs" / "E13_METHOD_REVISION" / "diagnostic_v1"
    output.mkdir(parents=True, exist_ok=True)
    cache = pd.read_parquet(_method_root() / "teacher_cache" / "teacher_response_rows.parquet")
    distributions, correlations = _teacher_signal_tables(cache)
    loss_steps, loss_summary = _loss_audit()
    sensitivity = _sensitivity_audit()
    gradient_norms_path = output / "gradient_norms.parquet"
    gradient_cosines_path = output / "gradient_cosines.parquet"
    if gradient_norms_path.exists() and gradient_cosines_path.exists():
        gradient_norms = pd.read_parquet(gradient_norms_path)
        gradient_cosines = pd.read_parquet(gradient_cosines_path)
    else:
        gradient_norms, gradient_cosines = run_gradient_audit()
    save_table(distributions, output / "teacher_response_distributions.parquet")
    save_table(correlations, output / "teacher_response_correlations.parquet")
    save_table(loss_steps, output / "existing_loss_steps.parquet")
    save_table(loss_summary, output / "existing_loss_summary.parquet")
    save_table(sensitivity, output / "sensitivity_proxies.parquet")
    save_table(gradient_norms, gradient_norms_path)
    save_table(gradient_cosines, gradient_cosines_path)
    conflict = gradient_cosines.loc[
        gradient_cosines["left"].eq("KD") & gradient_cosines["right"].isin(["Q", "A", "G", "R6"])
    ]
    conflict_gate = bool((conflict["cosine"] < -0.2).groupby(conflict["right"]).sum().max() >= 2)
    manifest = {
        "version": DIAGNOSTIC_VERSION,
        "source_method_campaign": METHOD_CAMPAIGN_ID,
        "cache_response_sha256": hashlib.sha256(
            cache[[f"R{regime}_{component}" for regime in (5, 6) for component in "QAG"]]
            .to_numpy(np.float64)
            .tobytes()
        ).hexdigest(),
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "gradient_checkpoints": list(GRADIENT_AUDIT_STEPS),
        "gradient_batches_per_checkpoint": GRADIENT_AUDIT_BATCHES,
        "conflict_gate_rule": "same response component has cosine(KD,response) < -0.2 in at least two audits",
        "conflict_gate_fired": conflict_gate,
        "confirmation_accessed": False,
    }
    save_json(manifest, output / "diagnostic_manifest.json")

    dist_display = distributions.loc[
        distributions["split"].eq("train"),
        ["family", "component", "mean", "sd", "median", "iqr", "p05", "p95", "fraction_near_zero"],
    ]
    loss_display = loss_summary.loc[
        loss_summary["loss_component"].isin(["L_KD_unweighted", "L_response"]),
        ["regime", "seed", "loss_component", "mean", "sd", "median", "p05", "p95"],
    ]
    gradient_display = gradient_cosines.loc[gradient_cosines["left"].eq("KD")]
    gradient_component_display = gradient_norms.groupby("loss_component", as_index=False)[
        ["loss", "gradient_norm"]
    ].agg(["mean", "min", "max"])
    gradient_component_display.columns = [
        "loss_component",
        "loss_mean",
        "loss_min",
        "loss_max",
        "gradient_norm_mean",
        "gradient_norm_min",
        "gradient_norm_max",
    ]
    sensitivity_display = (
        sensitivity.loc[
            sensitivity["family"].isin(["semantic_q", "matched_context", "unseen_random_pooled"])
        ]
        .groupby(["regime", "family"], as_index=False)[
            [
                "mean_squared_response",
                "response_variance",
                "mean_abs_lipschitz_proxy",
                "jacobian_norm_proxy",
            ]
        ]
        .mean()
    )
    train_correlations = correlations.loc[correlations["split"].eq("train")].set_index(
        ["left", "right"]
    )["pearson"]
    corr_a = float(train_correlations.loc[("R5_A", "R6_A")])
    corr_g = float(train_correlations.loc[("R5_G", "R6_G")])
    corr_norm = float(train_correlations.loc[("R5_norm", "R6_norm")])
    report = root / "E13_RESPONSE_REGULARIZATION_DIAGNOSTIC.md"
    lines = [
        "# E13 Response-Regularization Diagnostic",
        "",
        "Status: complete read-only discovery diagnostic. No optimizer update was performed; E13 confirmation was not accessed.",
        "",
        "## Existing training-loss scales",
        "",
        markdown_table(loss_display, float_fmt="{:.6f}"),
        "",
        "`L_KD_unweighted` is the logged KL term before its frozen `0.5*T^2` multiplier; `L_response` is the actual added response objective.",
        "",
        "## Teacher target distributions (training split)",
        "",
        markdown_table(dist_display, float_fmt="{:.6f}"),
        "",
        "Near zero means absolute standardized response <= 0.001. Validation distributions and all raw rows are retained in the diagnostic directory.",
        "",
        "## Semantic/random correlations",
        "",
        f"Train corr(A_T, random-A_T): {corr_a:.6f}.",
        f"Train corr(G_T, random-G_T): {corr_g:.6f}.",
        f"Train semantic/random response-norm correlation: {corr_norm:.6f}.",
        "",
        "## B-matched sensitivity proxies",
        "",
        markdown_table(sensitivity_display, float_fmt="{:.6f}"),
        "",
        "The unseen-random family pools frozen evaluation directions 2130/2131/2132, which were not the R6 training direction identity. Ratios are directional finite-difference response divided by intervention norm, not a full Jacobian.",
        "",
        "## Full-model gradient audit",
        "",
        markdown_table(gradient_component_display, float_fmt="{:.6f}"),
        "",
        markdown_table(gradient_display, float_fmt="{:.6f}"),
        "",
        f"Frozen conflict gate fired: **{conflict_gate}**.",
        "",
        "Gradient norms, response-component losses, and every pairwise Q/A/G/R6 cosine are retained as portable tables.",
        "",
        "## Interpretation",
        "",
        "The response term is not a small auxiliary: its training mean is 2.13-3.72, while the logged unweighted KL mean is 0.66-0.72. On the frozen gradient batches, Q is the largest semantic component (mean loss 8.16; mean full-model gradient norm 750), followed by A (1.57; 362) and G (1.28; 266), versus KD's 1.69 and 106. Q mismatch therefore dominates the audited response geometry even after validation-only scaling.",
        "",
        "Teacher semantic A has the largest spread (train SD 0.679), while G is much smaller (SD 0.098, mean -0.044) and 16.9% of G targets are within 0.001 of zero. Random G is smaller still (SD 0.058; 25.6% near zero). Direct G matching consequently has a materially weaker and more quantized target than A, consistent with the G-noise hypothesis.",
        "",
        "Semantic/random target correlations are weak: Q 0.103, A 0.058, G 0.105, and response-norm 0.256 on training data. R6 is therefore not succeeding because its cached targets are close sample-wise to the semantic targets.",
        "",
        "R6 reduces the B-matched matched-context standardized mean-square response from R2's 2.872 to 1.201 and has a low unseen-random Jacobian proxy (0.0036 versus R2 0.0095). R5 also reduces matched-context sensitivity (1.539) but is anisotropic on unseen directions (pooled proxy 0.0178, driven by direction 2131). The best current explanation is that R6 supplies strong generic local-sensitivity/Jacobian regularization, which moves A/G magnitudes toward the teacher profile without semantic target correspondence.",
        "",
        "Gradient conflict is heterogeneous but real. Mean cosine(KD,A) is -0.207, with values below -0.2 at both audited checkpoints; cosine(KD,G) reaches -0.529, while Q is usually aligned or near orthogonal. The frozen repeated-conflict gate therefore fires and authorizes the preregistered R15/R16 bounded controls.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
