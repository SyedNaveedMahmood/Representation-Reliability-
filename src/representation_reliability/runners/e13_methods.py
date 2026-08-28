"""Authorized E13 conversion-response methods and controls."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..adapters.intervention import (
    differentiable_resid_post_logits,
    forward_with_resid_post_capture,
    resid_post_hook_count,
)
from ..config import resolve_config
from ..interventions.orthogonal_context import orthogonal_component, standardize_orthogonal_context
from ..interventions.setpoint import source_free_setpoint_delta
from ..interventions.truth_coordinate import random_unit_direction
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
from .e01a import _selected_margin
from .e01a_support import (
    extract_resid_post_layers,
    run_intervention_batches,
    run_unintervened_batches,
)
from .e13 import (
    LAYER,
    MAX_STEPS,
    SELECTOR,
    SITE,
    _last_positions,
    _reference_from_model,
    _train_regime,
    build_e13_open_corpus,
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

METHOD_PROTOCOL_SHA256 = "3f3dd9a65347fc9ba6a20c29686aba11bd578f52b28d87818c399b422325846b"
METHOD_CAMPAIGN_ID = f"E13CR_{METHOD_PROTOCOL_SHA256[:12]}"
METHOD_REGIMES = ("R4", "R5", "R6", "R2C")
RANDOM_RESPONSE_SEED = 20261307
LIVE_CACHE_ROWS = 16
LIVE_CACHE_MAX_ABS = 2e-2
LIVE_CACHE_MEAN_ABS = 2e-3
logger = logging.getLogger(__name__)


def _orientation(labels, ids):
    return np.asarray([1.0 if int(labels[sid]) == 1 else -1.0 for sid in ids])


def _random_pair(
    hidden_dim: int,
    sid: str,
    direction: np.ndarray,
    q_norm: float,
    c_norm: float,
    model_identity: str,
):
    prefix = f"{RANDOM_RESPONSE_SEED}|{model_identity}|{sid}"
    digest = hashlib.sha256(f"{prefix}|q".encode()).digest()
    q_seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    q_unit = random_unit_direction(hidden_dim, q_seed, orthogonal_to=direction)
    digest = hashlib.sha256(f"{prefix}|c".encode()).digest()
    c_seed = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    rng = np.random.default_rng(c_seed)
    raw = rng.normal(size=hidden_dim)
    raw = raw - np.dot(raw, direction) * direction - np.dot(raw, q_unit) * q_unit
    norm = float(np.linalg.norm(raw))
    if norm <= 1e-8:
        raise RuntimeError("degenerate random-response orthogonalization")
    return q_unit * q_norm, raw / norm * c_norm


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_rows(
    ids: list[str],
    *,
    frame: pd.DataFrame,
    labels: dict[str, int],
    samples_by_id: dict[str, Any],
    token_indices: dict[str, int],
    token_sites: dict[str, dict[str, Any]],
    clean: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    margin_sigma: float,
    native_module: str,
) -> pd.DataFrame:
    split_by_id = frame.set_index("sample_id")["split"].astype(str).to_dict()
    rows = []
    for sid in ids:
        orient = 1.0 if labels[sid] == 1 else -1.0
        y0 = orient * _selected_margin(clean[sid])
        values = {
            name: orient * _selected_margin(result[sid]["selected_logits"])
            for name, result in arms.items()
        }
        site = token_sites[sid]
        rows.append(
            {
                "sample_id": sid,
                "pair_id": str(samples_by_id[sid].pair_id),
                "source_id": str(samples_by_id[sid].counterfactual_id),
                "split": split_by_id[sid],
                "raw_text": str(samples_by_id[sid].prompt),
                "Y00": y0,
                "r4_neg": (values["r4_neg"] - y0) / margin_sigma,
                "r4_pos": (values["r4_pos"] - y0) / margin_sigma,
                "R5_Q": (values["r5_q"] - y0) / margin_sigma,
                "R5_A": (values["r5_c"] - y0) / margin_sigma,
                "R5_G": ((values["r5_qc"] - values["r5_q"]) - (values["r5_c"] - y0)) / margin_sigma,
                "R6_Q": (values["r6_q"] - y0) / margin_sigma,
                "R6_A": (values["r6_c"] - y0) / margin_sigma,
                "R6_G": ((values["r6_qc"] - values["r6_q"]) - (values["r6_c"] - y0)) / margin_sigma,
                "selector_strategy": SELECTOR,
                "token_index": int(token_indices[sid]),
                "token_id": int(site["token_id"]),
                "token_text": str(site.get("token_text", "")),
                "char_start": site.get("char_start"),
                "char_end": site.get("char_end"),
                "chat_template_used": bool(site.get("chat_template_used", False)),
                "canonical_site": SITE,
                "native_module": native_module,
                "confirmation_accessed": False,
            }
        )
    return pd.DataFrame(rows)


def prepare_teacher_response_cache() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    baseline_dir = repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    paths = _reference_paths(baseline_dir)
    output = repo_root / "runs" / "E13_CONVERSION_RESPONSE" / METHOD_CAMPAIGN_ID / "teacher_cache"
    status = StatusFile.load(output)
    if status is not None and status.is_complete():
        _validate_cache_artifacts(output)
        return output
    output.mkdir(parents=True, exist_ok=True)
    status = status or StatusFile.create(output, f"{METHOD_CAMPAIGN_ID}-cache", "E13")
    try:
        cfg, _ = resolve_config(
            base_path=repo_root / "configs/base.yaml",
            model_path=repo_root / "configs/models/qwen3_1.7b.yaml",
            experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
            overrides=(),
        )
        (
            _samples,
            frame,
            _stats,
            labels,
            samples_by_id,
            _split_hash,
            corpus_digest,
        ) = _corpus_bundle()
        ids = (
            frame.loc[frame["split"].isin(["train", "validation"]), "sample_id"]
            .astype(str)
            .tolist()
        )
        with paths["summary"].open("r", encoding="utf-8") as handle:
            baseline_summary = json.load(handle)
        if baseline_summary["corpus_digest"] != corpus_digest:
            raise RuntimeError("teacher response-cache corpus digest mismatch")
        reference = _load_reference(paths)
        token_ids = list(map(int, baseline_summary["token_ids"]))
        teacher = load_adapter(cfg)
        teacher.model.eval()
        native_module = teacher.resolve_site(SITE, LAYER).native_module_name
        batch_size = int(cfg.runtime.batch_size)
        activations, token_indices, token_sites = extract_resid_post_layers(
            teacher,
            [samples_by_id[sid] for sid in ids],
            layers=[LAYER],
            token_selector=SELECTOR,
            batch_size=batch_size,
        )
        clean = run_unintervened_batches(
            teacher,
            samples_by_id,
            ids,
            token_indices=token_indices,
            output_token_ids=token_ids,
            batch_size=batch_size,
        )
        base = {sid: activations[LAYER][sid] for sid in ids}
        direction = np.asarray(reference["direction"], dtype=np.float64)
        targets = reference["targets"]
        margin_sigma = float(targets["sigma_margin_validation"])
        q_sigma = float(targets["sigma_q_validation"])
        model_identity = json.dumps(teacher.resolved_revisions(), sort_keys=True)
        semantic: dict[str, np.ndarray] = {}
        context: dict[str, np.ndarray] = {}
        random_q: dict[str, np.ndarray] = {}
        random_c: dict[str, np.ndarray] = {}
        for sid in ids:
            target = float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
            semantic[sid] = source_free_setpoint_delta(base[sid], direction, target)
            source = str(samples_by_id[sid].counterfactual_id)
            raw = orthogonal_component(activations[LAYER][source], base[sid], direction)
            context[sid], _ = standardize_orthogonal_context(
                raw, direction, float(np.linalg.norm(raw)), epsilon=1e-8
            )
            random_q[sid], random_c[sid] = _random_pair(
                teacher.hidden_size,
                sid,
                direction,
                float(np.linalg.norm(semantic[sid])),
                float(np.linalg.norm(context[sid])),
                model_identity,
            )

        all_deltas = {
            "r4_neg": {sid: -q_sigma * direction for sid in ids},
            "r4_pos": {sid: q_sigma * direction for sid in ids},
            "r5_q": semantic,
            "r5_c": context,
            "r5_qc": {sid: semantic[sid] + context[sid] for sid in ids},
            "r6_q": random_q,
            "r6_c": random_c,
            "r6_qc": {sid: random_q[sid] + random_c[sid] for sid in ids},
        }

        def run_arm(
            chosen_ids: list[str],
            deltas: dict[str, np.ndarray],
            adapter=teacher,
        ):
            return run_intervention_batches(
                adapter,
                samples_by_id,
                chosen_ids,
                layer=LAYER,
                token_indices=token_indices,
                deltas_by_id=deltas,
                output_token_ids=token_ids,
                capture_layers=[LAYER],
                batch_size=batch_size,
            )

        arms = {name: run_arm(ids, deltas) for name, deltas in all_deltas.items()}
        evidence = _response_rows(
            ids,
            frame=frame,
            labels=labels,
            samples_by_id=samples_by_id,
            token_indices=token_indices,
            token_sites=token_sites,
            clean=clean,
            arms=arms,
            margin_sigma=margin_sigma,
            native_module=native_module,
        )
        scales = {}
        validation = evidence["split"].eq("validation")
        for regime in ("R5", "R6"):
            for component in ("Q", "A", "G"):
                column = f"{regime}_{component}"
                scales[column] = max(float(evidence.loc[validation, column].std(ddof=0)), 1e-6)

        live_ids = sorted(ids, key=lambda sid: hashlib.sha256(sid.encode()).digest())[
            :LIVE_CACHE_ROWS
        ]
        live_clean = run_unintervened_batches(
            teacher,
            samples_by_id,
            live_ids,
            token_indices=token_indices,
            output_token_ids=token_ids,
            batch_size=batch_size,
        )
        live_arms = {name: run_arm(live_ids, deltas) for name, deltas in all_deltas.items()}
        live = _response_rows(
            live_ids,
            frame=frame,
            labels=labels,
            samples_by_id=samples_by_id,
            token_indices=token_indices,
            token_sites=token_sites,
            clean=live_clean,
            arms=live_arms,
            margin_sigma=margin_sigma,
            native_module=native_module,
        ).set_index("sample_id")
        cached = evidence.set_index("sample_id").loc[live_ids]
        response_columns = [
            "Y00",
            "r4_neg",
            "r4_pos",
            "R5_Q",
            "R5_A",
            "R5_G",
            "R6_Q",
            "R6_A",
            "R6_G",
        ]
        absolute = np.abs(
            live.loc[live_ids, response_columns].to_numpy(float)
            - cached.loc[live_ids, response_columns].to_numpy(float)
        )
        live_check = {
            "sample_ids": live_ids,
            "max_abs": float(absolute.max()),
            "mean_abs": float(absolute.mean()),
            "max_abs_tolerance": LIVE_CACHE_MAX_ABS,
            "mean_abs_tolerance": LIVE_CACHE_MEAN_ABS,
            "passed": bool(
                absolute.max() <= LIVE_CACHE_MAX_ABS and absolute.mean() <= LIVE_CACHE_MEAN_ABS
            ),
        }
        if not live_check["passed"]:
            raise RuntimeError(f"teacher response-cache live check failed: {live_check}")

        input_hasher = hashlib.sha256()
        for sid in ids:
            encoded = teacher.tokenize([samples_by_id[sid].prompt])
            input_hasher.update(sid.encode())
            input_hasher.update(np.asarray(encoded["input_ids"][0].cpu()).tobytes())
        numeric = evidence[response_columns].to_numpy(np.float64)
        table_path = save_table(evidence, output / "teacher_response_rows.parquet")
        metadata = {
            "protocol_sha256": METHOD_PROTOCOL_SHA256,
            "corpus_digest": corpus_digest,
            "teacher_revisions": teacher.resolved_revisions(),
            "ordered_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "input_ids_sha256": input_hasher.hexdigest(),
            "probe_digest": hashlib.sha256(direction.tobytes()).hexdigest(),
            "targets_digest": hashlib.sha256(
                json.dumps(targets, sort_keys=True).encode()
            ).hexdigest(),
            "response_tensor_sha256": hashlib.sha256(numeric.tobytes()).hexdigest(),
            "response_table_sha256": _sha256_file(table_path),
            "response_columns": response_columns,
            "margin_sigma": margin_sigma,
            "q_sigma": q_sigma,
            "component_scales": scales,
            "component_scale_split": "validation",
            "n_rows": len(evidence),
            "dtype": str(teacher.torch_dtype),
            "live_check": live_check,
            "confirmation_accessed": False,
        }
        save_json(metadata, output / "cache_manifest.json")
        _validate_cache_artifacts(output)
        status.complete("teacher response cache complete")
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        return output
    except Exception as exc:
        status.fail(f"{type(exc).__name__}: {exc}")
        raise


def _validate_cache_artifacts(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "cache_manifest.json"
    table_path = cache_dir / "teacher_response_rows.parquet"
    if not manifest_path.exists() or not table_path.exists():
        raise RuntimeError("teacher response-cache artifact is missing")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_sha256") != METHOD_PROTOCOL_SHA256:
        raise RuntimeError("teacher response-cache protocol mismatch")
    if metadata.get("confirmation_accessed") is not False:
        raise RuntimeError("invalid teacher response-cache namespace identity")
    if metadata.get("component_scale_split") != "validation":
        raise RuntimeError("teacher response-cache scale provenance mismatch")
    if metadata.get("response_table_sha256") != _sha256_file(table_path):
        raise RuntimeError("teacher response-cache table digest mismatch")
    if metadata.get("live_check", {}).get("passed") is not True:
        raise RuntimeError("teacher response-cache live check is not passing")
    evidence = pd.read_parquet(table_path)
    if len(evidence) != int(metadata["n_rows"]):
        raise RuntimeError("teacher response-cache row count mismatch")
    numeric = evidence[list(metadata["response_columns"])].to_numpy(np.float64)
    if hashlib.sha256(numeric.tobytes()).hexdigest() != metadata.get("response_tensor_sha256"):
        raise RuntimeError("teacher response-cache tensor digest mismatch")
    ordered = "\n".join(evidence["sample_id"].astype(str)).encode()
    if hashlib.sha256(ordered).hexdigest() != metadata.get("ordered_id_sha256"):
        raise RuntimeError("teacher response-cache ID ordering mismatch")
    return metadata


def make_response_loss(
    regime,
    cache_rows,
    cache_meta,
    samples_by_id,
    labels,
    student_reference,
    token_ids,
    model_identity,
):
    cache = cache_rows.set_index("sample_id")
    direction = np.asarray(student_reference["direction"], dtype=np.float64)
    targets = student_reference["targets"]
    margin_sigma = float(targets["sigma_margin_validation"])
    q_sigma = float(targets["sigma_q_validation"])

    def callback(
        *,
        student,
        batch_ids,
        clean_selected_logits,
        input_ids,
        attention_mask,
        positions,
        clean_hidden,
        regime,
        **_kwargs,
    ):
        orientation = torch.as_tensor(
            _orientation(labels, batch_ids),
            device=student.device,
            dtype=torch.float32,
        )
        clean_margin = orientation * (
            clean_selected_logits[:, 0].float() - clean_selected_logits[:, 1].float()
        )
        if regime == "R4":
            losses = []
            for sign, column in ((-1.0, "r4_neg"), (1.0, "r4_pos")):
                delta = torch.as_tensor(
                    sign * q_sigma * direction,
                    device=student.device,
                    dtype=clean_hidden.dtype,
                ).repeat(len(batch_ids), 1)
                logits = differentiable_resid_post_logits(
                    student,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer=LAYER,
                    token_indices=positions,
                    deltas=delta,
                    output_token_ids=token_ids,
                )
                response = (
                    orientation * (logits[:, 0].float() - logits[:, 1].float()) - clean_margin
                ) / margin_sigma
                target = torch.as_tensor(
                    cache.loc[batch_ids, column].to_numpy(dtype=float, copy=True),
                    device=student.device,
                    dtype=torch.float32,
                )
                losses.append((response - target).pow(2))
            return torch.stack(losses).mean()
        source_prompts = [
            samples_by_id[str(samples_by_id[sid].counterfactual_id)].prompt for sid in batch_ids
        ]
        encoded = student.tokenize(source_prompts)
        source_ids = encoded["input_ids"].to(student.device)
        source_attention = encoded["attention_mask"].to(student.device)
        source_pos = _last_positions(source_attention)
        with torch.no_grad():
            source_out = student.model(
                input_ids=source_ids,
                attention_mask=source_attention,
                output_hidden_states=True,
            )
            source_hidden = source_out.hidden_states[LAYER + 1][
                torch.arange(len(batch_ids), device=student.device), source_pos
            ]
        bases = clean_hidden.detach().float().cpu().numpy()
        sources = source_hidden.detach().float().cpu().numpy()
        q_deltas, c_deltas = [], []
        for index, sid in enumerate(batch_ids):
            target_q = float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
            sem = source_free_setpoint_delta(bases[index], direction, target_q)
            raw = orthogonal_component(sources[index], bases[index], direction)
            ctx, _ = standardize_orthogonal_context(
                raw, direction, float(np.linalg.norm(raw)), epsilon=1e-8
            )
            if regime == "R6":
                sem, ctx = _random_pair(
                    student.hidden_size,
                    sid,
                    direction,
                    float(np.linalg.norm(sem)),
                    float(np.linalg.norm(ctx)),
                    model_identity,
                )
            q_deltas.append(sem)
            c_deltas.append(ctx)
        q = torch.as_tensor(np.stack(q_deltas), device=student.device, dtype=clean_hidden.dtype)
        c = torch.as_tensor(np.stack(c_deltas), device=student.device, dtype=clean_hidden.dtype)
        arm_logits = []
        for delta in (q, c, q + c):
            if regime == "R2C":
                with torch.no_grad():
                    arm_logits.append(
                        differentiable_resid_post_logits(
                            student,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            layer=LAYER,
                            token_indices=positions,
                            deltas=delta,
                            output_token_ids=token_ids,
                        ).detach()
                    )
            else:
                arm_logits.append(
                    differentiable_resid_post_logits(
                        student,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        layer=LAYER,
                        token_indices=positions,
                        deltas=delta,
                        output_token_ids=token_ids,
                    )
                )
        if regime == "R2C":
            return clean_selected_logits.sum() * 0.0
        yq, yc, yqc = [orientation * (x[:, 0].float() - x[:, 1].float()) for x in arm_logits]
        student_components = [
            (yq - clean_margin) / margin_sigma,
            (yc - clean_margin) / margin_sigma,
            ((yqc - yq) - (yc - clean_margin)) / margin_sigma,
        ]
        losses = []
        for component, value in zip(("Q", "A", "G"), student_components):
            scale = float(cache_meta["component_scales"][f"{regime}_{component}"])
            target = torch.as_tensor(
                cache.loc[batch_ids, f"{regime}_{component}"].to_numpy(dtype=float, copy=True),
                device=student.device,
                dtype=torch.float32,
            )
            losses.append(((value / scale) - (target / scale)).pow(2))
        return torch.stack(losses).mean()

    return callback


def _student_response_reference(
    cfg,
    run_dir: Path,
    *,
    frame: pd.DataFrame,
    labels: dict[str, int],
    samples_by_id: dict[str, Any],
    token_ids: list[int],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensor_path = run_dir / "student_response_reference.safetensors"
    metadata_path = run_dir / "student_response_reference.json"
    sites_path = run_dir / "student_response_token_sites.parquet"
    if tensor_path.exists() and metadata_path.exists() and sites_path.exists():
        from safetensors.numpy import load_file

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol_sha256") != METHOD_PROTOCOL_SHA256:
            raise RuntimeError("student response-reference protocol mismatch")
        tensors = load_file(tensor_path)
        if metadata.get("tensor_sha256") != _sha256_file(tensor_path):
            raise RuntimeError("student response-reference tensor digest mismatch")
        if metadata.get("token_sites_sha256") != _sha256_file(sites_path):
            raise RuntimeError("student response-reference token-site digest mismatch")
        reference = {
            "direction": tensors["direction"],
            "probe": {
                "coef": tensors["probe_coef"],
                "intercept": tensors["probe_intercept"],
                "mean": tensors["probe_mean"],
                "scale": tensors["probe_scale"],
            },
            "targets": metadata["targets"],
        }
        return reference, metadata
    if any(path.exists() for path in (tensor_path, metadata_path, sites_path)):
        raise RuntimeError("partial student response-reference artifact")
    _seed_everything(seed)
    adapter = load_adapter(cfg)
    reference_frame = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    reference = _reference_from_model(
        adapter,
        samples_by_id,
        reference_frame,
        labels,
        token_ids,
        int(cfg.runtime.batch_size),
    )
    from safetensors.numpy import save_file

    save_file(
        {
            "direction": np.asarray(reference["direction"], dtype=np.float64),
            "probe_coef": np.asarray(reference["probe"]["coef"], dtype=np.float64),
            "probe_intercept": np.asarray(reference["probe"]["intercept"], dtype=np.float64),
            "probe_mean": np.asarray(reference["probe"]["mean"], dtype=np.float64),
            "probe_scale": np.asarray(reference["probe"]["scale"], dtype=np.float64),
        },
        tensor_path,
    )
    native_module = adapter.resolve_site(SITE, LAYER).native_module_name
    site_rows = []
    for sid in reference_frame["sample_id"].astype(str):
        site = reference["token_sites"][sid]
        site_rows.append(
            {
                "sample_id": sid,
                "raw_text": str(samples_by_id[sid].prompt),
                **site,
                "canonical_site": SITE,
                "layer": LAYER,
                "native_module": native_module,
                "confirmation_accessed": False,
            }
        )
    save_table(pd.DataFrame(site_rows), sites_path)
    metadata = {
        "protocol_sha256": METHOD_PROTOCOL_SHA256,
        "seed": int(seed),
        "model_revisions": adapter.resolved_revisions(),
        "fit_splits": ["train", "validation"],
        "targets": reference["targets"],
        "tensor_sha256": _sha256_file(tensor_path),
        "token_sites_sha256": _sha256_file(sites_path),
        "confirmation_accessed": False,
    }
    save_json(metadata, metadata_path)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return reference, metadata


def run_method_training_smoke() -> Path:
    """Two-row GPU contract for method gradients, hooks, and compute control."""
    repo_root = Path(__file__).resolve().parents[3]
    output = repo_root / "runs" / "E13_METHOD_SMOKE"
    output.mkdir(parents=True, exist_ok=True)
    cfg, _ = resolve_config(
        base_path=repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    samples, _frame, _stats = build_e13_open_corpus((("train", 8, 20261301),))
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    batch = samples[:2]
    batch_ids = [str(sample.sample_id) for sample in batch]
    labels = {str(sample.sample_id): int(sample.target_label) for sample in samples}
    student = load_adapter(cfg)
    student.model.config.use_cache = False
    candidates = candidate_token_id_lists(student, list(cfg.behavior.candidates_primary))
    token_ids = [int(item[0]) for item in candidates]
    direction = np.zeros(student.hidden_size, dtype=np.float64)
    direction[0] = 1.0
    reference = {
        "direction": direction,
        "targets": {
            "q0_star": -0.5,
            "q1_star": 0.5,
            "sigma_q_validation": 1.0,
            "sigma_margin_validation": 1.0,
        },
    }
    cache = pd.DataFrame(
        [
            {
                "sample_id": sid,
                "r4_neg": 0.0,
                "r4_pos": 0.0,
                "R5_Q": 0.0,
                "R5_A": 0.0,
                "R5_G": 0.0,
                "R6_Q": 0.0,
                "R6_A": 0.0,
                "R6_G": 0.0,
            }
            for sid in batch_ids
        ]
    )
    cache_meta = {
        "component_scales": {
            f"{regime}_{component}": 1.0 for regime in ("R5", "R6") for component in ("Q", "A", "G")
        }
    }
    encoded = student.tokenize([sample.prompt for sample in batch])
    input_ids = encoded["input_ids"].to(student.device)
    attention = encoded["attention_mask"].to(student.device)
    positions = _last_positions(attention)
    row_ids = torch.arange(len(batch), device=student.device)
    selected_ids = torch.as_tensor(token_ids, device=student.device)
    results = {}
    hook_count = resid_post_hook_count(student, layer=LAYER)
    for regime in METHOD_REGIMES:
        student.model.zero_grad(set_to_none=True)
        outputs, hidden_sequence = forward_with_resid_post_capture(
            student,
            input_ids=input_ids,
            attention_mask=attention,
            layer=LAYER,
        )
        clean = outputs.logits[row_ids, positions].index_select(-1, selected_ids)
        clean_hidden = hidden_sequence[row_ids, positions]
        callback = make_response_loss(
            regime,
            cache,
            cache_meta,
            samples_by_id,
            labels,
            reference,
            token_ids,
            "smoke-model",
        )
        loss = callback(
            student=student,
            batch_ids=batch_ids,
            clean_selected_logits=clean,
            input_ids=input_ids,
            attention_mask=attention,
            positions=positions,
            clean_hidden=clean_hidden,
            regime=regime,
        )
        loss.backward()
        squared = sum(
            float(parameter.grad.float().pow(2).sum())
            for parameter in student.model.parameters()
            if parameter.grad is not None
        )
        gradient_norm = float(np.sqrt(squared))
        results[regime] = {
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "finite": bool(np.isfinite([float(loss.detach()), gradient_norm]).all()),
        }
    results["R2C"]["zero_gradient"] = results["R2C"]["gradient_norm"] == 0.0
    results["hook_count_before"] = hook_count
    results["hook_count_after"] = resid_post_hook_count(student, layer=LAYER)
    results["confirmation_accessed"] = False
    if not all(results[name]["finite"] for name in METHOD_REGIMES):
        raise RuntimeError("nonfinite E13 method smoke result")
    if not results["R2C"]["zero_gradient"]:
        raise RuntimeError("R2-C response computation contributed a gradient")
    if results["hook_count_before"] != results["hook_count_after"]:
        raise RuntimeError("E13 method smoke leaked an intervention hook")
    save_json(results, output / "training_contract.json")
    return output


def run_method_job(regime: str, seed: int) -> Path:
    if regime not in METHOD_REGIMES or seed not in TRAINING_SEEDS:
        raise ValueError("method job outside frozen design")
    repo_root = Path(__file__).resolve().parents[3]
    cache_dir = prepare_teacher_response_cache()
    cache_meta = _validate_cache_artifacts(cache_dir)
    cache_rows = pd.read_parquet(cache_dir / "teacher_response_rows.parquet")
    baseline_dir = repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    paths = _reference_paths(baseline_dir)
    reference_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    identity_payload = {
        "protocol_sha256": METHOD_PROTOCOL_SHA256,
        "regime": regime,
        "seed": int(seed),
        "teacher_cache_response_sha256": cache_meta["response_tensor_sha256"],
        "teacher_targets_sha256": cache_meta["targets_digest"],
        "corpus_digest": cache_meta["corpus_digest"],
        "teacher_revisions": reference_summary["teacher_revisions"],
        "student_revisions": reference_summary["student_revisions"],
        "max_steps": MAX_STEPS,
        "confirmation_accessed": False,
    }
    identity = immutable_run_identity(identity_payload)
    run_dir = (
        repo_root
        / "runs"
        / "E13_CONVERSION_RESPONSE"
        / METHOD_CAMPAIGN_ID
        / "jobs"
        / f"{regime}_seed_{seed}_{identity[:12]}"
    )
    existing = StatusFile.load(run_dir)
    if existing is not None and existing.is_complete():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status = existing or StatusFile.create(run_dir, run_dir.name, "E13")
    manifest = RunManifest(run_dir)
    manifest.set_start(identity, {"identity": identity_payload}, {"training": seed})
    cfg, _ = resolve_config(
        base_path=repo_root / "configs/base.yaml",
        model_path=repo_root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=repo_root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    try:
        (
            _samples,
            frame,
            _stats,
            labels,
            samples_by_id,
            split_hash,
            corpus_digest,
        ) = _corpus_bundle()
        if corpus_digest != cache_meta["corpus_digest"]:
            raise RuntimeError("method job corpus digest mismatch")
        frozen_reference = _load_reference(paths)
        token_ids = list(map(int, reference_summary["token_ids"]))
        teacher_rows = pd.read_parquet(paths["teacher_rows"])
        teacher_primary = matched_profiles(teacher_rows)
        teacher_controlled = controlled_profiles(teacher_rows)
        student_reference, student_reference_meta = _student_response_reference(
            cfg,
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
            _checkpoint_adapter(cfg, resume_checkpoint)
            if resume_checkpoint is not None
            else load_adapter(cfg)
        )
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
        model_identity = json.dumps(student_reference_meta["model_revisions"], sort_keys=True)
        response = make_response_loss(
            regime,
            cache_rows,
            cache_meta,
            samples_by_id,
            labels,
            student_reference,
            token_ids,
            model_identity,
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
                batch_size=int(cfg.runtime.batch_size),
            )

        train_ids = frame.loc[frame["split"].eq("train"), "sample_id"].astype(str).tolist()
        status.update(message=f"training {regime} seed {seed}")
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
        )
        all_metrics = [baseline, *metrics]
        save_table(
            pd.json_normalize(all_metrics, sep="."),
            run_dir / "checkpoint_metrics.parquet",
        )
        save_table(pd.DataFrame(losses), run_dir / "training_losses.parquet")
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
            selection_rows, float(reference_summary["teacher"]["validation_B"])
        )
        save_json(selection, run_dir / "b_matched_selection.json")
        quality = {"step_000": reference_summary["R0_quality"]}
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
        for step in sorted({int(selection["selected_step"]), MAX_STEPS} - {0}):
            adapter = _checkpoint_adapter(cfg, run_dir / "checkpoints" / f"step_{step:03d}")
            quality[f"step_{step:03d}"] = _quality(
                adapter,
                run_dir / "quality" / f"step_{step:03d}",
                batch_size=16,
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
            "checkpoints": all_metrics,
            "b_matched": selection,
            "general_quality": quality,
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "job_summary.json")
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
        manifest.finish([{"regime": regime, "seed": seed, "status": "complete"}])
        status.complete(f"{regime} seed {seed} complete")
        return run_dir
    except Exception as exc:
        logger.exception("E13 conversion-response job failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.manifest.setdefault("errors", []).append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        manifest.finish()
        raise


def _selected_method_row(job: dict[str, Any]) -> dict[str, Any]:
    step = int(job["b_matched"]["selected_step"])
    selected = next(row for row in job["checkpoints"] if int(row["step"]) == step)
    quality = job["general_quality"][f"step_{step:03d}"]
    return {
        "regime": str(job["regime"]),
        "seed": int(job["seed"]),
        "selected_step": step,
        "B_student": float(selected["B"]["auroc"]),
        "validation_B_student": float(selected["validation_B"]),
        "absolute_validation_B_gap": float(job["b_matched"]["absolute_B_gap"]),
        "Q_z": float(selected["causal_effects"]["Q_z"]),
        "A_z": float(selected["causal_effects"]["A_z"]),
        "G_z": float(selected["causal_effects"]["G_z"]),
        "Q_gap": float(selected["COD"]["mean_abs_Q_z_gap"]),
        "A_gap": float(selected["COD"]["mean_abs_A_z_gap"]),
        "G_gap": float(selected["COD"]["mean_abs_G_z_gap"]),
        "COD": float(selected["COD"]["COD"]),
        "wikitext_perplexity": float(quality["wikitext"]["perplexity"]),
        "hellaswag_accuracy": float(quality["hellaswag"]["accuracy"]),
    }


def analyze_method_campaign(campaign_dir: Path | None = None) -> Path:
    """Compare authorized methods at validation-selected behavior-matched steps."""
    repo_root = Path(__file__).resolve().parents[3]
    campaign_dir = campaign_dir or (
        repo_root / "runs" / "E13_CONVERSION_RESPONSE" / METHOD_CAMPAIGN_ID
    )
    baseline_dir = repo_root / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    rows = []
    for regime in ("R2", "R3"):
        for seed in TRAINING_SEEDS:
            candidates = list(
                (baseline_dir / "jobs").glob(f"{regime}_seed_{seed}_*/job_summary.json")
            )
            if len(candidates) != 1:
                raise RuntimeError(f"expected one baseline {regime} seed {seed} job")
            job = json.loads(candidates[0].read_text(encoding="utf-8"))
            rows.append(_selected_method_row(job))
    for regime in METHOD_REGIMES:
        for seed in TRAINING_SEEDS:
            candidates = list(
                (campaign_dir / "jobs").glob(f"{regime}_seed_{seed}_*/job_summary.json")
            )
            if len(candidates) != 1:
                raise RuntimeError(f"expected one method {regime} seed {seed} job")
            job = json.loads(candidates[0].read_text(encoding="utf-8"))
            if job.get("status") != "complete" or job.get("confirmation_accessed") is not False:
                raise RuntimeError(f"invalid method {regime} seed {seed} identity")
            rows.append(_selected_method_row(job))
    results = pd.DataFrame(rows)
    order = ["R2", "R3", "R2C", "R4", "R5", "R6"]
    results["regime"] = pd.Categorical(results["regime"], order, ordered=True)
    results = results.sort_values(["regime", "seed"]).reset_index(drop=True)
    save_table(results, campaign_dir / "method_b_matched_results.parquet")
    measures = [
        "B_student",
        "Q_z",
        "A_z",
        "G_z",
        "Q_gap",
        "A_gap",
        "G_gap",
        "COD",
        "wikitext_perplexity",
        "hellaswag_accuracy",
    ]
    aggregate = results.groupby("regime", observed=True)[measures].agg(
        ["mean", "std", "min", "max"]
    )
    aggregate.columns = [f"{name}_{stat}" for name, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    save_table(aggregate, campaign_dir / "method_seed_aggregate.parquet")
    by_regime = results.set_index(["regime", "seed"])
    mean_cod = results.groupby("regime", observed=True)["COD"].mean().to_dict()
    comparators = [name for name in order if name != "R5"]
    r5_mean_best = all(mean_cod["R5"] < mean_cod[name] for name in comparators)
    r5_behavior = bool(
        results.loc[results["regime"].eq("R5"), "absolute_validation_B_gap"].le(0.03).all()
    )
    baseline_quality = json.loads(
        _reference_paths(baseline_dir)["summary"].read_text(encoding="utf-8")
    )["R0_quality"]
    base_ppl = float(baseline_quality["wikitext"]["perplexity"])
    base_acc = float(baseline_quality["hellaswag"]["accuracy"])
    r5_rows = results.loc[results["regime"].eq("R5")]
    r5_quality = bool(
        r5_rows["wikitext_perplexity"].lt(10.0 * base_ppl).all()
        and r5_rows["hellaswag_accuracy"].ge(base_acc - 0.20).all()
    )
    all_seed_superiority = all(
        float(by_regime.loc[("R5", seed), "COD"]) < float(by_regime.loc[(comparator, seed), "COD"])
        for seed in TRAINING_SEEDS
        for comparator in comparators
    )
    r2_means = results.loc[results["regime"].eq("R2")][["Q_gap", "A_gap", "G_gap"]].mean()
    r4_means = results.loc[results["regime"].eq("R4")][["Q_gap", "A_gap", "G_gap"]].mean()
    reductions = (r2_means - r4_means) / r2_means.clip(lower=1e-12)
    r4_preferential_q = bool(
        reductions["Q_gap"] > 0
        and reductions["Q_gap"] > reductions["A_gap"]
        and reductions["Q_gap"] > reductions["G_gap"]
    )
    r5_joint = all(
        float(results.loc[results["regime"].eq("R5"), component].mean())
        < float(results.loc[results["regime"].eq(comparator), component].mean())
        for component in ("Q_gap", "A_gap", "G_gap")
        for comparator in comparators
    )
    verdict = {
        "primary_success": bool(r5_mean_best and r5_behavior and r5_quality),
        "R5_lower_mean_COD_than_all_comparators": r5_mean_best,
        "R5_behavior_retained_all_seeds": r5_behavior,
        "R5_quality_controls_pass_all_seeds": r5_quality,
        "R5_lower_COD_than_all_comparators_each_seed": all_seed_superiority,
        "R4_preferential_Q_gap_reduction_vs_R2": r4_preferential_q,
        "R5_jointly_lower_mean_Q_A_G_gaps_than_all_comparators": r5_joint,
        "confirmation_accessed": False,
    }
    save_json(verdict, campaign_dir / "method_verdict.json")
    display_columns = [
        "regime",
        "seed",
        "selected_step",
        "B_student",
        "absolute_validation_B_gap",
        "Q_z",
        "A_z",
        "G_z",
        "COD",
        "wikitext_perplexity",
        "hellaswag_accuracy",
    ]
    report = repo_root / "E13_CONVERSION_RESPONSE_DISCOVERY_SUMMARY.md"
    lines = [
        "# E13 Conversion-Response Distillation Discovery",
        "",
        "Status date: 2026-08-28. Open discovery only; E13 confirmation was not accessed.",
        "",
        f"Protocol SHA-256: `{METHOD_PROTOCOL_SHA256}`  ",
        f"Campaign: `{campaign_dir.relative_to(repo_root)}`",
        "",
        "## Behavior-matched results",
        "",
        markdown_table(results[display_columns], float_fmt="{:.6f}"),
        "",
        "## Frozen primary method test",
        "",
        f"- Primary R5 success: {'PASS' if verdict['primary_success'] else 'FAIL'}",
        f"- R5 lower mean COD than every comparator: {r5_mean_best}",
        f"- R5 retains teacher-like validation B in all seeds: {r5_behavior}",
        f"- R5 quality controls pass in all seeds: {r5_quality}",
        f"- R5 beats every comparator within every seed: {all_seed_superiority}",
        f"- R4 preferentially reduces Q gap versus R2: {r4_preferential_q}",
        f"- R5 jointly lowers mean Q/A/G gaps versus all comparators: {r5_joint}",
        "",
        "All comparisons are descriptive with n=3 seeds. Per-example raw evidence, all checkpoints, controls, and quality artifacts remain in the campaign directory.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
