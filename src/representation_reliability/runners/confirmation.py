"""One-shot locked confirmation of the E01 actionable-representation mechanism."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import config_hash, resolve_config
from ..data.splits import discovery_view
from ..interventions.orthogonal_context import (
    orthogonal_component,
    resolve_context_reference_norm,
    standardize_orthogonal_context,
)
from ..interventions.setpoint import (
    norm_matched_direction_delta,
    source_free_setpoint_delta,
)
from ..interventions.truth_coordinate import coordinate_value, random_unit_direction
from ..metrics.causal import counterfactual_outcome, margin_toward_label
from ..metrics.confirmation import (
    confirmation_classification,
    evaluate_primary_hypotheses,
)
from ..metrics.factorial import factorial_estimands
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import (
    dataset_split_hash,
    environment_manifest,
    project_git_state,
    prompt_hash,
)
from ..runtime.status import StatusFile, atomic_write_json
from .confirmation_support import (
    CONFIRMATION_VERSION,
    CONTEXT_STRENGTHS,
    FROZEN_MODELS,
    TRACE_LAYERS,
    build_confirmation_source_plans,
    fit_locked_probe_layers,
    load_frozen_targets,
    open_access_record,
    route_confirmation_rows,
    source_plan_digest,
    source_plan_frame,
    validate_model_identity,
    validate_protocol_lock,
)
from .e00c import _generate_dataset, candidate_token_id_lists
from .e01a import _prediction, _selected_margin
from .e01a_support import (
    extract_resid_post_layers,
    intervention_base_activations,
    load_activation_snapshot,
    run_intervention_batches,
    run_unintervened_batches,
    save_activation_snapshot,
)
from .e01b2_support import CONTEXT_EPSILON
from .e01b_support import find_identity_matched_e01a_snapshot
from .extract import load_adapter

logger = logging.getLogger(__name__)
SITE = "resid_post"
SELECTOR = "last_prompt"
LAYER = 17
N_RANDOM = 10


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _condition_shard(
    model_dir: Path,
    *,
    kind: str,
    key: str,
    identity: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    root = model_dir / "shards"
    stem = f"{kind}.{_slug(key)}"
    marker = root / f"{stem}.complete.json"
    rows_path = root / f"{stem}.rows.parquet"
    trace_path = root / f"{stem}.trace.parquet"
    if not (marker.exists() and rows_path.exists() and trace_path.exists()):
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("identity") != identity:
        raise RuntimeError(f"confirmation shard identity mismatch: {stem}")
    return pd.read_parquet(rows_path), pd.read_parquet(trace_path)


def _save_condition_shard(
    model_dir: Path,
    *,
    kind: str,
    key: str,
    identity: str,
    rows: pd.DataFrame,
    trace: pd.DataFrame,
) -> None:
    root = model_dir / "shards"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}.{_slug(key)}"
    save_table(rows, root / f"{stem}.rows.parquet")
    save_table(trace, root / f"{stem}.trace.parquet")
    atomic_write_json(
        root / f"{stem}.complete.json",
        {"identity": identity, "n_rows": len(rows), "n_trace_rows": len(trace)},
    )


def _trace_native(adapter, result: dict[str, dict[str, Any]], ids: list[str], token_ids: list[int]):
    output: dict[int, dict[str, float]] = {layer: {} for layer in TRACE_LAYERS}
    for layer in TRACE_LAYERS:
        vectors = np.stack([result[sid]["captured"][layer] for sid in ids])
        logits = adapter.final_readout_token_logits(vectors, token_ids)
        for row, sid in enumerate(ids):
            output[layer][sid] = _selected_margin(logits[row])
    return output


def _load_development_activations(
    repo_root: Path,
    adapter,
    samples_by_id: dict[str, Any],
    discovery_df: pd.DataFrame,
    *,
    resolved_revision: str | None,
    split_hash: str,
    batch_size: int,
    model_dir: Path,
    model_id: str,
):
    discovery_ids = discovery_df["sample_id"].astype(str).tolist()
    cache = find_identity_matched_e01a_snapshot(
        repo_root,
        model_id=str(model_id),
        resolved_revision=resolved_revision,
        split_hash=split_hash,
        expected_sample_ids=discovery_ids,
        expected_layers=list(TRACE_LAYERS),
    )
    if cache is not None:
        snapshot, source = cache
        return snapshot[0], str(source.relative_to(repo_root))
    development = discovery_df[discovery_df["split"].isin(["train", "validation"])]
    ids = development["sample_id"].astype(str).tolist()
    activations, indices, sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=TRACE_LAYERS,
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    target = model_dir / "development_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    save_activation_snapshot(
        target, activations, sample_ids=ids, token_indices=indices, token_sites=sites
    )
    return activations, str(target.relative_to(repo_root))


def _load_confirmation_activations(
    adapter, samples_by_id, ids, *, batch_size: int, model_dir: Path
):
    cached = load_activation_snapshot(
        model_dir, expected_sample_ids=ids, expected_layers=TRACE_LAYERS
    )
    if cached is not None:
        return cached
    activations, indices, sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=TRACE_LAYERS,
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    save_activation_snapshot(
        model_dir,
        activations,
        sample_ids=ids,
        token_indices=indices,
        token_sites=sites,
    )
    return activations, indices, sites


def _scalar_rows_for_spec(
    *,
    cfg,
    adapter,
    samples_by_id,
    ids,
    bases,
    clean,
    clean_margins,
    clean_trace_q,
    clean_trace_m,
    token_indices,
    token_sites,
    probe_dirs,
    targets,
    spec,
    deltas,
    output_token_ids,
    layer_references,
):
    condition = str(spec["condition"])
    results = (
        clean
        if condition == "no_op"
        else run_intervention_batches(
            adapter,
            samples_by_id,
            ids,
            layer=LAYER,
            token_indices=token_indices,
            deltas_by_id=deltas,
            output_token_ids=output_token_ids,
            capture_layers=TRACE_LAYERS,
            batch_size=int(cfg.runtime.batch_size),
        )
    )
    trace_after = _trace_native(adapter, results, ids, output_token_ids)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for sid in ids:
        sample = samples_by_id[sid]
        gold = int(sample.target_label)
        target_label = None if condition == "source_free_grid" else 1 - gold
        before = clean_margins[sid]
        after = _selected_margin(results[sid]["selected_logits"])
        orientation = np.nan if target_label is None else (1.0 if target_label == 1 else -1.0)
        oriented_before = (
            np.nan if target_label is None else margin_toward_label(before, target_label)
        )
        oriented_after = (
            np.nan if target_label is None else margin_toward_label(after, target_label)
        )
        q_base = coordinate_value(bases[sid], probe_dirs[LAYER])
        captured = np.asarray(results[sid]["captured"][LAYER], dtype=np.float64)
        q_after = coordinate_value(captured, probe_dirs[LAYER])
        target = targets[sid]
        outcome = (
            {"counterfactual_flip": 0, "expected_label_after": 0}
            if target_label is None
            else counterfactual_outcome(_prediction(before), _prediction(after), target_label)
        )
        delta = np.asarray(deltas[sid], dtype=np.float64)
        site = token_sites[sid]
        rows.append(
            {
                "model_id": str(cfg.model.id),
                "resolved_revision": adapter.resolved_revisions().get("model_sha"),
                "base_sample_id": sid,
                "pair_id": str(sample.pair_id),
                "relation_family": str(sample.metadata["relation"]),
                "gold_label": gold,
                "target_label": target_label,
                "condition": condition,
                "target_name": spec["target_name"],
                "direction_seed": spec.get("direction_seed"),
                "q_base": q_base,
                "q_target": target,
                "q_after": q_after,
                "delta_q": q_after - q_base,
                "base_yes_no_margin": before,
                "intervened_yes_no_margin": after,
                "delta_yes_no_margin": after - before,
                "margin_toward_target_before": oriented_before,
                "margin_toward_target_after": oriented_after,
                "delta_margin_toward_target": oriented_after - oriented_before,
                "prediction_before": _prediction(before),
                "prediction_after": _prediction(after),
                "actual_target_flip": outcome["counterfactual_flip"],
                "expected_target_after": outcome["expected_label_after"],
                "activation_norm": float(np.linalg.norm(bases[sid])),
                "delta_norm": float(np.linalg.norm(delta)),
                "delta_norm_ratio": float(np.linalg.norm(delta))
                / max(float(np.linalg.norm(bases[sid])), 1e-12),
                "target_projection_error": np.nan
                if target is None
                else abs(q_after - float(target)),
                "site": SITE,
                "layer": LAYER,
                "native_module_name": adapter.resolve_site(SITE, LAYER).native_module_name,
                "token_selector": SELECTOR,
                "token_index": int(token_indices[sid]),
                "prompt_sequence_length": int(site["sequence_length"]),
                "token_id": int(site["token_id"]),
                "token_text": str(site["token_text"]),
                "raw_text": str(sample.prompt),
                "chat_template_used": bool(site["chat_template_used"]),
                "confirmation_accessed": True,
            }
        )
        for layer in TRACE_LAYERS:
            vector = np.asarray(results[sid]["captured"][layer])
            q = coordinate_value(vector, probe_dirs[layer])
            margin = trace_after[layer][sid]
            traces.append(
                {
                    "artifact_family": "scalar",
                    "model_id": str(cfg.model.id),
                    "base_sample_id": sid,
                    "pair_id": str(sample.pair_id),
                    "relation_family": str(sample.metadata["relation"]),
                    "target_label": target_label,
                    "condition": condition,
                    "target_name": spec["target_name"],
                    "direction_seed": spec.get("direction_seed"),
                    "lambda_context": np.nan,
                    "arm": condition,
                    "trace_layer": layer,
                    "clean_truth_coordinate": clean_trace_q[layer][sid],
                    "intervened_truth_coordinate": q,
                    "delta_truth_coordinate": q - clean_trace_q[layer][sid],
                    "oriented_delta_q_z": orientation
                    * (q - clean_trace_q[layer][sid])
                    / float(layer_references[str(layer)]["sigma_q_validation"]),
                    "clean_native_yes_no_margin": clean_trace_m[layer][sid],
                    "intervened_native_yes_no_margin": margin,
                    "delta_native_yes_no_margin": margin - clean_trace_m[layer][sid],
                    "oriented_delta_native_margin_z": orientation
                    * (margin - clean_trace_m[layer][sid])
                    / float(layer_references[str(layer)]["sigma_margin_validation"]),
                    "confirmation_accessed": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(traces)


def _run_model(
    *,
    cfg,
    repo_root: Path,
    campaign_dir: Path,
    samples,
    full_df: pd.DataFrame,
    confirmation_df: pd.DataFrame,
    plans,
    plan_digest: str,
    protocol_identity: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_dir = campaign_dir / _slug(str(cfg.model.id))
    model_dir.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(cfg)
    revisions = adapter.resolved_revisions()
    candidates = list(cfg.behavior.candidates_primary)
    candidate_lists = candidate_token_id_lists(adapter, candidates)
    if candidates != [" Yes", " No"] or any(len(item) != 1 for item in candidate_lists):
        raise RuntimeError("confirmation requires frozen single-token Yes/No candidates")
    output_token_ids = [int(item[0]) for item in candidate_lists]
    validate_model_identity(
        str(cfg.model.id),
        resolved_revision=revisions.get("model_sha"),
        tokenizer_revision=revisions.get("tokenizer_sha"),
        candidate_token_ids=output_token_ids,
    )
    if adapter.num_layers != 28 or any(layer >= adapter.num_layers for layer in TRACE_LAYERS):
        raise RuntimeError("frozen layer identity is invalid for confirmation model")

    discovery_df = discovery_view(full_df).reset_index(drop=True)
    split_map = dict(zip(discovery_df["sample_id"].astype(str), discovery_df["split"].astype(str)))
    discovery_hash = dataset_split_hash(split_map)
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    development_activations, development_source = _load_development_activations(
        repo_root,
        adapter,
        samples_by_id,
        discovery_df,
        resolved_revision=revisions.get("model_sha"),
        split_hash=discovery_hash,
        batch_size=int(cfg.runtime.batch_size),
        model_dir=model_dir,
        model_id=str(cfg.model.id),
    )
    development_df = discovery_df[discovery_df["split"].isin(["train", "validation"])]
    probe_dirs, _fits, probe_digest = fit_locked_probe_layers(
        development_activations,
        development_df,
        layers=TRACE_LAYERS,
        c_grid=cfg.probe.C_grid,
        seed=int(cfg.reproducibility.probe_seed),
    )
    expected_probe = str(FROZEN_MODELS[str(cfg.model.id)]["probe_digest"])
    if probe_digest != expected_probe:
        raise RuntimeError(f"frozen probe digest mismatch for {cfg.model.id}")
    frozen_targets = load_frozen_targets(repo_root, str(cfg.model.id))

    ids = confirmation_df["sample_id"].astype(str).tolist()
    activations, token_indices, token_sites = _load_confirmation_activations(
        adapter, samples_by_id, ids, batch_size=int(cfg.runtime.batch_size), model_dir=model_dir
    )
    if any(token_indices[sid] != int(token_sites[sid]["sequence_length"]) - 1 for sid in ids):
        raise RuntimeError("confirmation last_prompt index is not the final non-padding token")
    zeros = {sid: np.zeros(adapter.hidden_size, dtype=np.float64) for sid in ids}
    clean = run_intervention_batches(
        adapter,
        samples_by_id,
        ids,
        layer=LAYER,
        token_indices=token_indices,
        deltas_by_id=zeros,
        output_token_ids=output_token_ids,
        capture_layers=TRACE_LAYERS,
        batch_size=int(cfg.runtime.batch_size),
    )
    unhooked = run_unintervened_batches(
        adapter,
        samples_by_id,
        ids,
        token_indices=token_indices,
        output_token_ids=output_token_ids,
        batch_size=int(cfg.runtime.batch_size),
    )
    no_op_deviation = max(
        float(np.max(np.abs(clean[sid]["selected_logits"] - unhooked[sid]))) for sid in ids
    )
    if no_op_deviation > 1e-6:
        raise RuntimeError(f"confirmation no-op fidelity failed: {no_op_deviation}")
    bases = intervention_base_activations(clean, ids, layer=LAYER)
    clean_margins = {sid: _selected_margin(clean[sid]["selected_logits"]) for sid in ids}
    clean_trace_m = _trace_native(adapter, clean, ids, output_token_ids)
    clean_trace_q = {
        layer: {
            sid: coordinate_value(clean[sid]["captured"][layer], probe_dirs[layer]) for sid in ids
        }
        for layer in TRACE_LAYERS
    }
    u = probe_dirs[LAYER]
    q_targets = {
        sid: float(
            frozen_targets["q1_star"]
            if int(samples_by_id[sid].target_label) == 0
            else frozen_targets["q0_star"]
        )
        for sid in ids
    }
    semantic = {sid: source_free_setpoint_delta(bases[sid], u, q_targets[sid]) for sid in ids}
    random_dirs = {
        int(cfg.reproducibility.control_seed) + 10_000 + index: random_unit_direction(
            adapter.hidden_size, int(cfg.reproducibility.control_seed) + 10_000 + index
        )
        for index in range(N_RANDOM)
    }
    orthogonal_dirs = {
        int(cfg.reproducibility.control_seed) + 20_000 + index: random_unit_direction(
            adapter.hidden_size,
            int(cfg.reproducibility.control_seed) + 20_000 + index,
            orthogonal_to=u,
        )
        for index in range(N_RANDOM)
    }
    scalar_specs: list[dict[str, Any]] = [
        {"key": "no_op", "condition": "no_op", "target_name": "q_base"},
        {
            "key": "opposite",
            "condition": "source_free_opposite_class_median",
            "target_name": "opposite_class_median",
        },
    ]
    scalar_specs.extend(
        {
            "key": f"grid_{name}",
            "condition": "source_free_grid",
            "target_name": name,
            "q_target": value,
        }
        for name, value in frozen_targets["grid"].items()
    )
    scalar_specs.extend(
        {
            "key": f"random_{seed}",
            "condition": "random_direction",
            "target_name": "opposite_class_median_norm",
            "direction_seed": seed,
        }
        for seed in random_dirs
    )
    scalar_specs.extend(
        {
            "key": f"orthogonal_{seed}",
            "condition": "orthogonal_random",
            "target_name": "opposite_class_median_norm",
            "direction_seed": seed,
        }
        for seed in orthogonal_dirs
    )
    scalar_frames: list[pd.DataFrame] = []
    scalar_traces: list[pd.DataFrame] = []
    shard_base_identity = {
        **protocol_identity,
        "version": CONFIRMATION_VERSION,
        "model": str(cfg.model.id),
        "revision": revisions.get("model_sha"),
        "probe": probe_digest,
        "target_sha": FROZEN_MODELS[str(cfg.model.id)]["target_sha256"],
        "base_ids": ids,
    }
    for spec in scalar_specs:
        identity = _digest({**shard_base_identity, "kind": "scalar", "spec": spec})
        cached = _condition_shard(model_dir, kind="scalar", key=spec["key"], identity=identity)
        if cached is not None:
            rows, traces = cached
        else:
            condition = spec["condition"]
            if condition == "no_op":
                targets = {sid: coordinate_value(bases[sid], u) for sid in ids}
                deltas = zeros
            elif condition == "source_free_opposite_class_median":
                targets = q_targets
                deltas = semantic
            elif condition == "source_free_grid":
                targets = {sid: float(spec["q_target"]) for sid in ids}
                deltas = {
                    sid: source_free_setpoint_delta(bases[sid], u, targets[sid]) for sid in ids
                }
            elif condition == "random_direction":
                targets = {sid: None for sid in ids}
                direction = random_dirs[int(spec["direction_seed"])]
                deltas = {
                    sid: norm_matched_direction_delta(semantic[sid], direction) for sid in ids
                }
            elif condition == "orthogonal_random":
                targets = {sid: None for sid in ids}
                direction = orthogonal_dirs[int(spec["direction_seed"])]
                deltas = {
                    sid: norm_matched_direction_delta(semantic[sid], direction) for sid in ids
                }
            else:
                raise RuntimeError(f"unknown confirmation scalar condition {condition}")
            rows, traces = _scalar_rows_for_spec(
                cfg=cfg,
                adapter=adapter,
                samples_by_id=samples_by_id,
                ids=ids,
                bases=bases,
                clean=clean,
                clean_margins=clean_margins,
                clean_trace_q=clean_trace_q,
                clean_trace_m=clean_trace_m,
                token_indices=token_indices,
                token_sites=token_sites,
                probe_dirs=probe_dirs,
                targets=targets,
                spec=spec,
                deltas=deltas,
                output_token_ids=output_token_ids,
                layer_references=frozen_targets["layer_references"],
            )
            _save_condition_shard(
                model_dir,
                kind="scalar",
                key=spec["key"],
                identity=identity,
                rows=rows,
                trace=traces,
            )
        scalar_frames.append(rows)
        scalar_traces.append(traces)
    scalar_df = pd.concat(scalar_frames, ignore_index=True)
    semantic_rows = scalar_df[scalar_df["condition"] == "source_free_opposite_class_median"]
    semantic_result_margins = semantic_rows.set_index("base_sample_id")[
        "intervened_yes_no_margin"
    ].to_dict()

    source_attributes = {
        "matched_orthogonal": "matched_source_id",
        "same_family_shuffled_orthogonal": "same_family_source_id",
        "different_family_shuffled_orthogonal": "different_family_source_id",
        "same_label_orthogonal": "same_label_source_id",
    }
    fallback_norm = float(
        frozen_targets["matched_context_norm_fallback"][
            "median_nondegenerate_matched_orthogonal_norm"
        ]
    )
    context_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    context_plan_rows: list[dict[str, Any]] = []
    random_context_seeds = [
        int(cfg.reproducibility.control_seed) + 40_000 + index for index in range(N_RANDOM)
    ]
    for sid in ids:
        plan = plans[sid]
        matched_raw = orthogonal_component(
            activations[LAYER][plan.matched_source_id], bases[sid], u
        )
        reference_norm, reference_source, reference_fallback = resolve_context_reference_norm(
            float(np.linalg.norm(matched_raw)), fallback_norm, epsilon=CONTEXT_EPSILON
        )
        for condition, attribute in source_attributes.items():
            source_id = str(getattr(plan, attribute))
            raw = orthogonal_component(activations[LAYER][source_id], bases[sid], u)
            fallback_seed = int.from_bytes(
                hashlib.sha256(f"{sid}|{condition}".encode()).digest()[:4], "big"
            )
            context, diagnostics = standardize_orthogonal_context(
                raw,
                u,
                reference_norm,
                epsilon=CONTEXT_EPSILON,
                fallback_direction=random_unit_direction(
                    adapter.hidden_size, fallback_seed, orthogonal_to=u
                ),
            )
            context_vectors[(sid, condition, -1)] = context
            source = samples_by_id[source_id]
            context_plan_rows.append(
                {
                    "base_sample_id": sid,
                    "condition": condition,
                    "context_source_id": source_id,
                    "context_source_pair_id": str(source.pair_id),
                    "context_source_relation_family": str(source.metadata["relation"]),
                    "context_source_label": int(source.target_label),
                    "context_selection_seed": int(plan.selection_seed),
                    "direction_seed": np.nan,
                    "reference_norm": reference_norm,
                    "reference_norm_source": reference_source,
                    "reference_fallback_used": reference_fallback,
                    **diagnostics,
                }
            )
        for direction_seed in random_context_seeds:
            context, diagnostics = standardize_orthogonal_context(
                random_unit_direction(adapter.hidden_size, direction_seed, orthogonal_to=u),
                u,
                reference_norm,
                epsilon=CONTEXT_EPSILON,
            )
            context_vectors[(sid, "random_orthogonal", direction_seed)] = context
            context_plan_rows.append(
                {
                    "base_sample_id": sid,
                    "condition": "random_orthogonal",
                    "context_source_id": None,
                    "context_source_pair_id": None,
                    "context_source_relation_family": None,
                    "context_source_label": None,
                    "context_selection_seed": int(plan.selection_seed),
                    "direction_seed": direction_seed,
                    "reference_norm": reference_norm,
                    "reference_norm_source": reference_source,
                    "reference_fallback_used": reference_fallback,
                    **diagnostics,
                }
            )
    context_plan = pd.DataFrame(context_plan_rows)
    save_table(context_plan, model_dir / "context_vector_plan.parquet")
    model_plan_digest = _digest(context_plan.to_dict(orient="records"))

    factorial_frames: list[pd.DataFrame] = []
    factorial_traces: list[pd.DataFrame] = []
    factorial_specs: list[dict[str, Any]] = []
    for strength in CONTEXT_STRENGTHS:
        factorial_specs.extend(
            {
                "key": f"{condition}_{strength:g}",
                "condition": condition,
                "lambda_context": strength,
                "direction_seed": None,
            }
            for condition in source_attributes
        )
        factorial_specs.extend(
            {
                "key": f"random_{seed}_{strength:g}",
                "condition": "random_orthogonal",
                "lambda_context": strength,
                "direction_seed": seed,
            }
            for seed in random_context_seeds
        )
    for spec in factorial_specs:
        identity = _digest(
            {
                **shard_base_identity,
                "kind": "factorial",
                "spec": spec,
                "semantic_source_plan": plan_digest,
                "model_context_plan": model_plan_digest,
            }
        )
        cached = _condition_shard(model_dir, kind="factorial", key=spec["key"], identity=identity)
        if cached is not None:
            rows, traces = cached
        else:
            condition = str(spec["condition"])
            strength = float(spec["lambda_context"])
            seed_key = -1 if spec["direction_seed"] is None else int(spec["direction_seed"])
            contexts = {sid: context_vectors[(sid, condition, seed_key)] for sid in ids}
            y01_deltas = {sid: strength * contexts[sid] for sid in ids}
            y11_deltas = {sid: semantic[sid] + y01_deltas[sid] for sid in ids}
            y01 = run_intervention_batches(
                adapter,
                samples_by_id,
                ids,
                layer=LAYER,
                token_indices=token_indices,
                deltas_by_id=y01_deltas,
                output_token_ids=output_token_ids,
                capture_layers=TRACE_LAYERS,
                batch_size=int(cfg.runtime.batch_size),
            )
            y11 = run_intervention_batches(
                adapter,
                samples_by_id,
                ids,
                layer=LAYER,
                token_indices=token_indices,
                deltas_by_id=y11_deltas,
                output_token_ids=output_token_ids,
                capture_layers=TRACE_LAYERS,
                batch_size=int(cfg.runtime.batch_size),
            )
            y01_trace_m = _trace_native(adapter, y01, ids, output_token_ids)
            y11_trace_m = _trace_native(adapter, y11, ids, output_token_ids)
            rows_out: list[dict[str, Any]] = []
            traces_out: list[dict[str, Any]] = []
            plan_lookup = context_plan[
                (context_plan["condition"] == condition)
                & (
                    context_plan["direction_seed"].isna()
                    if seed_key == -1
                    else context_plan["direction_seed"].fillna(-1).astype(int).eq(seed_key)
                )
            ].set_index("base_sample_id")
            for sid in ids:
                sample = samples_by_id[sid]
                target_label = 1 - int(sample.target_label)
                orientation = 1.0 if target_label == 1 else -1.0
                y00 = margin_toward_label(clean_margins[sid], target_label)
                y10 = margin_toward_label(semantic_result_margins[sid], target_label)
                y01_value = margin_toward_label(
                    _selected_margin(y01[sid]["selected_logits"]), target_label
                )
                y11_value = margin_toward_label(
                    _selected_margin(y11[sid]["selected_logits"]), target_label
                )
                estimands = factorial_estimands(y00, y10, y01_value, y11_value)
                q01 = coordinate_value(y01[sid]["captured"][LAYER], u)
                q11 = coordinate_value(y11[sid]["captured"][LAYER], u)
                plan_row = plan_lookup.loc[sid]
                rows_out.append(
                    {
                        "model_id": str(cfg.model.id),
                        "resolved_revision": revisions.get("model_sha"),
                        "base_sample_id": sid,
                        "pair_id": str(sample.pair_id),
                        "relation_family": str(sample.metadata["relation"]),
                        "gold_label": int(sample.target_label),
                        "target_label": target_label,
                        "condition": condition,
                        "lambda_context": strength,
                        "direction_seed": spec["direction_seed"],
                        "context_source_id": plan_row["context_source_id"],
                        "context_source_pair_id": plan_row["context_source_pair_id"],
                        "context_source_relation_family": plan_row[
                            "context_source_relation_family"
                        ],
                        "context_source_label": plan_row["context_source_label"],
                        "context_selection_seed": int(plan_row["context_selection_seed"]),
                        "context_reference_norm": float(plan_row["reference_norm"]),
                        "context_applied_norm": float(plan_row["context_applied_norm"]),
                        "context_dot_truth_direction": float(
                            plan_row["context_dot_truth_direction"]
                        ),
                        "context_norm_relative_error": float(
                            plan_row["context_norm_relative_error"]
                        ),
                        "q_base": coordinate_value(bases[sid], u),
                        "q_target": q_targets[sid],
                        "q_after_context_only": q01,
                        "q_after_setpoint_context": q11,
                        "context_only_q_preservation_error": abs(
                            q01 - coordinate_value(bases[sid], u)
                        ),
                        "setpoint_projection_error": abs(q11 - q_targets[sid]),
                        "Y00": y00,
                        "Y10": y10,
                        "Y01": y01_value,
                        "Y11": y11_value,
                        "Q0": float(estimands["Q0"]),
                        "A_context": float(estimands["A_context"]),
                        "Q_context": float(estimands["Q_context"]),
                        "G_interaction": float(estimands["G_interaction"]),
                        "confirmation_accessed": True,
                    }
                )
                for layer in TRACE_LAYERS:
                    q00 = clean_trace_q[layer][sid]
                    q10 = float(
                        scalar_traces[1][
                            (scalar_traces[1]["base_sample_id"] == sid)
                            & (scalar_traces[1]["trace_layer"] == layer)
                        ]["intervened_truth_coordinate"].iloc[0]
                    )
                    q01_layer = coordinate_value(y01[sid]["captured"][layer], probe_dirs[layer])
                    q11_layer = coordinate_value(y11[sid]["captured"][layer], probe_dirs[layer])
                    m00 = clean_trace_m[layer][sid]
                    semantic_trace = scalar_traces[1][
                        (scalar_traces[1]["base_sample_id"] == sid)
                        & (scalar_traces[1]["trace_layer"] == layer)
                    ]
                    m10 = float(semantic_trace["intervened_native_yes_no_margin"].iloc[0])
                    m01 = y01_trace_m[layer][sid]
                    m11 = y11_trace_m[layer][sid]
                    q_est = factorial_estimands(q00, q10, q01_layer, q11_layer)
                    m_est = factorial_estimands(m00, m10, m01, m11)
                    ref = frozen_targets["layer_references"][str(layer)]
                    traces_out.append(
                        {
                            "artifact_family": "factorial",
                            "model_id": str(cfg.model.id),
                            "base_sample_id": sid,
                            "pair_id": str(sample.pair_id),
                            "relation_family": str(sample.metadata["relation"]),
                            "target_label": target_label,
                            "condition": condition,
                            "target_name": "opposite_class_median",
                            "direction_seed": spec["direction_seed"],
                            "lambda_context": strength,
                            "arm": "factorial_decomposition",
                            "trace_layer": layer,
                            "clean_truth_coordinate": q00,
                            "intervened_truth_coordinate": q11_layer,
                            "delta_truth_coordinate": q11_layer - q00,
                            "oriented_delta_q_z": orientation
                            * (q11_layer - q00)
                            / float(ref["sigma_q_validation"]),
                            "clean_native_yes_no_margin": m00,
                            "intervened_native_yes_no_margin": m11,
                            "delta_native_yes_no_margin": m11 - m00,
                            "oriented_delta_native_margin_z": orientation
                            * (m11 - m00)
                            / float(ref["sigma_margin_validation"]),
                            "A_q_z": orientation
                            * float(q_est["A_context"])
                            / float(ref["sigma_q_validation"]),
                            "G_q_z": orientation
                            * float(q_est["G_interaction"])
                            / float(ref["sigma_q_validation"]),
                            "A_margin_z": orientation
                            * float(m_est["A_context"])
                            / float(ref["sigma_margin_validation"]),
                            "G_margin_z": orientation
                            * float(m_est["G_interaction"])
                            / float(ref["sigma_margin_validation"]),
                            "confirmation_accessed": True,
                        }
                    )
            rows, traces = pd.DataFrame(rows_out), pd.DataFrame(traces_out)
            _save_condition_shard(
                model_dir,
                kind="factorial",
                key=spec["key"],
                identity=identity,
                rows=rows,
                trace=traces,
            )
        factorial_frames.append(rows)
        factorial_traces.append(traces)
    factorial_df = pd.concat(factorial_frames, ignore_index=True)
    trace_df = pd.concat([*scalar_traces, *factorial_traces], ignore_index=True)
    if (
        not np.isfinite(
            scalar_df[scalar_df["condition"] != "source_free_grid"][
                ["base_yes_no_margin", "intervened_yes_no_margin", "delta_norm"]
            ].to_numpy(float)
        ).all()
        or not np.isfinite(
            factorial_df[["Y00", "Y10", "Y01", "Y11", "A_context", "G_interaction"]].to_numpy(float)
        ).all()
    ):
        raise RuntimeError("confirmation contains non-finite primary evidence")
    max_context_dot = float(factorial_df["context_dot_truth_direction"].abs().max())
    max_context_norm = float(factorial_df["context_norm_relative_error"].max())
    max_q01 = float(factorial_df["context_only_q_preservation_error"].max())
    max_q11 = float(factorial_df["setpoint_projection_error"].max())
    sigma_q = float(frozen_targets["sigma_q_validation"])
    if (
        max_context_dot > 1e-10
        or max_context_norm > 1e-10
        or max(max_q01, max_q11) / sigma_q > 0.05
    ):
        raise RuntimeError("confirmation intervention fidelity gate failed")
    after_unhooked = run_unintervened_batches(
        adapter,
        samples_by_id,
        ids,
        token_indices=token_indices,
        output_token_ids=output_token_ids,
        batch_size=int(cfg.runtime.batch_size),
    )
    hook_leakage = max(float(np.max(np.abs(after_unhooked[sid] - unhooked[sid]))) for sid in ids)
    if hook_leakage > 1e-6:
        raise RuntimeError("confirmation hook leakage gate failed")
    save_table(scalar_df, model_dir / "scalar_rows.parquet")
    save_table(factorial_df, model_dir / "factorial_rows.parquet")
    save_table(trace_df, model_dir / "trace_rows.parquet")
    gates = {
        "no_op_max_logit_deviation": no_op_deviation,
        "hook_leakage_max_logit_deviation": hook_leakage,
        "max_context_dot_u": max_context_dot,
        "max_context_norm_relative_mismatch": max_context_norm,
        "max_context_only_q_preservation_error": max_q01,
        "max_setpoint_projection_error": max_q11,
        "finite_values": True,
        "trace_complete": len(trace_df) > 0,
        "candidate_token_ids": output_token_ids,
        "probe_digest": probe_digest,
        "semantic_source_plan_digest": plan_digest,
        "model_context_plan_digest": model_plan_digest,
        "development_activation_source": development_source,
    }
    save_json(gates, model_dir / "integrity_gates.json")
    del adapter
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError) as exc:
        logger.warning("CUDA cache cleanup failed: %s", exc)
    return scalar_df, factorial_df, trace_df, gates


def _write_summary(campaign_dir: Path, primary: pd.DataFrame, classification: str) -> None:
    lines = [
        "# E01 Actionability Confirmation",
        "",
        f"Classification: **{classification} confirmation**",
        "",
        "This report applies the preregistered H1-H4 family without post-access retuning.",
        "",
        "| Hypothesis | Estimate | 95% CI | Raw p | Holm p | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.hypothesis} | {row.estimate:.6g} | [{row.ci_low:.6g}, {row.ci_high:.6g}] | {row.raw_p:.6g} | {row.holm_p:.6g} | {row.verdict} |"
        )
    lines.extend(
        [
            "",
            "Secondary analyses remain explicitly secondary. The 0.6B interaction is reported as an estimate and interval; failure to detect it is not equivalence to zero.",
        ]
    )
    (campaign_dir / "CONFIRMATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_confirmation(*, protocol_commit: str) -> Path:
    """Run both frozen checkpoints in one indivisible confirmation campaign."""
    repo_root = Path(__file__).resolve().parents[3]
    protocol_identity = validate_protocol_lock(repo_root, protocol_commit)
    experiment_path = repo_root / "configs/experiments/E01_confirmation.yaml"
    configs = []
    provenances = []
    for identity in FROZEN_MODELS.values():
        cfg, provenance = resolve_config(
            base_path=repo_root / "configs/base.yaml",
            model_path=repo_root / str(identity["config"]),
            experiment_path=experiment_path,
            overrides=(),
        )
        if cfg.experiment.id != "E01_CONFIRMATION" or cfg.experiment.mode != "confirmation":
            raise RuntimeError("dedicated confirmation config identity mismatch")
        configs.append(cfg)
        provenances.append(provenance)
    campaign_id = f"CONFIRMATION_{protocol_identity['protocol_sha256'][:12]}"
    confirmation_root = repo_root / "runs" / "CONFIRMATION"
    campaign_dir = confirmation_root / campaign_id
    if campaign_dir.exists():
        status = StatusFile.load(campaign_dir)
        if status is not None and status.is_complete():
            raise RuntimeError("the authorized confirmation campaign is already complete")
    else:
        campaign_dir.mkdir(parents=True)
    status = StatusFile.load(campaign_dir) or StatusFile.create(
        campaign_dir, campaign_id, "E01_CONFIRMATION"
    )
    environment = environment_manifest()
    git_commit = str(project_git_state().get("sha"))
    access_record = open_access_record(
        confirmation_root,
        campaign_id=campaign_id,
        git_commit=git_commit,
        protocol_identity=protocol_identity,
        environment=environment,
    )
    # This is intentionally the first operation that materializes the holdout.
    samples, full_df = _generate_dataset(configs[0])
    confirmation_df = route_confirmation_rows(full_df)
    ids = confirmation_df["sample_id"].astype(str).tolist()
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    plans = build_confirmation_source_plans(
        confirmation_df,
        samples_by_id,
        seed=int(configs[0].reproducibility.control_seed),
    )
    plan_frame = source_plan_frame(plans)
    plan_sha = source_plan_digest(plan_frame)
    save_table(plan_frame, campaign_dir / "source_context_plan.parquet")
    split_hash = dataset_split_hash({sid: "confirmation" for sid in ids})
    manifest = {
        "version": CONFIRMATION_VERSION,
        "campaign_id": campaign_id,
        "start_time": datetime.now(UTC).isoformat(),
        "finish_time": None,
        "environment": environment,
        "project_git_sha": git_commit,
        "protocol": protocol_identity,
        "access_record": access_record,
        "config_hashes": [config_hash(cfg) for cfg in configs],
        "config_provenance": provenances,
        "models": FROZEN_MODELS,
        "dataset": {
            "split": "confirmation",
            "split_hash": split_hash,
            "n_examples": len(ids),
            "n_pairs": int(confirmation_df["pair_id"].nunique()),
            "prompt_hash_sample": prompt_hash(str(confirmation_df.iloc[0]["prompt"])),
        },
        "source_plan_sha256": plan_sha,
        "candidate_token_ids": [7414, 2308],
        "site": SITE,
        "layer": LAYER,
        "token_selector": SELECTOR,
        "trace_layers": list(TRACE_LAYERS),
        "lambdas": list(CONTEXT_STRENGTHS),
        "random_direction_seeds": list(range(20270830, 20270840)),
        "orthogonal_direction_seeds": list(range(20280830, 20280840)),
        "random_context_seeds": list(range(20300830, 20300840)),
    }
    atomic_write_json(campaign_dir / "manifest.json", manifest)
    try:
        scalar_by_model: dict[str, pd.DataFrame] = {}
        factorial_by_model: dict[str, pd.DataFrame] = {}
        traces: list[pd.DataFrame] = []
        gates: dict[str, Any] = {}
        for index, cfg in enumerate(configs):
            status.update(
                message=f"running frozen confirmation model {cfg.model.id}",
                progress={"model_index": index + 1, "models_total": 2},
            )
            scalar, factorial, trace, model_gates = _run_model(
                cfg=cfg,
                repo_root=repo_root,
                campaign_dir=campaign_dir,
                samples=samples,
                full_df=full_df,
                confirmation_df=confirmation_df,
                plans=plans,
                plan_digest=plan_sha,
                protocol_identity=protocol_identity,
            )
            scalar_by_model[str(cfg.model.id)] = scalar
            factorial_by_model[str(cfg.model.id)] = factorial
            traces.append(trace)
            gates[str(cfg.model.id)] = model_gates
        scalar_all = pd.concat(scalar_by_model.values(), ignore_index=True)
        factorial_all = pd.concat(factorial_by_model.values(), ignore_index=True)
        trace_all = pd.concat(traces, ignore_index=True)
        save_table(scalar_all, campaign_dir / "scalar_rows.parquet")
        save_table(factorial_all, campaign_dir / "factorial_rows.parquet")
        save_table(trace_all, campaign_dir / "trace_rows.parquet")
        primary, details = evaluate_primary_hypotheses(scalar_by_model, factorial_by_model)
        classification = confirmation_classification(primary)
        save_table(primary, campaign_dir / "primary_hypotheses.parquet")
        secondary = pd.concat(
            [
                scalar_all.groupby(
                    ["model_id", "condition", "target_name"], dropna=False, as_index=False
                )
                .agg(
                    mean_effect=("delta_margin_toward_target", "mean"), n=("base_sample_id", "size")
                )
                .assign(metric_family="scalar"),
                factorial_all.groupby(["model_id", "condition", "lambda_context"], as_index=False)
                .agg(
                    mean_A=("A_context", "mean"),
                    mean_G=("G_interaction", "mean"),
                    n=("base_sample_id", "size"),
                )
                .assign(metric_family="factorial"),
            ],
            ignore_index=True,
            sort=False,
        )
        save_table(secondary, campaign_dir / "secondary_metrics.parquet")
        save_json(details, campaign_dir / "primary_hypothesis_components.json")
        save_json(gates, campaign_dir / "integrity_gates.json")
        _write_summary(campaign_dir, primary, classification)
        manifest["finish_time"] = datetime.now(UTC).isoformat()
        manifest["classification"] = classification
        manifest["integrity_gates"] = gates
        atomic_write_json(campaign_dir / "manifest.json", manifest)
        status.complete(f"locked confirmation complete: {classification}")
        return campaign_dir
    except Exception as exc:
        logger.exception("locked confirmation failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        raise
