"""E01B-2: fixed semantic setpoint with orthogonal-context modulation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.splits import build_discovery_label_map, discovery_view
from ..interventions.orthogonal_context import (
    fixed_setpoint_context_edit,
    orthogonal_component,
    resolve_context_reference_norm,
    standardize_orthogonal_context,
)
from ..interventions.setpoint import (
    setpoint_fidelity_tolerances,
    source_free_setpoint_delta,
)
from ..interventions.truth_coordinate import coordinate_value, random_unit_direction
from ..metrics.causal import (
    cluster_bootstrap_mean_ci,
    counterfactual_outcome,
    margin_toward_label,
)
from ..metrics.orthogonal_context import (
    CONTEXT_CONDITIONS,
    aggregate_context_rows,
    aggregate_trace_context,
    attach_context_increments,
    attach_trace_context_increments,
    paired_context_contrast,
)
from ..metrics.setpoint import validation_setpoint_targets
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile, atomic_write_json
from .e00c import _generate_dataset, candidate_token_id_lists
from .e01a import _layer_probe, _prediction, _selected_margin
from .e01a_support import (
    deterministic_subset_pair_ids,
    extract_resid_post_layers,
    intervention_base_activations,
    parse_trace_layers,
    run_intervention_batches,
    run_unintervened_batches,
    save_activation_snapshot,
)
from .e01b2_support import (
    CONTEXT_EPSILON,
    build_context_source_plans,
    e01b2_profile_limits,
    find_frozen_e01b1_run,
    parse_context_strengths,
    validate_e01b2_artifact_shape,
    validation_matched_norm_fallback,
)
from .e01b_support import find_identity_matched_e01a_snapshot, validation_identity
from .extract import load_adapter

logger = logging.getLogger(__name__)

E01B2_VERSION = "e01b2-orthogonal-context-v1"
SITE = "resid_post"
SELECTOR = "last_prompt"


def _allocate_or_resume(
    output_root: Path, run_id: str, *, resume: bool
) -> tuple[Path, bool]:
    root = output_root / "E01B2"
    candidates: list[Path] = []
    canonical = root / run_id
    if canonical.exists():
        candidates.append(canonical)
    reruns = list(root.glob(f"{run_id}-r*"))
    reruns.sort(key=lambda path: int(path.name.rsplit("-r", 1)[1]))
    candidates.extend(reruns)
    if resume:
        for candidate in reversed(candidates):
            status = StatusFile.load(candidate)
            if status is not None and not status.is_complete():
                return candidate, True
    return allocate_run_dir(output_root, "E01B2", run_id), False


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_shard(
    run_dir: Path,
    spec: dict[str, Any],
    *,
    base_ids: list[str],
    plan_digest: str,
    n_traces: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    name = _slug(str(spec["key"]))
    root = run_dir / "shards"
    marker = root / f"{name}.complete.json"
    raw_path = root / f"{name}.intervention.parquet"
    trace_path = root / f"{name}.trace.parquet"
    if not marker.exists() or not raw_path.exists() or not trace_path.exists():
        return None
    expected = _digest(
        {
            "version": E01B2_VERSION,
            "spec": spec,
            "base_ids": base_ids,
            "plan_digest": plan_digest,
        }
    )
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("identity") != expected:
        raise RuntimeError(f"E01B-2 shard identity mismatch: {name}")
    raw = pd.read_parquet(raw_path)
    trace = pd.read_parquet(trace_path)
    if len(raw) != len(base_ids) or len(trace) != len(base_ids) * n_traces:
        raise RuntimeError(f"E01B-2 shard row-count mismatch: {name}")
    return raw, trace


def _save_shard(
    run_dir: Path,
    spec: dict[str, Any],
    *,
    base_ids: list[str],
    plan_digest: str,
    raw: pd.DataFrame,
    trace: pd.DataFrame,
) -> None:
    name = _slug(str(spec["key"]))
    root = run_dir / "shards"
    root.mkdir(parents=True, exist_ok=True)
    save_table(raw, root / f"{name}.intervention.parquet")
    save_table(trace, root / f"{name}.trace.parquet")
    atomic_write_json(
        root / f"{name}.complete.json",
        {
            "identity": _digest(
                {
                    "version": E01B2_VERSION,
                    "spec": spec,
                    "base_ids": base_ids,
                    "plan_digest": plan_digest,
                }
            ),
            "n_intervention_rows": len(raw),
            "n_trace_rows": len(trace),
        },
    )


def _relation_metrics(
    rows: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    block = rows[
        (rows["condition"] != "coordinate_only")
        & np.isclose(rows["lambda_context"].to_numpy(float), 1.0)
    ]
    for index, ((family, condition), family_rows) in enumerate(
        block.groupby(["relation_family", "condition"], sort=True)
    ):
        averaged = family_rows.groupby(
            ["base_sample_id", "pair_id"], as_index=False
        )["context_increment_vs_coordinate_only"].mean()
        ci = cluster_bootstrap_mean_ci(
            averaged["context_increment_vs_coordinate_only"].to_numpy(float),
            averaged["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=seed + index * 23,
        )
        output.append(
            {
                "relation_family": str(family),
                "condition": str(condition),
                "lambda_context": 1.0,
                "n_rows": len(averaged),
                "n_pairs": int(averaged["pair_id"].nunique()),
                "mean_context_increment": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return pd.DataFrame(output)


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# E01B-2 Orthogonal-Context Modulation",
        "",
        f"Status: `{summary['status']}`",
        f"Model: `{summary['model_id']}`",
        f"Profile: `{summary['profile']}`",
        "",
        "Discovery only. Confirmation was not accessed. The validation-only",
        "opposite-class scalar target is identical across every context for a base.",
        "",
        "## Numerical gates",
        "",
    ]
    for key, value in summary["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Context effects", ""])
    for row in summary["aggregate_metrics"]:
        if float(row["comparison_lambda"]) == 1.0:
            lines.append(
                f"- {row['condition']}: effect={row['mean_effect']:.6f}, "
                f"increment={row['mean_context_increment']:.6f}"
            )
    lines.extend(
        [
            "",
            "These smoke/pilot results are exploratory discovery evidence, not",
            "confirmation and not authorization for a full E01B-2 sweep.",
            "",
        ]
    )
    (run_dir / "E01B_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_e01b2(
    base_path=None,
    model_path=None,
    experiment_path=None,
    overrides: tuple[str, ...] = (),
    *,
    layer: int = 17,
    profile: str = "full",
    max_pairs: int | None = None,
    context_strengths: str = "0.5,1.0",
    random_orthogonal_directions: int = 10,
    trace_layers: str = "17,20,23,27",
) -> Path:
    cfg, provenance = resolve_config(
        base_path=base_path,
        model_path=model_path,
        experiment_path=experiment_path,
        overrides=overrides,
    )
    if str(cfg.experiment.mode) != "discovery":
        raise RuntimeError("E01B-2 is discovery-only; confirmation is not authorized")
    if str(cfg.experiment.id) != "E01B2":
        raise ValueError(f"E01B-2 runner received experiment id {cfg.experiment.id!r}")
    if int(layer) != 17:
        raise ValueError("E01B-2 intervention layer is frozen at 17")
    lambdas = parse_context_strengths(context_strengths)
    pair_cap, n_random = e01b2_profile_limits(
        profile, max_pairs, random_orthogonal_directions
    )

    samples, full_df = _generate_dataset(cfg)
    discovery_df = discovery_view(full_df).reset_index(drop=True)
    validation_meta = validation_identity(discovery_df)
    label_map = build_discovery_label_map(discovery_df)
    discovery_ids = discovery_df["sample_id"].astype(str).tolist()
    discovery_id_set = set(discovery_ids)
    samples_by_id = {
        str(sample.sample_id): sample
        for sample in samples
        if str(sample.sample_id) in discovery_id_set
    }
    split_of = dict(
        zip(discovery_df["sample_id"].astype(str), discovery_df["split"].astype(str))
    )

    adapter = load_adapter(cfg)
    if not 0 <= int(layer) < adapter.num_layers:
        raise ValueError(f"intervention layer {layer} out of range")
    traces = parse_trace_layers(
        trace_layers, intervention_layer=int(layer), num_layers=adapter.num_layers
    )
    if traces != [17, 20, 23, 27]:
        raise ValueError("E01B-2 trace layers are frozen at 17,20,23,27")
    needed_layers = sorted({int(layer), *traces})
    candidates = list(cfg.behavior.candidates_primary)
    candidate_lists = candidate_token_id_lists(adapter, candidates)
    if len(candidates) != 2 or any(len(ids) != 1 for ids in candidate_lists):
        raise RuntimeError("E01B-2 requires single-token Yes/No candidates")
    output_token_ids = [int(ids[0]) for ids in candidate_lists]
    if len(set(output_token_ids)) != 2:
        raise RuntimeError("Yes/No candidate token IDs must be distinct")

    revisions = adapter.resolved_revisions()
    resolved_revision = revisions.get("model_sha")
    split_hash = dataset_split_hash(split_of)
    shape = {
        "version": E01B2_VERSION,
        "profile": str(profile).lower(),
        "layer": int(layer),
        "trace_layers": traces,
        "max_pairs": pair_cap,
        "context_strengths": lambdas,
        "random_orthogonal_directions": n_random,
    }
    resolved_hash = hashlib.sha256(
        (config_hash(cfg) + "|" + json.dumps(shape, sort_keys=True)).encode()
    ).hexdigest()
    run_id = make_run_id(
        experiment_id="E01B2",
        config_hash=resolved_hash,
        seed=int(cfg.reproducibility.seed),
        model_revision=resolved_revision or cfg.model.revision or "unpinned",
        dataset_split_hash=split_hash,
    )
    repo_root = Path(__file__).resolve().parents[3]
    run_dir, resumed = _allocate_or_resume(
        repo_root / cfg.project.output_root,
        run_id,
        resume=bool(cfg.runtime.resume),
    )
    save_resolved_config(
        cfg,
        run_dir / "config.resolved.yaml",
        {**provenance, "e01b2_version": E01B2_VERSION, "e01b2_shape": shape},
    )
    manifest = RunManifest(run_dir)
    if resumed:
        status = StatusFile.load(run_dir)
        if status is None or not manifest.path.exists():
            raise RuntimeError("resumable E01B-2 run lacks status or manifest")
        status.update(state="running", message="resuming E01B-2 by context shard")
        manifest.manifest = json.loads(manifest.path.read_text(encoding="utf-8"))
    else:
        status = StatusFile.create(run_dir, run_id, "E01B2")
        manifest.set_start(
            resolved_hash,
            {**provenance, "e01b2_version": E01B2_VERSION, "e01b2_shape": shape},
            cfg.effective_seeds(),
        )
    manifest.update_model_info(
        id=cfg.model.id,
        revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        resolved_revision=resolved_revision,
        tokenizer_revision=revisions.get("tokenizer_sha"),
        num_layers=adapter.num_layers,
        hidden_size=adapter.hidden_size,
        candidate_texts=candidates,
        candidate_token_ids=output_token_ids,
        resolved_native_modules={
            f"{SITE}:{trace_layer}": adapter.resolve_site(
                SITE, trace_layer
            ).native_module_name
            for trace_layer in needed_layers
        },
    )
    manifest.update_dataset_info(
        split_hash=split_hash,
        prompt_hash_sample=prompt_hash(str(discovery_df.iloc[0]["prompt"])),
        validation_target_provenance=validation_meta,
        confirmation_accessed=False,
    )

    try:
        frozen_e01b1_dir = find_frozen_e01b1_run(
            repo_root,
            model_id=str(cfg.model.id),
            resolved_revision=resolved_revision,
        )
        frozen_targets = json.loads(
            (frozen_e01b1_dir / "setpoint_targets.json").read_text(encoding="utf-8")
        )
        if frozen_targets.get("confirmation_accessed") is not False:
            raise RuntimeError("frozen E01B-1 target artifact lacks confirmation lock")

        cache_match = find_identity_matched_e01a_snapshot(
            repo_root,
            model_id=str(cfg.model.id),
            resolved_revision=resolved_revision,
            split_hash=split_hash,
            expected_sample_ids=discovery_ids,
            expected_layers=needed_layers,
        )
        if cache_match is None:
            activations, token_indices, token_sites = extract_resid_post_layers(
                adapter,
                [samples_by_id[sid] for sid in discovery_ids],
                layers=needed_layers,
                token_selector=SELECTOR,
                batch_size=int(cfg.runtime.batch_size),
            )
            save_activation_snapshot(
                run_dir,
                activations,
                sample_ids=discovery_ids,
                token_indices=token_indices,
                token_sites=token_sites,
            )
            activation_source = str(run_dir.relative_to(repo_root))
        else:
            (activations, token_indices, token_sites), source_dir = cache_match
            activation_source = str(source_dir.relative_to(repo_root))
        manifest.manifest.setdefault("cache", {}).update(
            {
                "activation_snapshot_source": activation_source,
                "frozen_e01b1_run": str(frozen_e01b1_dir.relative_to(repo_root)),
            }
        )
        manifest.write()

        probe_dirs: dict[int, np.ndarray] = {}
        probe_metrics: list[dict[str, Any]] = []
        for trace_layer in needed_layers:
            _fit, direction, metrics = _layer_probe(
                trace_layer,
                activations,
                discovery_df,
                label_map,
                c_grid=cfg.probe.C_grid,
                seed=int(cfg.reproducibility.probe_seed),
            )
            probe_dirs[trace_layer] = direction
            probe_metrics.append(metrics)
        save_table(pd.DataFrame(probe_metrics), run_dir / "probe_metrics.parquet")
        u = probe_dirs[int(layer)]

        validation_ids = discovery_df.loc[
            discovery_df["split"].astype(str) == "validation", "sample_id"
        ].astype(str).tolist()
        validation_labels = np.asarray(
            [int(label_map[sid]) for sid in validation_ids], dtype=int
        )
        validation_vectors = np.stack(
            [activations[int(layer)][sid] for sid in validation_ids]
        )
        validation_clean = run_unintervened_batches(
            adapter,
            samples_by_id,
            validation_ids,
            token_indices=token_indices,
            output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        validation_margins = np.asarray(
            [_selected_margin(validation_clean[sid]) for sid in validation_ids]
        )
        reconstructed = validation_setpoint_targets(
            validation_vectors @ u, validation_labels, validation_margins
        )
        target_keys = ("q0_star", "q1_star", "sigma_q_validation", "sigma_margin_validation")
        target_reconstruction_max = max(
            abs(float(reconstructed[key]) - float(frozen_targets[key]))
            for key in target_keys
        )
        if target_reconstruction_max > 1e-10:
            raise RuntimeError("E01B-1 frozen target reconstruction mismatch")
        if (
            frozen_targets.get("provenance", {}).get("sample_ids_sha256")
            != validation_meta["sample_ids_sha256"]
        ):
            raise RuntimeError("E01B-1 validation identity mismatch")

        fallback = validation_matched_norm_fallback(
            discovery_df,
            samples_by_id,
            activations[int(layer)],
            u,
            epsilon=CONTEXT_EPSILON,
        )
        targets_payload = {
            **frozen_targets,
            "frozen_e01b1_run": str(frozen_e01b1_dir.relative_to(repo_root)),
            "target_reconstruction_max_abs_deviation": target_reconstruction_max,
            "matched_context_norm_fallback": fallback,
            "context_strengths": list(lambdas),
            "confirmation_accessed": False,
        }
        save_json(targets_payload, run_dir / "setpoint_targets.json")

        selected_pairs = deterministic_subset_pair_ids(
            discovery_df,
            max_pairs=pair_cap,
            seed=int(cfg.reproducibility.control_seed),
        )
        base_df = discovery_df[
            (discovery_df["split"].astype(str) == "discovery_test")
            & discovery_df["pair_id"].astype(str).isin(selected_pairs)
        ].copy()
        base_ids = base_df["sample_id"].astype(str).tolist()
        if not base_ids:
            raise RuntimeError("E01B-2 selected no discovery examples")
        if any(
            token_indices[sid] != int(token_sites[sid]["sequence_length"]) - 1
            for sid in base_ids
        ):
            raise RuntimeError("last_prompt did not resolve to final non-padding token")

        hidden_dim = int(adapter.hidden_size)
        zeros = {sid: np.zeros(hidden_dim, dtype=np.float64) for sid in base_ids}
        clean = run_intervention_batches(
            adapter,
            samples_by_id,
            base_ids,
            layer=int(layer),
            token_indices=token_indices,
            deltas_by_id=zeros,
            output_token_ids=output_token_ids,
            capture_layers=traces,
            batch_size=int(cfg.runtime.batch_size),
        )
        unhooked = run_unintervened_batches(
            adapter,
            samples_by_id,
            base_ids,
            token_indices=token_indices,
            output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        bases = intervention_base_activations(clean, base_ids, layer=int(layer))
        clean_margins = {
            sid: _selected_margin(clean[sid]["selected_logits"]) for sid in base_ids
        }
        clean_trace_q: dict[int, dict[str, float]] = {tl: {} for tl in traces}
        clean_trace_native: dict[int, dict[str, float]] = {tl: {} for tl in traces}
        for trace_layer in traces:
            vectors = np.stack([clean[sid]["captured"][trace_layer] for sid in base_ids])
            logits = adapter.final_readout_token_logits(vectors, output_token_ids)
            for row, sid in enumerate(base_ids):
                clean_trace_q[trace_layer][sid] = coordinate_value(
                    vectors[row], probe_dirs[trace_layer]
                )
                clean_trace_native[trace_layer][sid] = _selected_margin(logits[row])

        plans = build_context_source_plans(
            discovery_df,
            samples_by_id,
            base_sample_ids=base_ids,
            seed=int(cfg.reproducibility.control_seed),
        )
        q_targets = {
            sid: float(
                frozen_targets["q1_star"]
                if int(label_map[sid]) == 0
                else frozen_targets["q0_star"]
            )
            for sid in base_ids
        }
        semantic_deltas = {
            sid: source_free_setpoint_delta(bases[sid], u, q_targets[sid])
            for sid in base_ids
        }

        reference_info: dict[str, dict[str, Any]] = {}
        standardized_contexts: dict[str, dict[str, np.ndarray]] = {
            condition: {} for condition in CONTEXT_CONDITIONS if condition != "coordinate_only"
        }
        context_diagnostics: dict[str, dict[str, dict[str, Any]]] = {
            condition: {} for condition in standardized_contexts
        }
        source_id_by_condition = {
            "matched_orthogonal": "matched_source_id",
            "same_family_shuffled_orthogonal": "same_family_source_id",
            "different_family_shuffled_orthogonal": "different_family_source_id",
            "same_label_orthogonal": "same_label_source_id",
        }
        random_seeds = [
            int(cfg.reproducibility.control_seed) + 40_000 + index
            for index in range(n_random)
        ]
        random_contexts: dict[int, dict[str, np.ndarray]] = {
            seed: {} for seed in random_seeds
        }
        random_diagnostics: dict[int, dict[str, dict[str, Any]]] = {
            seed: {} for seed in random_seeds
        }
        context_plan_rows: list[dict[str, Any]] = []
        for sid in base_ids:
            plan = plans[sid]
            matched_raw = orthogonal_component(
                activations[int(layer)][plan.matched_source_id], bases[sid], u
            )
            matched_norm = float(np.linalg.norm(matched_raw))
            reference_norm, reference_source, reference_fallback = (
                resolve_context_reference_norm(
                    matched_norm,
                    fallback["median_nondegenerate_matched_orthogonal_norm"],
                    epsilon=CONTEXT_EPSILON,
                )
            )
            reference_info[sid] = {
                "reference_norm": reference_norm,
                "reference_norm_source": reference_source,
                "reference_fallback_used": reference_fallback,
                "matched_raw_norm": matched_norm,
            }
            for condition, attribute in source_id_by_condition.items():
                source_id = str(getattr(plan, attribute))
                source = samples_by_id[source_id]
                raw_context = orthogonal_component(
                    activations[int(layer)][source_id], bases[sid], u
                )
                fallback_seed = int.from_bytes(
                    hashlib.sha256(f"{sid}|{condition}".encode()).digest()[:4],
                    "big",
                )
                fallback_direction = random_unit_direction(
                    hidden_dim, fallback_seed, orthogonal_to=u
                )
                context, diagnostics = standardize_orthogonal_context(
                    raw_context,
                    u,
                    reference_norm,
                    epsilon=CONTEXT_EPSILON,
                    fallback_direction=fallback_direction,
                )
                standardized_contexts[condition][sid] = context
                context_diagnostics[condition][sid] = diagnostics
                context_plan_rows.append(
                    {
                        "base_sample_id": sid,
                        "base_pair_id": str(samples_by_id[sid].pair_id),
                        "base_relation_family": str(samples_by_id[sid].metadata["relation"]),
                        "base_label": int(label_map[sid]),
                        "condition": condition,
                        "context_source_id": source_id,
                        "context_source_pair_id": str(source.pair_id),
                        "context_source_relation_family": str(source.metadata["relation"]),
                        "context_source_label": int(label_map[source_id]),
                        "context_selection_seed": int(plan.selection_seed),
                        "direction_seed": None,
                        **reference_info[sid],
                        **diagnostics,
                    }
                )
            for direction_seed in random_seeds:
                direction = random_unit_direction(
                    hidden_dim, direction_seed, orthogonal_to=u
                )
                context, diagnostics = standardize_orthogonal_context(
                    direction,
                    u,
                    reference_norm,
                    epsilon=CONTEXT_EPSILON,
                )
                random_contexts[direction_seed][sid] = context
                random_diagnostics[direction_seed][sid] = diagnostics
                context_plan_rows.append(
                    {
                        "base_sample_id": sid,
                        "base_pair_id": str(samples_by_id[sid].pair_id),
                        "base_relation_family": str(samples_by_id[sid].metadata["relation"]),
                        "base_label": int(label_map[sid]),
                        "condition": "random_orthogonal",
                        "context_source_id": None,
                        "context_source_pair_id": None,
                        "context_source_relation_family": None,
                        "context_source_label": None,
                        "context_selection_seed": int(plan.selection_seed),
                        "direction_seed": direction_seed,
                        **reference_info[sid],
                        **diagnostics,
                    }
                )
        context_plan = pd.DataFrame(context_plan_rows)
        save_table(context_plan, run_dir / "source_context_plan.parquet")
        plan_digest = _digest(context_plan.to_dict(orient="records"))

        specs: list[dict[str, Any]] = [
            {
                "key": "coordinate_only",
                "condition": "coordinate_only",
                "lambda_context": 0.0,
                "direction_seed": None,
            }
        ]
        structured = [
            "matched_orthogonal",
            "same_family_shuffled_orthogonal",
            "different_family_shuffled_orthogonal",
            "same_label_orthogonal",
        ]
        for strength in lambdas:
            specs.extend(
                {
                    "key": f"{condition}_lambda_{strength:g}",
                    "condition": condition,
                    "lambda_context": strength,
                    "direction_seed": None,
                }
                for condition in structured
            )
            specs.extend(
                {
                    "key": f"random_orthogonal_lambda_{strength:g}_seed_{seed}",
                    "condition": "random_orthogonal",
                    "lambda_context": strength,
                    "direction_seed": seed,
                }
                for seed in random_seeds
            )

        all_raw: list[pd.DataFrame] = []
        all_trace: list[pd.DataFrame] = []
        for spec_index, spec in enumerate(specs):
            resumed_shard = _load_shard(
                run_dir,
                spec,
                base_ids=base_ids,
                plan_digest=plan_digest,
                n_traces=len(traces),
            )
            if resumed_shard is not None:
                raw_shard, trace_shard = resumed_shard
                all_raw.append(raw_shard)
                all_trace.append(trace_shard)
                continue
            condition = str(spec["condition"])
            strength = float(spec["lambda_context"])
            direction_seed = spec.get("direction_seed")
            deltas: dict[str, np.ndarray] = {}
            scaled_contexts: dict[str, np.ndarray] = {}
            diagnostics_by_id: dict[str, dict[str, Any]] = {}
            source_ids: dict[str, str | None] = {}
            for sid in base_ids:
                if condition == "coordinate_only":
                    context = np.zeros(hidden_dim, dtype=np.float64)
                    diagnostics = {
                        "context_raw_norm": 0.0,
                        "context_projected_raw_norm": 0.0,
                        "context_applied_norm": 0.0,
                        "context_dot_truth_direction": 0.0,
                        "context_norm_relative_error": 0.0,
                        "context_vector_fallback_used": False,
                    }
                    source_id = None
                elif condition == "random_orthogonal":
                    context = random_contexts[int(direction_seed)][sid]
                    diagnostics = random_diagnostics[int(direction_seed)][sid]
                    source_id = None
                else:
                    context = standardized_contexts[condition][sid]
                    diagnostics = context_diagnostics[condition][sid]
                    source_id = str(getattr(plans[sid], source_id_by_condition[condition]))
                semantic, scaled_context, total = fixed_setpoint_context_edit(
                    bases[sid], u, q_targets[sid], context, strength
                )
                if not np.array_equal(semantic, semantic_deltas[sid]):
                    raise RuntimeError("coordinate-only semantic delta diverged from E01B-1")
                if float(np.max(np.abs(total - semantic - scaled_context))) > 1e-12:
                    raise RuntimeError("E01B-2 total edit decomposition failed")
                deltas[sid] = total
                scaled_contexts[sid] = scaled_context
                diagnostics_by_id[sid] = diagnostics
                source_ids[sid] = source_id

            results = run_intervention_batches(
                adapter,
                samples_by_id,
                base_ids,
                layer=int(layer),
                token_indices=token_indices,
                deltas_by_id=deltas,
                output_token_ids=output_token_ids,
                capture_layers=traces,
                batch_size=int(cfg.runtime.batch_size),
            )
            trace_native_after: dict[int, dict[str, float]] = {tl: {} for tl in traces}
            for trace_layer in traces:
                vectors = np.stack(
                    [results[sid]["captured"][trace_layer] for sid in base_ids]
                )
                logits = adapter.final_readout_token_logits(vectors, output_token_ids)
                for row, sid in enumerate(base_ids):
                    trace_native_after[trace_layer][sid] = _selected_margin(logits[row])

            raw_rows: list[dict[str, Any]] = []
            trace_rows: list[dict[str, Any]] = []
            for sid in base_ids:
                sample = samples_by_id[sid]
                gold = int(label_map[sid])
                target_label = 1 - gold
                orientation = 1.0 if target_label == 1 else -1.0
                h_base = bases[sid]
                captured = np.asarray(results[sid]["captured"][int(layer)])
                delta = deltas[sid]
                scaled_context = scaled_contexts[sid]
                q_base = coordinate_value(h_base, u)
                q_after = coordinate_value(captured, u)
                margin_before = clean_margins[sid]
                margin_after = _selected_margin(results[sid]["selected_logits"])
                oriented_before = margin_toward_label(margin_before, target_label)
                oriented_after = margin_toward_label(margin_after, target_label)
                oriented_delta = oriented_after - oriented_before
                prediction_before = _prediction(margin_before)
                prediction_after = _prediction(margin_after)
                outcome = counterfactual_outcome(
                    prediction_before, prediction_after, target_label
                )
                source_id = source_ids[sid]
                source = samples_by_id[source_id] if source_id is not None else None
                reference = reference_info[sid]
                diagnostics = diagnostics_by_id[sid]
                expected_state = h_base + delta
                state_error = captured - expected_state
                semantic_norm = float(np.linalg.norm(semantic_deltas[sid]))
                context_delta_norm = float(np.linalg.norm(scaled_context))
                total_norm = float(np.linalg.norm(delta))
                site = token_sites[sid]
                raw_rows.append(
                    {
                        "model_id": str(cfg.model.id),
                        "resolved_revision": resolved_revision,
                        "base_sample_id": sid,
                        "sample_id": sid,
                        "base_pair_id": str(sample.pair_id),
                        "base_relation_family": str(sample.metadata["relation"]),
                        "base_label": gold,
                        "pair_id": str(sample.pair_id),
                        "relation_family": str(sample.metadata["relation"]),
                        "gold_label": gold,
                        "target_label": target_label,
                        "q_base": q_base,
                        "q_target": q_targets[sid],
                        "q_after": q_after,
                        "delta_q": q_after - q_base,
                        "oriented_delta_q_z": orientation
                        * (q_after - q_base)
                        / float(frozen_targets["sigma_q_validation"]),
                        "condition": condition,
                        "lambda_context": strength,
                        "direction_seed": direction_seed,
                        "context_source_id": source_id,
                        "context_source_pair_id": None if source is None else str(source.pair_id),
                        "context_source_relation_family": None
                        if source is None
                        else str(source.metadata["relation"]),
                        "context_source_label": None
                        if source_id is None
                        else int(label_map[source_id]),
                        "context_selection_seed": int(plans[sid].selection_seed),
                        "context_reference_norm": reference["reference_norm"],
                        "reference_norm": reference["reference_norm"],
                        "reference_norm_source": reference["reference_norm_source"],
                        "reference_fallback_used": reference["reference_fallback_used"],
                        "context_raw_norm": diagnostics["context_raw_norm"],
                        "context_projected_raw_norm": diagnostics[
                            "context_projected_raw_norm"
                        ],
                        "context_applied_norm": diagnostics["context_applied_norm"],
                        "context_dot_truth_direction": diagnostics[
                            "context_dot_truth_direction"
                        ],
                        "context_norm_relative_error": diagnostics[
                            "context_norm_relative_error"
                        ],
                        "context_vector_fallback_used": diagnostics[
                            "context_vector_fallback_used"
                        ],
                        "context_fallback_used": bool(
                            reference["reference_fallback_used"]
                            or diagnostics["context_vector_fallback_used"]
                        ),
                        "fallback_used": bool(
                            reference["reference_fallback_used"]
                            or diagnostics["context_vector_fallback_used"]
                        ),
                        "semantic_delta_norm": semantic_norm,
                        "context_delta_norm": context_delta_norm,
                        "total_delta_norm": total_norm,
                        "activation_norm": float(np.linalg.norm(h_base)),
                        "total_delta_norm_ratio": total_norm
                        / max(float(np.linalg.norm(h_base)), 1e-12),
                        "base_yes_no_margin": margin_before,
                        "intervened_yes_no_margin": margin_after,
                        "delta_yes_no_margin": margin_after - margin_before,
                        "margin_toward_target_before": oriented_before,
                        "margin_toward_target_after": oriented_after,
                        "delta_margin_toward_target": oriented_delta,
                        "prediction_before": prediction_before,
                        "prediction_after": prediction_after,
                        "target_flip": outcome["counterfactual_flip"],
                        "expected_target_after": outcome["expected_label_after"],
                        "setpoint_projection_error": abs(q_after - q_targets[sid]),
                        "setpoint_projection_error_validation_sigma": abs(
                            q_after - q_targets[sid]
                        )
                        / float(frozen_targets["sigma_q_validation"]),
                        "target_state_relative_l2_error": float(
                            np.linalg.norm(state_error)
                            / max(float(np.linalg.norm(expected_state)), 1e-12)
                        ),
                        "total_decomposition_max_abs_error": float(
                            np.max(
                                np.abs(
                                    delta - semantic_deltas[sid] - scaled_context
                                )
                            )
                        ),
                        "delta_m_z": oriented_delta
                        / float(frozen_targets["sigma_margin_validation"]),
                        "site": SITE,
                        "layer": int(layer),
                        "native_module_name": adapter.resolve_site(
                            SITE, int(layer)
                        ).native_module_name,
                        "token_selector": SELECTOR,
                        "token_index": int(token_indices[sid]),
                        "prompt_sequence_length": int(site["sequence_length"]),
                        "token_id": int(site["token_id"]),
                        "token_text": str(site["token_text"]),
                        "token_char_start": site["char_start"],
                        "token_char_end": site["char_end"],
                        "chat_template_used": bool(site["chat_template_used"]),
                        "raw_text": str(sample.prompt),
                        "confirmation_accessed": False,
                    }
                )
                for trace_layer in traces:
                    after_vector = np.asarray(results[sid]["captured"][trace_layer])
                    after_q = coordinate_value(after_vector, probe_dirs[trace_layer])
                    delta_q_trace = after_q - clean_trace_q[trace_layer][sid]
                    after_native = trace_native_after[trace_layer][sid]
                    delta_native = after_native - clean_trace_native[trace_layer][sid]
                    layer_ref = frozen_targets["layer_references"][str(trace_layer)]
                    trace_rows.append(
                        {
                            "model_id": str(cfg.model.id),
                            "base_sample_id": sid,
                            "pair_id": str(sample.pair_id),
                            "relation_family": str(sample.metadata["relation"]),
                            "condition": condition,
                            "lambda_context": strength,
                            "direction_seed": direction_seed,
                            "trace_layer": trace_layer,
                            "clean_truth_coordinate": clean_trace_q[trace_layer][sid],
                            "intervened_truth_coordinate": after_q,
                            "delta_truth_coordinate": delta_q_trace,
                            "oriented_delta_q_z": orientation
                            * delta_q_trace
                            / float(layer_ref["sigma_q_validation"]),
                            "clean_native_yes_no_margin": clean_trace_native[
                                trace_layer
                            ][sid],
                            "intervened_native_yes_no_margin": after_native,
                            "delta_native_yes_no_margin": delta_native,
                            "oriented_delta_native_margin_z": orientation
                            * delta_native
                            / float(layer_ref["sigma_margin_validation"]),
                            "confirmation_accessed": False,
                        }
                    )
            raw_shard = pd.DataFrame(raw_rows)
            trace_shard = pd.DataFrame(trace_rows)
            _save_shard(
                run_dir,
                spec,
                base_ids=base_ids,
                plan_digest=plan_digest,
                raw=raw_shard,
                trace=trace_shard,
            )
            all_raw.append(raw_shard)
            all_trace.append(trace_shard)
            status.update(
                message=f"completed context shard {spec['key']}",
                progress={
                    "completed_context_shards": spec_index + 1,
                    "total_context_shards": len(specs),
                },
            )

        raw_df = attach_context_increments(pd.concat(all_raw, ignore_index=True))
        trace_df = attach_trace_context_increments(
            pd.concat(all_trace, ignore_index=True)
        )
        validate_e01b2_artifact_shape(
            raw_df,
            trace_df,
            n_base_examples=len(base_ids),
            n_specs=len(specs),
            n_trace_layers=len(traces),
        )
        finite_columns = [
            "q_base",
            "q_target",
            "q_after",
            "context_reference_norm",
            "context_applied_norm",
            "context_dot_truth_direction",
            "semantic_delta_norm",
            "context_delta_norm",
            "total_delta_norm",
            "activation_norm",
            "base_yes_no_margin",
            "intervened_yes_no_margin",
            "delta_margin_toward_target",
            "context_increment_vs_coordinate_only",
            "setpoint_projection_error",
            "target_state_relative_l2_error",
        ]
        if not np.isfinite(raw_df[finite_columns].to_numpy(float)).all():
            raise RuntimeError("non-finite E01B-2 raw evidence")
        save_table(raw_df, run_dir / "intervention_rows.parquet")
        save_table(trace_df, run_dir / "trace_rows.parquet")

        after_unhooked = run_unintervened_batches(
            adapter,
            samples_by_id,
            base_ids,
            token_indices=token_indices,
            output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        hook_leakage = max(
            float(np.max(np.abs(np.asarray(after_unhooked[sid]) - np.asarray(unhooked[sid]))))
            for sid in base_ids
        )
        non_coordinate = raw_df[raw_df["condition"] != "coordinate_only"]
        max_dot = float(non_coordinate["context_dot_truth_direction"].abs().max())
        max_norm_error = float(non_coordinate["context_norm_relative_error"].max())
        max_projection_sigma = float(
            raw_df["setpoint_projection_error_validation_sigma"].max()
        )
        max_state_error = float(raw_df["target_state_relative_l2_error"].max())
        max_decomposition = float(raw_df["total_decomposition_max_abs_error"].max())
        e01b1_raw = pd.read_parquet(frozen_e01b1_dir / "intervention_rows.parquet")
        e01b1_reference = e01b1_raw[
            e01b1_raw["condition"] == "source_free_opposite_class_median"
        ][["base_sample_id", "q_target", "intervened_yes_no_margin"]]
        coordinate = raw_df[raw_df["condition"] == "coordinate_only"]
        reproduction = coordinate.merge(
            e01b1_reference,
            on="base_sample_id",
            validate="one_to_one",
            suffixes=("_e01b2", "_e01b1"),
        )
        if len(reproduction) != len(coordinate):
            raise RuntimeError("coordinate-only E01B-1 comparison lost rows")
        coordinate_target_deviation = float(
            np.max(
                np.abs(
                    reproduction["q_target_e01b2"].to_numpy(float)
                    - reproduction["q_target_e01b1"].to_numpy(float)
                )
            )
        )
        coordinate_output_deviation = float(
            np.max(
                np.abs(
                    reproduction["intervened_yes_no_margin_e01b2"].to_numpy(float)
                    - reproduction["intervened_yes_no_margin_e01b1"].to_numpy(float)
                )
            )
        )
        tolerances = setpoint_fidelity_tolerances(str(cfg.model.dtype))
        gates = {
            "setpoint_projection_max_validation_sigma": max_projection_sigma,
            "context_dot_u_max_abs": max_dot,
            "context_norm_max_relative_mismatch": max_norm_error,
            "total_decomposition_max_abs_error": max_decomposition,
            "target_state_max_relative_l2_error": max_state_error,
            "coordinate_only_target_vs_e01b1_max_abs": coordinate_target_deviation,
            "coordinate_only_output_vs_e01b1_max_abs": coordinate_output_deviation,
            "hook_leakage_max_logit_deviation": hook_leakage,
            "finite_values": True,
            "source_selection_invariants": True,
            "trace_complete": True,
            "confirmation_accessed": False,
            "tolerances": tolerances,
        }
        if max_projection_sigma > tolerances["projection_validation_sigma"]:
            raise RuntimeError(f"E01B-2 semantic setpoint gate failed: {gates}")
        if max_dot > 1e-10 or max_norm_error > 1e-10 or max_decomposition > 1e-12:
            raise RuntimeError(f"E01B-2 orthogonal context gate failed: {gates}")
        if max_state_error > tolerances["target_state_relative_l2"]:
            raise RuntimeError(f"E01B-2 target-state gate failed: {gates}")
        if coordinate_target_deviation > 1e-10 or coordinate_output_deviation > 0.25:
            raise RuntimeError(f"E01B-2 coordinate-only reproduction failed: {gates}")
        if hook_leakage > 1e-6:
            raise RuntimeError(f"E01B-2 hook leakage gate failed: {gates}")

        n_bootstraps = int(cfg.statistics.bootstrap_samples)
        confidence_level = float(cfg.statistics.confidence_level)
        bootstrap_seed = int(cfg.reproducibility.bootstrap_seed)
        aggregates = aggregate_context_rows(
            raw_df,
            context_strengths=lambdas,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        contrast_pairs = [
            ("matched_orthogonal", "coordinate_only"),
            ("same_family_shuffled_orthogonal", "coordinate_only"),
            ("different_family_shuffled_orthogonal", "coordinate_only"),
            ("same_label_orthogonal", "coordinate_only"),
            ("random_orthogonal", "coordinate_only"),
            ("matched_orthogonal", "random_orthogonal"),
            ("same_family_shuffled_orthogonal", "random_orthogonal"),
            ("different_family_shuffled_orthogonal", "random_orthogonal"),
            ("same_label_orthogonal", "random_orthogonal"),
            ("matched_orthogonal", "different_family_shuffled_orthogonal"),
            (
                "same_family_shuffled_orthogonal",
                "different_family_shuffled_orthogonal",
            ),
        ]
        contrasts: list[dict[str, Any]] = []
        for lambda_index, strength in enumerate(lambdas):
            for pair_index, (left, right) in enumerate(contrast_pairs):
                contrasts.append(
                    paired_context_contrast(
                        raw_df,
                        left_condition=left,
                        right_condition=right,
                        context_strength=strength,
                        n_bootstraps=n_bootstraps,
                        confidence_level=confidence_level,
                        seed=bootstrap_seed + lambda_index * 1000 + pair_index * 43,
                    )
                )
        contrast_df = pd.DataFrame(contrasts)
        relation_metrics = _relation_metrics(
            raw_df,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        trace_metrics = aggregate_trace_context(
            trace_df, context_strengths=lambdas
        )
        save_table(aggregates, run_dir / "aggregate_metrics.parquet")
        save_table(contrast_df, run_dir / "control_contrasts.parquet")
        save_table(relation_metrics, run_dir / "relation_family_metrics.parquet")
        save_table(trace_metrics, run_dir / "trace_metrics.parquet")

        summary = {
            "status": "complete",
            "version": E01B2_VERSION,
            "model_id": str(cfg.model.id),
            "resolved_revision": resolved_revision,
            "profile": str(profile).lower(),
            "layer": int(layer),
            "trace_layers": traces,
            "context_strengths": list(lambdas),
            "random_orthogonal_directions": n_random,
            "n_pairs": len(selected_pairs),
            "n_base_examples": len(base_ids),
            "n_specs": len(specs),
            "candidate_token_ids": output_token_ids,
            "frozen_e01b1_run": str(frozen_e01b1_dir.relative_to(repo_root)),
            "validation_fallback": fallback,
            "fallback_rows": int(raw_df["context_fallback_used"].sum()),
            "gates": gates,
            "aggregate_metrics": aggregates.to_dict(orient="records"),
            "control_contrasts": contrast_df.to_dict(orient="records"),
            "trace_metrics": trace_metrics.to_dict(orient="records"),
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "e01b2_metrics.json")
        _write_summary(run_dir, summary)
        status.complete("E01B-2 bounded discovery run complete")
        manifest.finish(
            runs_summary=[
                {
                    "profile": str(profile).lower(),
                    "n_pairs": len(selected_pairs),
                    "n_rows": len(raw_df),
                    "confirmation_accessed": False,
                }
            ]
        )
        return run_dir
    except Exception as exc:
        logger.exception("E01B-2 failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish(runs_summary=[{"status": "failed", "error": repr(exc)}])
        raise
