"""E01B-3: additive-vs-gating factorial decomposition."""

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
    standardize_orthogonal_context,
)
from ..interventions.setpoint import setpoint_fidelity_tolerances
from ..interventions.truth_coordinate import coordinate_value, random_unit_direction
from ..metrics.causal import counterfactual_outcome, margin_toward_label
from ..metrics.factorial import (
    FACTORIAL_CONTEXTS,
    aggregate_factorial_metrics,
    aggregate_factorial_trace,
    build_factorial_rows,
    build_factorial_trace_rows,
    paired_factorial_contrast,
    relation_family_factorial_metrics,
)
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
from .e01b2_support import CONTEXT_EPSILON, build_context_source_plans, parse_context_strengths
from .e01b3_support import (
    e01b3_profile_limits,
    find_frozen_e01b2_run,
    probe_scaler_digest,
    validate_e01b3_artifact_shape,
    validate_prior_manifest_identity,
    validate_source_plan_identity,
)
from .e01b_support import find_identity_matched_e01a_snapshot, validation_identity
from .extract import load_adapter

logger = logging.getLogger(__name__)

E01B3_VERSION = "e01b3-factorial-v1"
SITE = "resid_post"
SELECTOR = "last_prompt"


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _allocate_or_resume(output_root: Path, run_id: str, *, resume: bool) -> tuple[Path, bool]:
    root = output_root / "E01B3"
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
    return allocate_run_dir(output_root, "E01B3", run_id), False


def _shard_identity(spec: dict[str, Any], base_ids: list[str], plan_digest: str) -> str:
    return _digest(
        {
            "version": E01B3_VERSION,
            "spec": spec,
            "base_ids": base_ids,
            "source_plan_digest": plan_digest,
        }
    )


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
    raw_path = root / f"{name}.context_only.parquet"
    trace_path = root / f"{name}.context_only_trace.parquet"
    if not marker.exists() or not raw_path.exists() or not trace_path.exists():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("identity") != _shard_identity(spec, base_ids, plan_digest):
        raise RuntimeError(f"E01B-3 shard identity mismatch: {name}")
    raw = pd.read_parquet(raw_path)
    trace = pd.read_parquet(trace_path)
    if len(raw) != len(base_ids) or len(trace) != len(base_ids) * n_traces:
        raise RuntimeError(f"E01B-3 shard row-count mismatch: {name}")
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
    save_table(raw, root / f"{name}.context_only.parquet")
    save_table(trace, root / f"{name}.context_only_trace.parquet")
    atomic_write_json(
        root / f"{name}.complete.json",
        {
            "identity": _shard_identity(spec, base_ids, plan_digest),
            "n_context_only_rows": len(raw),
            "n_context_only_trace_rows": len(trace),
        },
    )


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# E01B-3 Additive-vs-Gating Factorial Decomposition",
        "",
        f"Status: `{summary['status']}`",
        f"Model: `{summary['model_id']}`",
        f"Profile: `{summary['profile']}`",
        "",
        "Discovery-only bounded evidence. Confirmation was not accessed.",
        "Only the Y01 context-only arm was generated for the full selected set;",
        "Y00/Y10/Y11 came from identity-audited immutable E01B-2 evidence.",
        "",
        "## Compatibility and numerical gates",
        "",
    ]
    for key, value in summary["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Lambda-1 factorial effects", ""])
    additive = {
        row["condition"]: row
        for row in summary["additive_metrics"]
        if float(row["lambda_context"]) == 1.0
    }
    interaction = {
        row["condition"]: row
        for row in summary["interaction_metrics"]
        if float(row["lambda_context"]) == 1.0
    }
    for condition in FACTORIAL_CONTEXTS:
        a = additive[condition]
        g = interaction[condition]
        lines.append(
            f"- {condition}: A={a['mean']:.6f} "
            f"[{a['ci_low']:.6f}, {a['ci_high']:.6f}], "
            f"G={g['mean']:.6f} [{g['ci_low']:.6f}, {g['ci_high']:.6f}]"
        )
    lines.extend(
        [
            "",
            "These are smoke/pilot discovery estimates. They do not authorize",
            "full discovery or confirmation access.",
            "",
        ]
    )
    (run_dir / "E01B3_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _reconstruct_context_plan(
    *,
    persisted: pd.DataFrame,
    discovery_df: pd.DataFrame,
    samples_by_id: dict[str, Any],
    activations: dict[int, dict[str, np.ndarray]],
    bases: dict[str, np.ndarray],
    base_ids: list[str],
    u: np.ndarray,
    layer: int,
    seed: int,
) -> tuple[dict[tuple[str, str, int], np.ndarray], pd.DataFrame]:
    """Rebuild frozen E01B-2 vectors while retaining its exact source plan."""
    deterministic = build_context_source_plans(
        discovery_df,
        samples_by_id,
        base_sample_ids=base_ids,
        seed=seed,
    )
    expected_sources = {
        "matched_orthogonal": "matched_source_id",
        "same_family_shuffled_orthogonal": "same_family_source_id",
        "different_family_shuffled_orthogonal": "different_family_source_id",
        "same_label_orthogonal": "same_label_source_id",
    }
    context_vectors: dict[tuple[str, str, int], np.ndarray] = {}
    reconstructed: list[dict[str, Any]] = []
    for record in persisted.to_dict(orient="records"):
        sid = str(record["base_sample_id"])
        condition = str(record["condition"])
        direction_seed = -1 if pd.isna(record["direction_seed"]) else int(record["direction_seed"])
        if condition == "random_orthogonal":
            raw = random_unit_direction(len(u), direction_seed, orthogonal_to=u)
            fallback_direction = None
            expected_source = None
        else:
            expected_source = str(getattr(deterministic[sid], expected_sources[condition]))
            if str(record["context_source_id"]) != expected_source:
                raise RuntimeError(f"frozen source-plan mismatch for {sid}/{condition}")
            raw = orthogonal_component(activations[int(layer)][expected_source], bases[sid], u)
            fallback_seed = int.from_bytes(
                hashlib.sha256(f"{sid}|{condition}".encode()).digest()[:4], "big"
            )
            fallback_direction = random_unit_direction(len(u), fallback_seed, orthogonal_to=u)
        context, diagnostics = standardize_orthogonal_context(
            raw,
            u,
            float(record["reference_norm"]),
            epsilon=CONTEXT_EPSILON,
            fallback_direction=fallback_direction,
        )
        key = (sid, condition, direction_seed)
        if key in context_vectors:
            raise RuntimeError(f"duplicate reconstructed context vector: {key}")
        context_vectors[key] = context
        matched_source = str(deterministic[sid].matched_source_id)
        matched_raw = orthogonal_component(activations[int(layer)][matched_source], bases[sid], u)
        reconstructed.append(
            {
                **record,
                "context_source_id": expected_source,
                "matched_raw_norm": float(np.linalg.norm(matched_raw)),
                **diagnostics,
            }
        )
    return context_vectors, pd.DataFrame(reconstructed)


def run_e01b3(
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
        raise RuntimeError("E01B-3 is discovery-only; confirmation is not authorized")
    if str(cfg.experiment.id) != "E01B3":
        raise ValueError(f"E01B-3 runner received experiment id {cfg.experiment.id!r}")
    if int(layer) != 17:
        raise ValueError("E01B-3 intervention layer is frozen at 17")
    lambdas = parse_context_strengths(context_strengths)
    pair_cap, n_random = e01b3_profile_limits(profile, max_pairs, random_orthogonal_directions)

    samples, full_df = _generate_dataset(cfg)
    discovery_df = discovery_view(full_df).reset_index(drop=True)
    validation_meta = validation_identity(discovery_df)
    label_map = build_discovery_label_map(discovery_df)
    discovery_ids = discovery_df["sample_id"].astype(str).tolist()
    discovery_set = set(discovery_ids)
    samples_by_id = {
        str(sample.sample_id): sample
        for sample in samples
        if str(sample.sample_id) in discovery_set
    }
    split_of = dict(zip(discovery_df["sample_id"].astype(str), discovery_df["split"].astype(str)))
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
        raise RuntimeError("E01B-3 selected no discovery examples")

    adapter = load_adapter(cfg)
    if not 0 <= int(layer) < adapter.num_layers:
        raise ValueError(f"intervention layer {layer} out of range")
    traces = parse_trace_layers(
        trace_layers, intervention_layer=int(layer), num_layers=adapter.num_layers
    )
    if traces != [17, 20, 23, 27]:
        raise ValueError("E01B-3 trace layers are frozen at 17,20,23,27")
    needed_layers = sorted({int(layer), *traces})
    candidates = list(cfg.behavior.candidates_primary)
    candidate_lists = candidate_token_id_lists(adapter, candidates)
    if len(candidates) != 2 or any(len(ids) != 1 for ids in candidate_lists):
        raise RuntimeError("E01B-3 requires single-token Yes/No candidates")
    output_token_ids = [int(ids[0]) for ids in candidate_lists]
    if len(set(output_token_ids)) != 2:
        raise RuntimeError("Yes/No candidate token IDs must be distinct")
    revisions = adapter.resolved_revisions()
    resolved_revision = revisions.get("model_sha")
    split_hash = dataset_split_hash(split_of)

    repo_root = Path(__file__).resolve().parents[3]
    prior_dir = find_frozen_e01b2_run(
        repo_root,
        model_id=str(cfg.model.id),
        resolved_revision=resolved_revision,
        profile=str(profile).lower(),
        n_pairs=len(selected_pairs),
        random_directions=n_random,
        lambdas=lambdas,
        trace_layers=traces,
    )
    prior_manifest = json.loads((prior_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_prior_manifest_identity(
        prior_manifest,
        model_id=str(cfg.model.id),
        resolved_revision=resolved_revision,
        tokenizer_revision=revisions.get("tokenizer_sha"),
        candidate_token_ids=output_token_ids,
        split_hash=split_hash,
    )
    prior_rows = pd.read_parquet(prior_dir / "intervention_rows.parquet")
    prior_trace = pd.read_parquet(prior_dir / "trace_rows.parquet")
    prior_plan = pd.read_parquet(prior_dir / "source_context_plan.parquet")
    frozen_targets = json.loads((prior_dir / "setpoint_targets.json").read_text(encoding="utf-8"))
    if (
        frozen_targets.get("confirmation_accessed") is not False
        or set(prior_rows["base_sample_id"].astype(str)) != set(base_ids)
        or not bool(prior_rows["confirmation_accessed"].eq(False).all())
        or not bool(prior_trace["confirmation_accessed"].eq(False).all())
    ):
        raise RuntimeError("prior E01B-2 evidence identity or confirmation lock failed")

    shape = {
        "version": E01B3_VERSION,
        "profile": str(profile).lower(),
        "layer": int(layer),
        "trace_layers": traces,
        "max_pairs": pair_cap,
        "context_strengths": lambdas,
        "random_orthogonal_directions": n_random,
        "prior_e01b2_run": prior_dir.name,
    }
    resolved_hash = hashlib.sha256(
        (config_hash(cfg) + "|" + json.dumps(shape, sort_keys=True)).encode()
    ).hexdigest()
    run_id = make_run_id(
        experiment_id="E01B3",
        config_hash=resolved_hash,
        seed=int(cfg.reproducibility.seed),
        model_revision=resolved_revision or cfg.model.revision or "unpinned",
        dataset_split_hash=split_hash,
    )
    run_dir, resumed = _allocate_or_resume(
        repo_root / cfg.project.output_root,
        run_id,
        resume=bool(cfg.runtime.resume),
    )
    save_resolved_config(
        cfg,
        run_dir / "config.resolved.yaml",
        {**provenance, "e01b3_version": E01B3_VERSION, "e01b3_shape": shape},
    )
    manifest = RunManifest(run_dir)
    if resumed:
        status = StatusFile.load(run_dir)
        if status is None or not manifest.path.exists():
            raise RuntimeError("resumable E01B-3 run lacks status or manifest")
        status.update(state="running", message="resuming E01B-3 by Y01 shard")
        manifest.manifest = json.loads(manifest.path.read_text(encoding="utf-8"))
    else:
        status = StatusFile.create(run_dir, run_id, "E01B3")
        manifest.set_start(
            resolved_hash,
            {**provenance, "e01b3_version": E01B3_VERSION, "e01b3_shape": shape},
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
            f"{SITE}:{trace_layer}": adapter.resolve_site(SITE, trace_layer).native_module_name
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
        if any(
            token_indices[sid] != int(token_sites[sid]["sequence_length"]) - 1 for sid in base_ids
        ):
            raise RuntimeError("last_prompt did not resolve to final non-padding token")

        probe_dirs: dict[int, np.ndarray] = {}
        probe_fits: dict[int, dict[str, Any]] = {}
        probe_metrics: list[dict[str, Any]] = []
        for trace_layer in needed_layers:
            fit, direction, metrics = _layer_probe(
                trace_layer,
                activations,
                discovery_df,
                label_map,
                c_grid=cfg.probe.C_grid,
                seed=int(cfg.reproducibility.probe_seed),
            )
            probe_dirs[trace_layer] = direction
            probe_fits[trace_layer] = fit
            probe_metrics.append(metrics)
        probe_digest = probe_scaler_digest(probe_fits)
        probe_metrics_df = pd.DataFrame(probe_metrics)
        prior_probe = pd.read_parquet(prior_dir / "probe_metrics.parquet")
        compare_columns = ["layer", "site", "token_selector", "chosen_C", "sample_ids_digest"]
        if not probe_metrics_df[compare_columns].equals(prior_probe[compare_columns]):
            raise RuntimeError("reconstructed probe scientific identity mismatch")
        save_table(probe_metrics_df, run_dir / "probe_metrics.parquet")
        u = probe_dirs[int(layer)]

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
        clean_margins = {sid: _selected_margin(clean[sid]["selected_logits"]) for sid in base_ids}
        clean_trace_q: dict[int, dict[str, float]] = {trace: {} for trace in traces}
        clean_trace_m: dict[int, dict[str, float]] = {trace: {} for trace in traces}
        for trace_layer in traces:
            vectors = np.stack([clean[sid]["captured"][trace_layer] for sid in base_ids])
            logits = adapter.final_readout_token_logits(vectors, output_token_ids)
            for row, sid in enumerate(base_ids):
                clean_trace_q[trace_layer][sid] = coordinate_value(
                    vectors[row], probe_dirs[trace_layer]
                )
                clean_trace_m[trace_layer][sid] = _selected_margin(logits[row])

        context_vectors, reconstructed_plan = _reconstruct_context_plan(
            persisted=prior_plan,
            discovery_df=discovery_df,
            samples_by_id=samples_by_id,
            activations=activations,
            bases=bases,
            base_ids=base_ids,
            u=u,
            layer=int(layer),
            seed=int(cfg.reproducibility.control_seed),
        )
        plan_audit = validate_source_plan_identity(prior_plan, reconstructed_plan)
        save_table(prior_plan, run_dir / "source_context_plan.parquet")
        plan_digest = _digest(prior_plan.to_dict(orient="records"))

        q_targets = {
            sid: float(
                frozen_targets["q1_star"] if int(label_map[sid]) == 0 else frozen_targets["q0_star"]
            )
            for sid in base_ids
        }
        prior_coordinate = prior_rows[prior_rows["condition"] == "coordinate_only"].set_index(
            "base_sample_id"
        )
        target_reproduction = max(
            abs(q_targets[sid] - float(prior_coordinate.loc[sid, "q_target"])) for sid in base_ids
        )
        q_base_reproduction = max(
            abs(coordinate_value(bases[sid], u) - float(prior_coordinate.loc[sid, "q_base"]))
            for sid in base_ids
        )
        y00_reproduction = max(
            abs(clean_margins[sid] - float(prior_coordinate.loc[sid, "base_yes_no_margin"]))
            for sid in base_ids
        )

        # Independently reproduce all reused arms on one exact-size leading batch.
        contract_ids = base_ids[: max(1, min(int(cfg.runtime.batch_size), len(base_ids)))]
        semantic_deltas: dict[str, np.ndarray] = {}
        combined_deltas: dict[str, np.ndarray] = {}
        for sid in contract_ids:
            semantic, context, total = fixed_setpoint_context_edit(
                bases[sid],
                u,
                q_targets[sid],
                context_vectors[(sid, "matched_orthogonal", -1)],
                1.0,
            )
            semantic_deltas[sid] = semantic
            combined_deltas[sid] = total
            if abs(float(np.dot(context, u))) > 1e-10:
                raise RuntimeError("contract context lost orthogonality")
        contract_y10 = run_intervention_batches(
            adapter,
            samples_by_id,
            contract_ids,
            layer=int(layer),
            token_indices=token_indices,
            deltas_by_id=semantic_deltas,
            output_token_ids=output_token_ids,
            capture_layers=[int(layer)],
            batch_size=int(cfg.runtime.batch_size),
        )
        contract_y11 = run_intervention_batches(
            adapter,
            samples_by_id,
            contract_ids,
            layer=int(layer),
            token_indices=token_indices,
            deltas_by_id=combined_deltas,
            output_token_ids=output_token_ids,
            capture_layers=[int(layer)],
            batch_size=int(cfg.runtime.batch_size),
        )
        prior_matched = prior_rows[
            (prior_rows["condition"] == "matched_orthogonal")
            & np.isclose(prior_rows["lambda_context"].to_numpy(float), 1.0)
        ].set_index("base_sample_id")
        y10_output_reproduction = max(
            abs(
                _selected_margin(contract_y10[sid]["selected_logits"])
                - float(prior_coordinate.loc[sid, "intervened_yes_no_margin"])
            )
            for sid in contract_ids
        )
        y11_output_reproduction = max(
            abs(
                _selected_margin(contract_y11[sid]["selected_logits"])
                - float(prior_matched.loc[sid, "intervened_yes_no_margin"])
            )
            for sid in contract_ids
        )
        y10_q_reproduction = max(
            abs(
                coordinate_value(contract_y10[sid]["captured"][int(layer)], u)
                - float(prior_coordinate.loc[sid, "q_after"])
            )
            for sid in contract_ids
        )
        y11_q_reproduction = max(
            abs(
                coordinate_value(contract_y11[sid]["captured"][int(layer)], u)
                - float(prior_matched.loc[sid, "q_after"])
            )
            for sid in contract_ids
        )

        specs: list[dict[str, Any]] = []
        for strength in lambdas:
            for condition in FACTORIAL_CONTEXTS:
                if condition == "random_orthogonal":
                    seeds = sorted(
                        int(seed)
                        for seed in prior_plan.loc[
                            prior_plan["condition"] == condition, "direction_seed"
                        ]
                        .dropna()
                        .unique()
                    )
                else:
                    seeds = [-1]
                for direction_seed in seeds:
                    specs.append(
                        {
                            "key": f"{condition}_lambda_{strength:g}_seed_{direction_seed}",
                            "condition": condition,
                            "lambda_context": float(strength),
                            "direction_seed": direction_seed,
                        }
                    )
        if (
            sum(spec["condition"] == "random_orthogonal" for spec in specs)
            != len(lambdas) * n_random
        ):
            raise RuntimeError("frozen random-seed count mismatch")

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
            direction_seed = int(spec["direction_seed"])
            deltas = {
                sid: strength * context_vectors[(sid, condition, direction_seed)]
                for sid in base_ids
            }
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
            trace_after_m: dict[int, dict[str, float]] = {trace: {} for trace in traces}
            for trace_layer in traces:
                vectors = np.stack([results[sid]["captured"][trace_layer] for sid in base_ids])
                logits = adapter.final_readout_token_logits(vectors, output_token_ids)
                for row, sid in enumerate(base_ids):
                    trace_after_m[trace_layer][sid] = _selected_margin(logits[row])

            raw_records: list[dict[str, Any]] = []
            trace_records: list[dict[str, Any]] = []
            plan_block = prior_plan[
                (prior_plan["condition"] == condition)
                & (
                    pd.to_numeric(prior_plan["direction_seed"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    == direction_seed
                )
            ].set_index("base_sample_id")
            for sid in base_ids:
                sample = samples_by_id[sid]
                gold = int(label_map[sid])
                target_label = 1 - gold
                orientation = 1.0 if target_label == 1 else -1.0
                h_base = bases[sid]
                h_after = np.asarray(results[sid]["captured"][int(layer)])
                q_base = coordinate_value(h_base, u)
                q_after = coordinate_value(h_after, u)
                margin_before = clean_margins[sid]
                margin_after = _selected_margin(results[sid]["selected_logits"])
                oriented_before = margin_toward_label(margin_before, target_label)
                oriented_after = margin_toward_label(margin_after, target_label)
                prediction_before = _prediction(margin_before)
                prediction_after = _prediction(margin_after)
                outcome = counterfactual_outcome(prediction_before, prediction_after, target_label)
                plan = plan_block.loc[sid]
                delta = deltas[sid]
                expected_state = h_base + delta
                site = token_sites[sid]
                source_id = (
                    None if pd.isna(plan["context_source_id"]) else str(plan["context_source_id"])
                )
                raw_records.append(
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
                        "direction_seed": None if direction_seed == -1 else direction_seed,
                        "context_source_id": source_id,
                        "context_source_pair_id": plan["context_source_pair_id"],
                        "context_source_relation_family": plan["context_source_relation_family"],
                        "context_source_label": plan["context_source_label"],
                        "context_selection_seed": int(plan["context_selection_seed"]),
                        "context_reference_norm": float(plan["reference_norm"]),
                        "reference_norm": float(plan["reference_norm"]),
                        "reference_norm_source": str(plan["reference_norm_source"]),
                        "reference_fallback_used": bool(plan["reference_fallback_used"]),
                        "context_raw_norm": float(plan["context_raw_norm"]),
                        "context_projected_raw_norm": float(plan["context_projected_raw_norm"]),
                        "context_applied_norm": float(plan["context_applied_norm"]),
                        "context_dot_truth_direction": float(plan["context_dot_truth_direction"]),
                        "context_norm_relative_error": float(plan["context_norm_relative_error"]),
                        "context_vector_fallback_used": bool(plan["context_vector_fallback_used"]),
                        "context_fallback_used": bool(
                            plan["reference_fallback_used"] or plan["context_vector_fallback_used"]
                        ),
                        "semantic_delta_norm": 0.0,
                        "context_delta_norm": float(np.linalg.norm(delta)),
                        "total_delta_norm": float(np.linalg.norm(delta)),
                        "activation_norm": float(np.linalg.norm(h_base)),
                        "total_delta_norm_ratio": float(np.linalg.norm(delta))
                        / max(float(np.linalg.norm(h_base)), 1e-12),
                        "base_yes_no_margin": margin_before,
                        "intervened_yes_no_margin": margin_after,
                        "delta_yes_no_margin": margin_after - margin_before,
                        "margin_toward_target_before": oriented_before,
                        "margin_toward_target_after": oriented_after,
                        "delta_margin_toward_target": oriented_after - oriented_before,
                        "prediction_before": prediction_before,
                        "prediction_after": prediction_after,
                        "target_flip": outcome["counterfactual_flip"],
                        "expected_target_after": outcome["expected_label_after"],
                        "context_only_q_preservation_error": abs(q_after - q_base),
                        "context_only_q_preservation_validation_sigma": abs(q_after - q_base)
                        / float(frozen_targets["sigma_q_validation"]),
                        "target_state_relative_l2_error": float(
                            np.linalg.norm(h_after - expected_state)
                            / max(float(np.linalg.norm(expected_state)), 1e-12)
                        ),
                        "delta_m_z": (oriented_after - oriented_before)
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
                    after_m = trace_after_m[trace_layer][sid]
                    trace_records.append(
                        {
                            "model_id": str(cfg.model.id),
                            "base_sample_id": sid,
                            "pair_id": str(sample.pair_id),
                            "relation_family": str(sample.metadata["relation"]),
                            "condition": condition,
                            "lambda_context": strength,
                            "direction_seed": None if direction_seed == -1 else direction_seed,
                            "trace_layer": trace_layer,
                            "clean_truth_coordinate": clean_trace_q[trace_layer][sid],
                            "intervened_truth_coordinate": after_q,
                            "delta_truth_coordinate": after_q - clean_trace_q[trace_layer][sid],
                            "clean_native_yes_no_margin": clean_trace_m[trace_layer][sid],
                            "intervened_native_yes_no_margin": after_m,
                            "delta_native_yes_no_margin": after_m - clean_trace_m[trace_layer][sid],
                            "confirmation_accessed": False,
                        }
                    )
            raw_shard = pd.DataFrame(raw_records)
            trace_shard = pd.DataFrame(trace_records)
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
                message=f"completed context-only shard {spec['key']}",
                progress={
                    "completed_context_only_shards": spec_index + 1,
                    "total_context_only_shards": len(specs),
                },
            )

        context_df = pd.concat(all_raw, ignore_index=True)
        context_trace_df = pd.concat(all_trace, ignore_index=True)
        factorial_df, compatibility = build_factorial_rows(
            prior_rows,
            context_df,
            y11_run_id=prior_dir.name,
            y01_run_id=run_dir.name,
            numeric_tolerance=1e-8,
        )
        factorial_trace_df = build_factorial_trace_rows(
            prior_trace,
            context_trace_df,
            factorial_df[["base_sample_id", "target_label"]],
            layer_references=frozen_targets["layer_references"],
        )
        validate_e01b3_artifact_shape(
            context_df,
            context_trace_df,
            factorial_df,
            factorial_trace_df,
            n_base_examples=len(base_ids),
            n_context_specs=len(specs),
            n_trace_layers=len(traces),
        )
        finite_columns = [
            "q_base",
            "q_after",
            "context_applied_norm",
            "context_dot_truth_direction",
            "context_delta_norm",
            "activation_norm",
            "base_yes_no_margin",
            "intervened_yes_no_margin",
            "context_only_q_preservation_error",
            "target_state_relative_l2_error",
        ]
        if not np.isfinite(context_df[finite_columns].to_numpy(float)).all():
            raise RuntimeError("non-finite E01B-3 context-only evidence")

        save_table(context_df, run_dir / "context_only_rows.parquet")
        save_table(context_trace_df, run_dir / "context_only_trace_rows.parquet")
        save_table(factorial_df, run_dir / "factorial_rows.parquet")
        save_table(factorial_trace_df, run_dir / "factorial_trace_rows.parquet")

        tolerances = setpoint_fidelity_tolerances(str(cfg.model.dtype))
        max_q_preservation = float(context_df["context_only_q_preservation_error"].max())
        max_q_preservation_sigma = float(
            context_df["context_only_q_preservation_validation_sigma"].max()
        )
        max_context_dot = float(context_df["context_dot_truth_direction"].abs().max())
        expected_norm = context_df["lambda_context"].to_numpy(float) * context_df[
            "context_applied_norm"
        ].to_numpy(float)
        norm_relative = np.abs(
            context_df["context_delta_norm"].to_numpy(float) - expected_norm
        ) / np.maximum(expected_norm, 1e-12)
        max_norm_mismatch = float(np.max(norm_relative))
        max_state_error = float(context_df["target_state_relative_l2_error"].max())
        after_unhooked = run_unintervened_batches(
            adapter,
            samples_by_id,
            base_ids,
            token_indices=token_indices,
            output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        hook_leakage = max(
            float(np.max(np.abs(after_unhooked[sid] - unhooked[sid]))) for sid in base_ids
        )
        gates = {
            "compatibility_mismatches": compatibility["compatibility_mismatches"],
            "coordinate_only_reproduction_max_deviation": y10_output_reproduction,
            "q_plus_context_reproduction_max_deviation": y11_output_reproduction,
            "Y00_reproduction_max_deviation": y00_reproduction,
            "q_base_reproduction_max_deviation": q_base_reproduction,
            "q_target_reproduction_max_deviation": target_reproduction,
            "Y10_q_reproduction_max_deviation": y10_q_reproduction,
            "Y11_q_reproduction_max_deviation": y11_q_reproduction,
            "context_only_q_preservation_max_abs": max_q_preservation,
            "context_only_q_preservation_max_validation_sigma": max_q_preservation_sigma,
            "context_dot_u_max_abs": max_context_dot,
            "context_norm_max_relative_mismatch": max_norm_mismatch,
            "target_state_max_relative_l2_error": max_state_error,
            "source_plan_mismatch_count": plan_audit["source_plan_mismatch_count"],
            "hook_leakage_max_logit_deviation": hook_leakage,
            "finite_values": True,
            "trace_complete": True,
            "confirmation_accessed": False,
            "tolerances": tolerances,
        }
        if (
            max_q_preservation_sigma > tolerances["projection_validation_sigma"]
            or max_context_dot > 1e-10
            or max_norm_mismatch > 1e-10
        ):
            raise RuntimeError(f"E01B-3 context-only numerical gate failed: {gates}")
        if max_state_error > tolerances["target_state_relative_l2"]:
            raise RuntimeError(f"E01B-3 target-state gate failed: {gates}")
        if (
            y00_reproduction > 1e-6
            or y10_output_reproduction > 1e-6
            or y11_output_reproduction > 1e-6
            or target_reproduction > 1e-10
        ):
            raise RuntimeError(f"E01B-3 prior-arm reproduction failed: {gates}")
        if hook_leakage > 1e-6:
            raise RuntimeError(f"E01B-3 hook leakage gate failed: {gates}")

        n_bootstraps = int(cfg.statistics.bootstrap_samples)
        confidence_level = float(cfg.statistics.confidence_level)
        bootstrap_seed = int(cfg.reproducibility.bootstrap_seed)
        additive, interaction = aggregate_factorial_metrics(
            factorial_df,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        contrast_pairs = [
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
        contrast_records: list[dict[str, Any]] = []
        for lambda_index, strength in enumerate(lambdas):
            for metric_index, metric in enumerate(("A_context", "G_interaction")):
                for pair_index, (left, right) in enumerate(contrast_pairs):
                    contrast_records.append(
                        paired_factorial_contrast(
                            factorial_df,
                            metric=metric,
                            left_condition=left,
                            right_condition=right,
                            context_strength=strength,
                            n_bootstraps=n_bootstraps,
                            confidence_level=confidence_level,
                            seed=bootstrap_seed
                            + lambda_index * 1000
                            + metric_index * 300
                            + pair_index * 41,
                        )
                    )
        contrasts = pd.DataFrame(contrast_records)
        relation_metrics = relation_family_factorial_metrics(
            factorial_df,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        trace_metrics = aggregate_factorial_trace(factorial_trace_df)
        save_table(additive, run_dir / "additive_metrics.parquet")
        save_table(interaction, run_dir / "interaction_metrics.parquet")
        save_table(contrasts, run_dir / "structured_contrasts.parquet")
        save_table(relation_metrics, run_dir / "relation_family_metrics.parquet")
        save_table(trace_metrics, run_dir / "trace_metrics.parquet")

        factorial_manifest = {
            "version": E01B3_VERSION,
            "project_scientific_identity": shape,
            "prior_e01b2_run": str(prior_dir.relative_to(repo_root)),
            "activation_snapshot_source": activation_source,
            "probe_scaler_sha256": probe_digest,
            "source_context_plan_sha256": plan_digest,
            "candidate_token_ids": output_token_ids,
            "validation_target_artifact": str(
                (prior_dir / "setpoint_targets.json").relative_to(repo_root)
            ),
            "arm_provenance": {
                "Y00": prior_dir.name,
                "Y10": prior_dir.name,
                "Y01": run_dir.name,
                "Y11": prior_dir.name,
            },
            "compatibility_audit": {**compatibility, **plan_audit},
            "confirmation_accessed": False,
        }
        save_json(factorial_manifest, run_dir / "factorial_manifest.json")
        manifest.manifest.setdefault("cache", {}).update(
            {
                "activation_snapshot_source": activation_source,
                "prior_e01b2_run": str(prior_dir.relative_to(repo_root)),
                "probe_scaler_sha256": probe_digest,
                "source_context_plan_sha256": plan_digest,
            }
        )
        manifest.write()
        summary = {
            "status": "complete",
            "version": E01B3_VERSION,
            "model_id": str(cfg.model.id),
            "resolved_revision": resolved_revision,
            "profile": str(profile).lower(),
            "layer": int(layer),
            "trace_layers": traces,
            "context_strengths": list(lambdas),
            "random_orthogonal_directions": n_random,
            "n_pairs": len(selected_pairs),
            "n_base_examples": len(base_ids),
            "n_context_specs": len(specs),
            "prior_e01b2_run": str(prior_dir.relative_to(repo_root)),
            "candidate_token_ids": output_token_ids,
            "probe_scaler_sha256": probe_digest,
            "gates": gates,
            "additive_metrics": additive.to_dict(orient="records"),
            "interaction_metrics": interaction.to_dict(orient="records"),
            "structured_contrasts": contrasts.to_dict(orient="records"),
            "trace_metrics": trace_metrics.to_dict(orient="records"),
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "e01b3_metrics.json")
        _write_summary(run_dir, summary)
        status.complete("E01B-3 bounded factorial decomposition complete")
        manifest.finish(
            runs_summary=[
                {
                    "profile": str(profile).lower(),
                    "n_pairs": len(selected_pairs),
                    "n_context_only_rows": len(context_df),
                    "confirmation_accessed": False,
                }
            ]
        )
        return run_dir
    except Exception as exc:
        logger.exception("E01B-3 failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish(runs_summary=[{"status": "failed", "error": repr(exc)}])
        raise
