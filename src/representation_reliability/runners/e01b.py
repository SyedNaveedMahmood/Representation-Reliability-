"""E01B-1: discovery-only source-free truth-coordinate setpoints."""

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
from ..interventions.setpoint import (
    norm_matched_direction_delta,
    setpoint_identity_diagnostics,
    source_free_setpoint_delta,
)
from ..interventions.truth_coordinate import (
    coordinate_value,
    random_unit_direction,
)
from ..metrics.causal import (
    cluster_bootstrap_mean_ci,
    counterfactual_outcome,
    margin_toward_label,
)
from ..metrics.setpoint import (
    GRID_QUANTILES,
    aggregate_setpoint_rows,
    paired_setpoint_control_contrast,
    summarize_grid_response,
    validation_setpoint_targets,
    validation_standardized_effect,
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
from .e01b_support import (
    find_identity_matched_e01a_snapshot,
    profile_limits,
    validate_artifact_shape,
    validation_identity,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)

E01B1_VERSION = "e01b1-source-free-setpoints-v1"
SITE = "resid_post"
SELECTOR = "last_prompt"


def _allocate_or_resume(
    output_root: Path, run_id: str, *, resume: bool
) -> tuple[Path, bool]:
    root = output_root / "E01B1"
    candidates: list[Path] = []
    canonical = root / run_id
    if canonical.exists():
        candidates.append(canonical)
    reruns = list(root.glob(f"{run_id}-r*"))
    reruns.sort(key=lambda p: int(p.name.rsplit("-r", 1)[1]))
    candidates.extend(reruns)
    if resume:
        for candidate in reversed(candidates):
            status = StatusFile.load(candidate)
            if status is not None and not status.is_complete():
                return candidate, True
    return allocate_run_dir(output_root, "E01B1", run_id), False


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _spec_digest(spec: dict[str, Any], base_ids: list[str]) -> str:
    payload = json.dumps(
        {"version": E01B1_VERSION, "spec": spec, "base_ids": base_ids},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_shard(
    run_dir: Path, spec: dict[str, Any], base_ids: list[str], n_traces: int
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    name = _slug(str(spec["key"]))
    root = run_dir / "shards"
    marker = root / f"{name}.complete.json"
    raw_path = root / f"{name}.intervention.parquet"
    trace_path = root / f"{name}.trace.parquet"
    if not marker.exists() or not raw_path.exists() or not trace_path.exists():
        return None
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta.get("identity") != _spec_digest(spec, base_ids):
        raise RuntimeError(f"E01B shard identity mismatch: {name}")
    raw = pd.read_parquet(raw_path)
    trace = pd.read_parquet(trace_path)
    if len(raw) != len(base_ids) or len(trace) != len(base_ids) * n_traces:
        raise RuntimeError(f"E01B shard row-count mismatch: {name}")
    return raw, trace


def _save_shard(
    run_dir: Path,
    spec: dict[str, Any],
    base_ids: list[str],
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
            "identity": _spec_digest(spec, base_ids),
            "n_intervention_rows": len(raw),
            "n_trace_rows": len(trace),
        },
    )


def _trace_metrics(
    traces: pd.DataFrame,
    *,
    n_bootstraps: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    block = traces[traces["condition"] == "source_free_opposite_class_median"]
    output: list[dict[str, Any]] = []
    for index, (layer, rows) in enumerate(block.groupby("trace_layer", sort=True)):
        q_ci = cluster_bootstrap_mean_ci(
            rows["oriented_delta_q_z"].to_numpy(float),
            rows["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=seed + index * 37,
        )
        m_ci = cluster_bootstrap_mean_ci(
            rows["oriented_delta_native_margin_z"].to_numpy(float),
            rows["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=seed + index * 37 + 1,
        )
        output.append(
            {
                "trace_layer": int(layer),
                "n_rows": len(rows),
                "n_pairs": int(rows["pair_id"].nunique()),
                "mean_oriented_delta_q": float(rows["oriented_delta_q"].mean()),
                "mean_oriented_delta_q_z": q_ci["mean"],
                "delta_q_z_ci_low": q_ci["ci_low"],
                "delta_q_z_ci_high": q_ci["ci_high"],
                "mean_oriented_delta_native_margin": float(
                    rows["oriented_delta_native_margin"].mean()
                ),
                "mean_oriented_delta_native_margin_z": m_ci["mean"],
                "delta_native_margin_z_ci_low": m_ci["ci_low"],
                "delta_native_margin_z_ci_high": m_ci["ci_high"],
            }
        )
    return pd.DataFrame(output)


def _summary_markdown(summary: dict[str, Any]) -> str:
    target = summary["validation_setpoints"]
    return f"""# E01B-1 Source-Free Setpoint Causality

Status: `{summary['status']}`
Model: `{summary['model_id']}`
Profile: `{summary['profile']}`
Site: `resid_post / L{summary['layer']} / last_prompt`

## Scope and provenance

- discovery-only causal evaluation;
- probe fitted on train with validation-only model selection;
- all setpoints and standardization scales constructed from validation only;
- no donor/source activation was used;
- confirmation was not accessed.

## Validation-defined targets

- q0*: `{target['q0_star']}`
- q1*: `{target['q1_star']}`
- grid: `{target['grid']}`
- sigma_q(validation): `{target['sigma_q_validation']}`
- sigma_margin(validation): `{target['sigma_margin_validation']}`

## Numerical gates

- no-op max selected-logit deviation: `{summary['gates']['no_op_max_logit_deviation']}`
- hook-leakage max selected-logit deviation: `{summary['gates']['hook_leakage_max_logit_deviation']}`
- max setpoint projection relative deviation: `{summary['gates']['projection_max_relative_deviation']}`
- max orthogonal relative deviation: `{summary['gates']['orthogonal_max_relative_deviation']}`
- max control norm relative mismatch: `{summary['gates']['control_norm_max_relative_mismatch']}`

## Effects (bounded validation run)

Opposite-class median mean target-oriented margin change: `{summary['opposite_class']['mean_delta_margin_toward_target']}`
Grid median within-example Spearman: `{summary['grid_response']['median_within_base_spearman']}`
Grid monotonic fraction: `{summary['grid_response']['fraction_monotonic_nondecreasing']}`

These are exploratory discovery results from a smoke/pilot profile, not a full-discovery or confirmation result.
"""


def run_e01b(
    base_path=None,
    model_path=None,
    experiment_path=None,
    overrides: tuple[str, ...] = (),
    *,
    layer: int = 17,
    profile: str = "full",
    max_pairs: int | None = None,
    random_directions: int = 10,
    orthogonal_directions: int = 10,
    trace_layers: str = "17,20,23,27",
) -> Path:
    """Run the authorized E01B-1 source-free discovery experiment."""
    cfg, provenance = resolve_config(
        base_path=base_path,
        model_path=model_path,
        experiment_path=experiment_path,
        overrides=overrides,
    )
    if str(cfg.experiment.mode) != "discovery":
        raise RuntimeError("E01B-1 is discovery-only; confirmation is not authorized")
    if str(cfg.experiment.id) != "E01B1":
        raise ValueError(f"E01B-1 runner received experiment id {cfg.experiment.id!r}")
    pair_cap, n_random, n_orthogonal = profile_limits(
        profile, max_pairs, random_directions, orthogonal_directions
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
        raise ValueError(
            f"intervention layer {layer} out of range for {adapter.num_layers} layers"
        )
    traces = parse_trace_layers(
        trace_layers, intervention_layer=int(layer), num_layers=adapter.num_layers
    )
    needed_layers = sorted({int(layer), *traces})
    candidates = list(cfg.behavior.candidates_primary)
    if len(candidates) != 2:
        raise ValueError("E01B-1 requires exactly two primary candidates")
    candidate_lists = candidate_token_id_lists(adapter, candidates)
    if any(len(ids) != 1 for ids in candidate_lists):
        raise RuntimeError(
            "E01B-1 uses first-token readout and requires single-token Yes/No candidates"
        )
    output_token_ids = [int(ids[0]) for ids in candidate_lists]
    if len(set(output_token_ids)) != 2:
        raise RuntimeError("Yes/No candidate token IDs must be distinct")

    revisions = adapter.resolved_revisions()
    resolved_revision = revisions.get("model_sha")
    split_hash = dataset_split_hash(split_of)
    shape = {
        "version": E01B1_VERSION,
        "profile": str(profile).lower(),
        "layer": int(layer),
        "trace_layers": traces,
        "max_pairs": pair_cap,
        "random_directions": n_random,
        "orthogonal_directions": n_orthogonal,
        "grid": [name for name, _ in GRID_QUANTILES],
    }
    shape_json = json.dumps(shape, sort_keys=True)
    resolved_hash = hashlib.sha256(
        (config_hash(cfg) + "|" + shape_json).encode("utf-8")
    ).hexdigest()
    run_id = make_run_id(
        experiment_id="E01B1",
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
        {**provenance, "e01b1_version": E01B1_VERSION, "e01b1_shape": shape},
    )
    manifest = RunManifest(run_dir)
    if resumed:
        status = StatusFile.load(run_dir)
        if status is None or not manifest.path.exists():
            raise RuntimeError("resumable E01B-1 run lacks status or manifest")
        status.update(state="running", message="resuming E01B-1 by treatment shard")
        manifest.manifest = json.loads(manifest.path.read_text(encoding="utf-8"))
    else:
        status = StatusFile.create(run_dir, run_id, "E01B1")
        manifest.set_start(
            resolved_hash,
            {**provenance, "e01b1_version": E01B1_VERSION, "e01b1_shape": shape},
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
        candidate_token_lengths=[len(ids) for ids in candidate_lists],
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
        cache_match = find_identity_matched_e01a_snapshot(
            repo_root,
            model_id=str(cfg.model.id),
            resolved_revision=resolved_revision,
            split_hash=split_hash,
            expected_sample_ids=discovery_ids,
            expected_layers=needed_layers,
        )
        if cache_match is None:
            discovery_samples = [samples_by_id[sid] for sid in discovery_ids]
            activations, token_indices, token_sites = extract_resid_post_layers(
                adapter,
                discovery_samples,
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
            activation_provenance = str(run_dir.relative_to(repo_root))
        else:
            (activations, token_indices, token_sites), source_dir = cache_match
            activation_provenance = str(source_dir.relative_to(repo_root))
        manifest.manifest.setdefault("cache", {})["activation_snapshot_source"] = (
            activation_provenance
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

        validation_ids = discovery_df.loc[
            discovery_df["split"].astype(str) == "validation", "sample_id"
        ].astype(str).tolist()
        validation_labels = np.asarray(
            [int(label_map[sid]) for sid in validation_ids], dtype=int
        )
        primary_vectors = np.stack(
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
        validation_output_margins = np.asarray(
            [_selected_margin(validation_clean[sid]) for sid in validation_ids],
            dtype=np.float64,
        )
        setpoints = validation_setpoint_targets(
            primary_vectors @ probe_dirs[int(layer)],
            validation_labels,
            validation_output_margins,
        )
        layer_references: dict[int, dict[str, float]] = {}
        for trace_layer in needed_layers:
            vectors = np.stack(
                [activations[trace_layer][sid] for sid in validation_ids]
            )
            q_values = vectors @ probe_dirs[trace_layer]
            native_logits = adapter.final_readout_token_logits(vectors, output_token_ids)
            native_margins = native_logits[:, 0] - native_logits[:, 1]
            sigma_q = float(np.std(q_values, ddof=1))
            sigma_margin = float(np.std(native_margins, ddof=1))
            if sigma_q <= 1e-12 or sigma_margin <= 1e-12:
                raise RuntimeError(
                    f"degenerate validation trace scale at layer {trace_layer}"
                )
            layer_references[trace_layer] = {
                "sigma_q_validation": sigma_q,
                "sigma_margin_validation": sigma_margin,
            }
        targets_payload = {
            **setpoints,
            "grid": dict(setpoints["grid"]),
            "layer_references": {str(k): v for k, v in layer_references.items()},
            "provenance": validation_meta,
            "site": SITE,
            "layer": int(layer),
            "token_selector": SELECTOR,
            "candidate_texts": candidates,
            "candidate_token_ids": output_token_ids,
            "candidate_token_lengths": [len(ids) for ids in candidate_lists],
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
            raise RuntimeError("E01B-1 selected no discovery-test examples")
        if any(token_indices[sid] != int(token_sites[sid]["sequence_length"]) - 1 for sid in base_ids):
            raise RuntimeError("last_prompt did not resolve to the final non-padding token")

        hidden_dim = int(adapter.hidden_size)
        u = probe_dirs[int(layer)]
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
        no_op_max = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(clean[sid]["selected_logits"])
                        - np.asarray(unhooked[sid])
                    )
                )
            )
            for sid in base_ids
        )
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

        opposite_targets = {
            sid: float(setpoints["q1_star"] if int(label_map[sid]) == 0 else setpoints["q0_star"])
            for sid in base_ids
        }
        same_targets = {
            sid: float(setpoints["q1_star"] if int(label_map[sid]) == 1 else setpoints["q0_star"])
            for sid in base_ids
        }
        semantic_deltas = {
            sid: source_free_setpoint_delta(bases[sid], u, opposite_targets[sid])
            for sid in base_ids
        }
        random_dirs = {
            int(cfg.reproducibility.control_seed) + 10_000 + index: random_unit_direction(
                hidden_dim, int(cfg.reproducibility.control_seed) + 10_000 + index
            )
            for index in range(n_random)
        }
        orthogonal_dirs = {
            int(cfg.reproducibility.control_seed) + 20_000 + index: random_unit_direction(
                hidden_dim,
                int(cfg.reproducibility.control_seed) + 20_000 + index,
                orthogonal_to=u,
            )
            for index in range(n_orthogonal)
        }
        if any(abs(float(np.dot(direction, u))) > 1e-10 for direction in orthogonal_dirs.values()):
            raise RuntimeError("orthogonal control construction failed")

        specs: list[dict[str, Any]] = [
            {"key": "no_op", "condition": "no_op", "target_name": "q_base"},
            {
                "key": "source_free_opposite_class_median",
                "condition": "source_free_opposite_class_median",
                "target_name": "opposite_class_median",
            },
            {
                "key": "same_class_median",
                "condition": "same_class_median",
                "target_name": "same_class_median",
            },
        ]
        specs.extend(
            {
                "key": f"source_free_grid_{name}",
                "condition": "source_free_grid",
                "target_name": name,
                "q_target": float(setpoints["grid"][name]),
            }
            for name, _ in GRID_QUANTILES
        )
        specs.extend(
            {
                "key": f"random_direction_seed_{seed}",
                "condition": "random_direction",
                "target_name": "opposite_class_median_norm",
                "direction_seed": seed,
            }
            for seed in random_dirs
        )
        specs.extend(
            {
                "key": f"orthogonal_random_seed_{seed}",
                "condition": "orthogonal_random",
                "target_name": "opposite_class_median_norm",
                "direction_seed": seed,
            }
            for seed in orthogonal_dirs
        )

        all_raw: list[pd.DataFrame] = []
        all_trace: list[pd.DataFrame] = []
        for spec_index, spec in enumerate(specs):
            resumed_shard = _load_shard(run_dir, spec, base_ids, len(traces))
            if resumed_shard is not None:
                raw_shard, trace_shard = resumed_shard
                all_raw.append(raw_shard)
                all_trace.append(trace_shard)
                continue
            condition = str(spec["condition"])
            target_name = str(spec["target_name"])
            seed_value = spec.get("direction_seed")
            targets: dict[str, float | None] = {}
            labels: dict[str, int | None] = {}
            deltas: dict[str, np.ndarray] = {}
            for sid in base_ids:
                gold = int(label_map[sid])
                if condition == "no_op":
                    targets[sid] = coordinate_value(bases[sid], u)
                    labels[sid] = 1 - gold
                    deltas[sid] = zeros[sid]
                elif condition == "source_free_opposite_class_median":
                    targets[sid] = opposite_targets[sid]
                    labels[sid] = 1 - gold
                    deltas[sid] = semantic_deltas[sid]
                elif condition == "same_class_median":
                    targets[sid] = same_targets[sid]
                    labels[sid] = gold
                    deltas[sid] = source_free_setpoint_delta(
                        bases[sid], u, same_targets[sid]
                    )
                elif condition == "source_free_grid":
                    targets[sid] = float(spec["q_target"])
                    labels[sid] = None
                    deltas[sid] = source_free_setpoint_delta(
                        bases[sid], u, float(spec["q_target"])
                    )
                elif condition == "random_direction":
                    targets[sid] = None
                    labels[sid] = 1 - gold
                    deltas[sid] = norm_matched_direction_delta(
                        semantic_deltas[sid], random_dirs[int(seed_value)]
                    )
                elif condition == "orthogonal_random":
                    targets[sid] = None
                    labels[sid] = 1 - gold
                    deltas[sid] = norm_matched_direction_delta(
                        semantic_deltas[sid], orthogonal_dirs[int(seed_value)]
                    )
                else:
                    raise RuntimeError(f"unknown E01B condition {condition}")
            results = (
                clean
                if condition == "no_op"
                else run_intervention_batches(
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
                target_label = labels[sid]
                h_base = bases[sid]
                delta = np.asarray(deltas[sid], dtype=np.float64)
                captured = np.asarray(results[sid]["captured"][int(layer)], dtype=np.float64)
                q_base = coordinate_value(h_base, u)
                q_after = coordinate_value(captured, u)
                margin_before = clean_margins[sid]
                margin_after = _selected_margin(results[sid]["selected_logits"])
                prediction_before = _prediction(margin_before)
                prediction_after = _prediction(margin_after)
                if target_label is None:
                    oriented_before = oriented_after = oriented_delta = float("nan")
                    actual_flip = expected_after = 0
                    standardized = {
                        "delta_q_z": (q_after - q_base) / setpoints["sigma_q_validation"],
                        "delta_m_z": (margin_after - margin_before) / setpoints["sigma_margin_validation"],
                        "kappa_z": float("nan"),
                    }
                    orientation_sign = float("nan")
                else:
                    orientation_sign = 1.0 if int(target_label) == 1 else -1.0
                    oriented_before = margin_toward_label(margin_before, target_label)
                    oriented_after = margin_toward_label(margin_after, target_label)
                    oriented_delta = oriented_after - oriented_before
                    outcome = counterfactual_outcome(
                        prediction_before, prediction_after, target_label
                    )
                    actual_flip = outcome["counterfactual_flip"]
                    expected_after = outcome["expected_label_after"]
                    standardized = validation_standardized_effect(
                        q_after - q_base,
                        oriented_delta,
                        sigma_q_validation=setpoints["sigma_q_validation"],
                        sigma_margin_validation=setpoints["sigma_margin_validation"],
                    )
                expected_state = h_base + delta
                fidelity = captured - expected_state
                diagnostics = (
                    setpoint_identity_diagnostics(h_base, captured, u, targets[sid])
                    if targets[sid] is not None
                    else {
                        "projection_abs_deviation": float("nan"),
                        "projection_relative_deviation": float("nan"),
                        "orthogonal_abs_deviation": float("nan"),
                        "orthogonal_relative_deviation": float("nan"),
                    }
                )
                semantic_norm = float(np.linalg.norm(semantic_deltas[sid]))
                delta_norm = float(np.linalg.norm(delta))
                norm_mismatch = (
                    abs(delta_norm - semantic_norm)
                    if condition in {"random_direction", "orthogonal_random"}
                    else 0.0
                )
                site = token_sites[sid]
                raw_rows.append(
                    {
                        "model_id": str(cfg.model.id),
                        "resolved_revision": resolved_revision,
                        "sample_id": sid,
                        "base_sample_id": sid,
                        "pair_id": str(sample.pair_id),
                        "relation_family": str(sample.metadata.get("relation")),
                        "gold_label": gold,
                        "target_label": target_label,
                        "condition": condition,
                        "target_name": target_name,
                        "direction_seed": seed_value,
                        "site": SITE,
                        "layer": int(layer),
                        "native_module_name": adapter.resolve_site(SITE, int(layer)).native_module_name,
                        "token_selector": SELECTOR,
                        "token_index": int(token_indices[sid]),
                        "prompt_sequence_length": int(site["sequence_length"]),
                        "token_id": int(site["token_id"]),
                        "token_text": str(site["token_text"]),
                        "token_char_start": site["char_start"],
                        "token_char_end": site["char_end"],
                        "chat_template_used": bool(site["chat_template_used"]),
                        "raw_text": str(sample.prompt),
                        "q_base": q_base,
                        "q_target": targets[sid],
                        "q_after": q_after,
                        "delta_q": q_after - q_base,
                        "oriented_delta_q": orientation_sign * (q_after - q_base),
                        "base_yes_no_margin": margin_before,
                        "intervened_yes_no_margin": margin_after,
                        "delta_yes_no_margin": margin_after - margin_before,
                        "margin_toward_target_before": oriented_before,
                        "margin_toward_target_after": oriented_after,
                        "delta_margin_toward_target": oriented_delta,
                        "prediction_before": prediction_before,
                        "prediction_after": prediction_after,
                        "expected_target_after": expected_after,
                        "actual_target_flip": actual_flip,
                        "activation_norm": float(np.linalg.norm(h_base)),
                        "delta_norm": delta_norm,
                        "delta_norm_ratio": delta_norm / max(float(np.linalg.norm(h_base)), 1e-12),
                        "semantic_reference_delta_norm": semantic_norm,
                        "control_norm_abs_mismatch": norm_mismatch,
                        "control_norm_relative_mismatch": norm_mismatch / max(semantic_norm, 1e-12),
                        "target_state_max_abs_deviation": float(np.max(np.abs(fidelity))),
                        "target_state_relative_l2_deviation": float(np.linalg.norm(fidelity) / max(float(np.linalg.norm(expected_state)), 1e-12)),
                        **diagnostics,
                        **standardized,
                        "oriented_delta_q_z": orientation_sign * standardized["delta_q_z"],
                        "confirmation_accessed": False,
                    }
                )
                for trace_layer in traces:
                    after_vector = np.asarray(results[sid]["captured"][trace_layer])
                    after_q = coordinate_value(after_vector, probe_dirs[trace_layer])
                    delta_q_trace = after_q - clean_trace_q[trace_layer][sid]
                    after_native = trace_native_after[trace_layer][sid]
                    delta_native = after_native - clean_trace_native[trace_layer][sid]
                    layer_ref = layer_references[trace_layer]
                    trace_rows.append(
                        {
                            "model_id": str(cfg.model.id),
                            "base_sample_id": sid,
                            "pair_id": str(sample.pair_id),
                            "relation_family": str(sample.metadata.get("relation")),
                            "gold_label": gold,
                            "target_label": target_label,
                            "condition": condition,
                            "target_name": target_name,
                            "direction_seed": seed_value,
                            "trace_layer": trace_layer,
                            "q_target": targets[sid],
                            "clean_truth_coordinate": clean_trace_q[trace_layer][sid],
                            "intervened_truth_coordinate": after_q,
                            "delta_truth_coordinate": delta_q_trace,
                            "oriented_delta_q": orientation_sign * delta_q_trace,
                            "oriented_delta_q_z": orientation_sign * delta_q_trace / layer_ref["sigma_q_validation"],
                            "clean_native_yes_no_margin": clean_trace_native[trace_layer][sid],
                            "intervened_native_yes_no_margin": after_native,
                            "delta_native_yes_no_margin": delta_native,
                            "oriented_delta_native_margin": orientation_sign * delta_native,
                            "oriented_delta_native_margin_z": orientation_sign * delta_native / layer_ref["sigma_margin_validation"],
                            "confirmation_accessed": False,
                        }
                    )
            raw_shard = pd.DataFrame(raw_rows)
            trace_shard = pd.DataFrame(trace_rows)
            _save_shard(run_dir, spec, base_ids, raw_shard, trace_shard)
            all_raw.append(raw_shard)
            all_trace.append(trace_shard)
            status.update(
                message=f"completed treatment shard {spec['key']}",
                progress={"completed_treatment_shards": spec_index + 1, "total_treatment_shards": len(specs)},
            )

        raw_df = pd.concat(all_raw, ignore_index=True)
        trace_df = pd.concat(all_trace, ignore_index=True)
        validate_artifact_shape(
            raw_df,
            trace_df,
            n_base_examples=len(base_ids),
            n_treatment_specs=len(specs),
            n_trace_layers=len(traces),
        )
        finite_columns = [
            "q_base", "q_after", "delta_q", "base_yes_no_margin",
            "intervened_yes_no_margin", "delta_yes_no_margin",
            "activation_norm", "delta_norm", "delta_norm_ratio",
            "target_state_max_abs_deviation", "target_state_relative_l2_deviation",
        ]
        if not np.isfinite(raw_df[finite_columns].to_numpy(float)).all():
            raise RuntimeError("non-finite E01B raw evidence")
        save_table(raw_df, run_dir / "intervention_rows.parquet")
        save_table(trace_df, run_dir / "trace_rows.parquet")

        hooked_after = run_unintervened_batches(
            adapter,
            samples_by_id,
            base_ids,
            token_indices=token_indices,
            output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        hook_leakage = max(
            float(np.max(np.abs(np.asarray(hooked_after[sid]) - np.asarray(unhooked[sid]))))
            for sid in base_ids
        )
        semantic = raw_df[
            raw_df["condition"].isin(
                {"no_op", "source_free_opposite_class_median", "same_class_median", "source_free_grid"}
            )
        ]
        projection_max = float(semantic["projection_relative_deviation"].max())
        orthogonal_max = float(semantic["orthogonal_relative_deviation"].max())
        fidelity_max = float(raw_df["target_state_relative_l2_deviation"].max())
        control_rows = raw_df[
            raw_df["condition"].isin({"random_direction", "orthogonal_random"})
        ]
        norm_max = float(control_rows["control_norm_relative_mismatch"].max())
        gates = {
            "no_op_max_logit_deviation": no_op_max,
            "hook_leakage_max_logit_deviation": hook_leakage,
            "projection_max_relative_deviation": projection_max,
            "orthogonal_max_relative_deviation": orthogonal_max,
            "target_state_max_relative_l2_deviation": fidelity_max,
            "control_norm_max_relative_mismatch": norm_max,
            "finite_values": True,
            "last_prompt_indexing": True,
            "candidate_single_token": True,
            "row_identity": True,
            "right_padding_safe": True,
        }
        if no_op_max > 1e-6 or hook_leakage > 1e-6:
            raise RuntimeError(f"E01B no-op/hook-leakage gate failed: {gates}")
        if projection_max > 0.02 or orthogonal_max > 0.02 or fidelity_max > 0.02:
            raise RuntimeError(f"E01B setpoint fidelity gate failed: {gates}")
        if norm_max > 1e-10:
            raise RuntimeError(f"E01B control norm gate failed: {gates}")

        n_bootstraps = int(cfg.statistics.bootstrap_samples)
        confidence_level = float(cfg.statistics.confidence_level)
        bootstrap_seed = int(cfg.reproducibility.bootstrap_seed)
        aggregates = aggregate_setpoint_rows(
            raw_df,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        grid_rows = raw_df[raw_df["condition"] == "source_free_grid"].copy()
        grid_metrics, grid_examples, grid_summary = summarize_grid_response(
            grid_rows,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        contrasts = pd.DataFrame(
            [
                paired_setpoint_control_contrast(
                    raw_df,
                    control=control,
                    n_bootstraps=n_bootstraps,
                    confidence_level=confidence_level,
                    seed=bootstrap_seed + index * 101,
                )
                for index, control in enumerate(("random_direction", "orthogonal_random"))
            ]
        )
        trace_metrics = _trace_metrics(
            trace_df,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        save_table(aggregates, run_dir / "aggregate_metrics.parquet")
        save_table(grid_metrics, run_dir / "grid_metrics.parquet")
        save_table(grid_examples, run_dir / "grid_example_metrics.parquet")
        save_table(contrasts, run_dir / "control_contrasts.parquet")
        save_table(trace_metrics, run_dir / "trace_metrics.parquet")

        opposite_block = raw_df[
            raw_df["condition"] == "source_free_opposite_class_median"
        ]
        opposite_ci = cluster_bootstrap_mean_ci(
            opposite_block["delta_margin_toward_target"].to_numpy(float),
            opposite_block["pair_id"].astype(str).tolist(),
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        opposite_summary = {
            "n": len(opposite_block),
            "mean_delta_margin_toward_target": opposite_ci["mean"],
            "median_delta_margin_toward_target": float(opposite_block["delta_margin_toward_target"].median()),
            "ci_low": opposite_ci["ci_low"],
            "ci_high": opposite_ci["ci_high"],
            "actual_flip_rate": float(opposite_block["actual_target_flip"].mean()),
            "expected_target_rate_after": float(opposite_block["expected_target_after"].mean()),
            "mean_oriented_delta_q_z": float(opposite_block["oriented_delta_q_z"].mean()),
            "mean_delta_m_z": float(opposite_block["delta_m_z"].mean()),
            "mean_kappa_z": float(opposite_block["kappa_z"].mean()),
        }
        summary = {
            "status": "complete",
            "version": E01B1_VERSION,
            "model_id": str(cfg.model.id),
            "resolved_revision": resolved_revision,
            "profile": str(profile).lower(),
            "layer": int(layer),
            "trace_layers": traces,
            "n_pairs": len(selected_pairs),
            "n_base_examples": len(base_ids),
            "conditions": sorted(raw_df["condition"].unique().tolist()),
            "random_directions": n_random,
            "orthogonal_directions": n_orthogonal,
            "candidate_texts": candidates,
            "candidate_token_ids": output_token_ids,
            "candidate_token_lengths": [len(ids) for ids in candidate_lists],
            "activation_snapshot_source": activation_provenance,
            "validation_setpoints": targets_payload,
            "opposite_class": opposite_summary,
            "grid_response": grid_summary,
            "control_contrasts": contrasts.to_dict(orient="records"),
            "trace_trajectory": trace_metrics.to_dict(orient="records"),
            "gates": gates,
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "e01b1_metrics.json")
        (run_dir / "E01B_SUMMARY.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
        status.complete("E01B-1 discovery run complete")
        manifest.finish(
            runs_summary=[
                {
                    "profile": str(profile).lower(),
                    "n_pairs": len(selected_pairs),
                    "n_rows": len(raw_df),
                    "mean_opposite_target_effect": opposite_summary[
                        "mean_delta_margin_toward_target"
                    ],
                    "confirmation_accessed": False,
                }
            ]
        )
        return run_dir
    except Exception as exc:
        logger.exception("E01B-1 failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish(runs_summary=[{"status": "failed", "error": repr(exc)}])
        raise
