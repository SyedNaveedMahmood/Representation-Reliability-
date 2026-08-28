"""E01A — causal conversion of a frozen decoded truth coordinate.

Discovery-only runner. It trains frozen layer-specific truth probes using
train/validation activations, then evaluates controlled residual-stream edits
on discovery-test matched twins. Confirmation is never exposed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.splits import build_discovery_label_map, discovery_view
from ..interventions.truth_coordinate import (
    coordinate_transfer_delta,
    coordinate_value,
    full_residual_patch_delta,
    normalized_direction,
    random_unit_direction,
)
from ..metrics.causal import (
    aggregate_intervention_rows,
    dose_response_summary,
    margin_toward_label,
    paired_control_contrast,
)
from ..metrics.decoding import classification_metrics
from ..probes.linear import fit_probe, raw_probe_direction, transform_features
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .e00c import _generate_dataset, candidate_token_id_lists
from .extract import load_adapter
from .e01a_support import (
    alpha_profile,
    build_source_plans,
    deterministic_subset_pair_ids,
    extract_resid_post_layers,
    parse_trace_layers,
    run_intervention_batches,
)

logger = logging.getLogger(__name__)

E01A_VERSION = "e01a-causal-conversion-v1"
SITE = "resid_post"
SELECTOR = "last_prompt"


def _profile_limits(
    profile: str, max_pairs: int | None, random_directions: int
) -> tuple[int | None, int]:
    p = str(profile).lower()
    if max_pairs is not None:
        pair_cap = int(max_pairs)
    elif p == "smoke":
        pair_cap = 25
    elif p == "pilot":
        pair_cap = 75
    else:
        pair_cap = None
    if p == "smoke":
        n_random = 1
    elif p == "pilot":
        n_random = min(3, int(random_directions))
    else:
        n_random = int(random_directions)
    if n_random <= 0:
        raise ValueError("random_directions must be positive")
    return pair_cap, n_random


def _layer_probe(
    layer: int,
    activations: dict[int, dict[str, np.ndarray]],
    discovery_df: pd.DataFrame,
    label_map: dict[str, int],
    *,
    c_grid,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    def block(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        ids = discovery_df.loc[
            discovery_df["split"].astype(str) == split, "sample_id"
        ].astype(str).tolist()
        X = np.stack([activations[int(layer)][sid] for sid in ids])
        y = np.asarray([int(label_map[sid]) for sid in ids], dtype=int)
        return X, y, ids

    Xtr, ytr, _ = block("train")
    Xval, yval, _ = block("validation")
    Xte, yte, test_ids = block("discovery_test")
    fit = fit_probe(
        Xtr,
        ytr,
        Xval,
        yval,
        c_grid=c_grid,
        seed=int(seed),
        standardize=True,
        class_weight="balanced",
    )
    scores = fit["classifier"].decision_function(transform_features(fit, Xte))
    metrics = classification_metrics(yte, scores)
    metrics.update(
        {
            "layer": int(layer),
            "site": SITE,
            "token_selector": SELECTOR,
            "chosen_C": float(fit["chosen_C"]),
            "n_discovery_test": int(len(yte)),
            "sample_ids_digest": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        }
    )
    direction = normalized_direction(raw_probe_direction(fit))
    return fit, direction, metrics


def _selected_margin(logits: np.ndarray) -> float:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(arr) != 2:
        raise ValueError("E01A expects exactly two selected answer logits")
    return float(arr[0] - arr[1])


def _prediction(margin_yes_no: float) -> int:
    return int(float(margin_yes_no) >= 0.0)


def _write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# E01A Causal Conversion Discovery

Status: `{summary['status']}`  
Model: `{summary['model_id']}`  
Profile: `{summary['profile']}`  
Intervention: `resid_post / L{summary['layer']} / last_prompt`

## Scope

- discovery only; confirmation remained locked;
- frozen truth direction trained on train and selected on validation;
- causal evaluation on discovery-test matched twins;
- primary treatment changes only the probe-defined truth coordinate;
- full residual replacement is an upper-bound control.

## Probe validity

Intervention-layer discovery AUROC: `{summary['intervention_layer_probe_auroc']:.6f}`

## Run shape

- selected pairs: `{summary['n_pairs']}`
- directed base examples: `{summary['n_base_examples']}`
- alphas: `{summary['alphas']}`
- random control directions: `{summary['random_directions']}`
- trace layers: `{summary['trace_layers']}`

## Fidelity

- max alpha=0 selected-logit deviation from clean baseline: `{summary['alpha0_max_logit_deviation']:.6g}`
- max target-layer intervention fidelity deviation: `{summary['target_fidelity_max_abs']:.6g}`
- max random/orthogonal norm mismatch: `{summary['control_norm_match_max_abs']:.6g}`

## Pilot/discovery effect snapshot

Truth-coordinate alpha=1 mean counterfactual-oriented delta margin: `{summary.get('truth_alpha1_mean_effect')}`

Dose-response slope: `{summary['truth_dose_response']['slope']}`  
Dose-response Pearson r: `{summary['truth_dose_response']['pearson_r']}`

This runner establishes causal sensitivity under an intervention. It does not
by itself prove that the unperturbed model naturally uses exactly the same
one-dimensional coordinate as its endogenous causal variable.
"""
    (run_dir / "E01A_SUMMARY.md").write_text(text, encoding="utf-8")


def run_e01a(
    base_path=None,
    model_path=None,
    experiment_path=None,
    overrides: tuple[str, ...] = (),
    *,
    layer: int = 17,
    profile: str = "full",
    max_pairs: int | None = None,
    random_directions: int = 10,
    trace_layers: str = "17,20,23,26,27",
) -> Path:
    cfg, provenance = resolve_config(
        base_path=base_path,
        model_path=model_path,
        experiment_path=experiment_path,
        overrides=overrides,
    )
    if str(cfg.experiment.mode) != "discovery":
        raise RuntimeError("E01A runner is discovery-only; confirmation is not authorized")
    if str(cfg.experiment.id) != "E01A":
        raise ValueError(f"E01A runner received experiment id {cfg.experiment.id!r}")

    alphas = alpha_profile(profile)
    pair_cap, n_random = _profile_limits(profile, max_pairs, random_directions)

    samples, df = _generate_dataset(cfg)
    discovery_df = discovery_view(df).reset_index(drop=True)
    label_map = build_discovery_label_map(discovery_df)
    discovery_id_set = set(discovery_df["sample_id"].astype(str))
    samples_by_id = {
        str(s.sample_id): s for s in samples if str(s.sample_id) in discovery_id_set
    }
    split_of = dict(
        zip(discovery_df["sample_id"].astype(str), discovery_df["split"].astype(str))
    )
    if any(split == "confirmation" for split in split_of.values()):
        raise RuntimeError("confirmation rows leaked into E01A discovery view")

    adapter = load_adapter(cfg)
    if not (0 <= int(layer) < adapter.num_layers):
        raise ValueError(
            f"intervention layer {layer} out of range for {adapter.num_layers} layers"
        )
    traces = parse_trace_layers(
        trace_layers, intervention_layer=int(layer), num_layers=adapter.num_layers
    )
    needed_layers = sorted(set([int(layer), *traces]))

    candidates = list(cfg.behavior.candidates_primary)
    if len(candidates) != 2:
        raise ValueError("E01A requires exactly two primary behavior candidates")
    candidate_lists = candidate_token_id_lists(adapter, candidates)
    output_token_ids = [int(ids[0]) for ids in candidate_lists]
    if any(len(ids) == 0 for ids in candidate_lists):
        raise RuntimeError("empty answer candidate tokenization")

    revisions = adapter.resolved_revisions()
    resolved_revision = revisions.get("model_sha")
    split_hash = dataset_split_hash(split_of)
    shape_payload = json.dumps(
        {
            "version": E01A_VERSION,
            "profile": str(profile),
            "layer": int(layer),
            "trace_layers": traces,
            "max_pairs": pair_cap,
            "random_directions": n_random,
            "alphas": alphas,
        },
        sort_keys=True,
    )
    resolved_hash = hashlib.sha256(
        (config_hash(cfg) + "|" + shape_payload).encode()
    ).hexdigest()
    run_id = make_run_id(
        experiment_id="E01A",
        config_hash=resolved_hash,
        seed=int(cfg.reproducibility.seed),
        model_revision=resolved_revision or cfg.model.revision or "unpinned",
        dataset_split_hash=split_hash,
    )
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = allocate_run_dir(repo_root / cfg.project.output_root, "E01A", run_id)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_resolved_config(
        cfg,
        run_dir / "config.resolved.yaml",
        {
            **provenance,
            "e01a_version": E01A_VERSION,
            "e01a_shape": json.loads(shape_payload),
        },
    )
    status = StatusFile.create(run_dir, run_id, "E01A")
    manifest = RunManifest(run_dir)
    manifest.set_start(
        resolved_hash,
        {
            **provenance,
            "e01a_version": E01A_VERSION,
            "e01a_shape": json.loads(shape_payload),
        },
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
        notes={"revision_resolution": revisions},
    )
    manifest.update_dataset_info(split_hash=split_hash)

    try:
        discovery_samples = [
            samples_by_id[sid]
            for sid in discovery_df["sample_id"].astype(str).tolist()
        ]
        activations, token_indices = extract_resid_post_layers(
            adapter,
            discovery_samples,
            layers=needed_layers,
            token_selector=SELECTOR,
            batch_size=int(cfg.runtime.batch_size),
        )

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
            probe_dirs[int(trace_layer)] = direction
            probe_metrics.append(metrics)
        save_table(pd.DataFrame(probe_metrics), run_dir / "probe_metrics.parquet")

        selected_pair_ids = deterministic_subset_pair_ids(
            discovery_df,
            max_pairs=pair_cap,
            seed=int(cfg.reproducibility.control_seed),
        )
        base_df = discovery_df[
            (discovery_df["split"] == "discovery_test")
            & (discovery_df["pair_id"].astype(str).isin(selected_pair_ids))
        ].copy()
        base_ids = base_df["sample_id"].astype(str).tolist()
        if not base_ids:
            raise RuntimeError("E01A selected no discovery-test base examples")
        plans = build_source_plans(
            discovery_df,
            samples_by_id,
            base_sample_ids=base_ids,
            seed=int(cfg.reproducibility.control_seed),
        )

        hidden_dim = int(adapter.hidden_size)
        u = probe_dirs[int(layer)]
        random_dirs: dict[int, np.ndarray] = {}
        ortho_dirs: dict[int, np.ndarray] = {}
        for j in range(n_random):
            r_seed = int(cfg.reproducibility.control_seed) + 10_000 + j
            o_seed = int(cfg.reproducibility.control_seed) + 20_000 + j
            random_dirs[r_seed] = random_unit_direction(hidden_dim, r_seed)
            ortho_dirs[o_seed] = random_unit_direction(
                hidden_dim, o_seed, orthogonal_to=u
            )

        zeros = {
            sid: np.zeros(hidden_dim, dtype=np.float64) for sid in base_ids
        }
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
        clean_margin = {
            sid: _selected_margin(clean[sid]["selected_logits"]) for sid in base_ids
        }
        clean_trace_native: dict[int, dict[str, float]] = {
            tl: {} for tl in traces
        }
        clean_trace_coord: dict[int, dict[str, float]] = {
            tl: {} for tl in traces
        }
        for tl in traces:
            vectors = np.stack([clean[sid]["captured"][tl] for sid in base_ids])
            trace_logits = adapter.final_readout_token_logits(
                vectors, output_token_ids
            )
            for row, sid in enumerate(base_ids):
                clean_trace_native[tl][sid] = _selected_margin(trace_logits[row])
                clean_trace_coord[tl][sid] = coordinate_value(
                    vectors[row], probe_dirs[tl]
                )

        raw_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        max_fidelity = 0.0
        max_norm_mismatch = 0.0
        alpha0_max_logit_dev = 0.0

        def evaluate_condition(
            *,
            condition: str,
            alpha: float,
            deltas: dict[str, np.ndarray],
            source_ids: dict[str, str],
            direction_seed: int | None = None,
        ) -> None:
            nonlocal max_fidelity, max_norm_mismatch, alpha0_max_logit_dev
            if float(alpha) == 0.0:
                results = clean
            else:
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
            trace_native_after: dict[int, dict[str, float]] = {
                tl: {} for tl in traces
            }
            for tl in traces:
                vectors = np.stack(
                    [results[sid]["captured"][tl] for sid in base_ids]
                )
                logits = adapter.final_readout_token_logits(vectors, output_token_ids)
                for row, sid in enumerate(base_ids):
                    trace_native_after[tl][sid] = _selected_margin(logits[row])

            for sid in base_ids:
                plan = plans[sid]
                sample = samples_by_id[sid]
                matched_id = plan.matched_source_id
                source_id = source_ids[sid]
                expected_label = int(sample.expected_counterfactual_label)
                if expected_label != int(label_map[matched_id]):
                    raise RuntimeError(f"counterfactual label mismatch for {sid}")
                h_base = activations[int(layer)][sid]
                h_matched = activations[int(layer)][matched_id]
                delta = np.asarray(deltas[sid], dtype=np.float64)
                activation_norm = float(np.linalg.norm(h_base))
                delta_norm = float(np.linalg.norm(delta))
                q_base = coordinate_value(h_base, u)
                q_matched = coordinate_value(h_matched, u)
                captured_target = np.asarray(
                    results[sid]["captured"][int(layer)], dtype=np.float64
                )
                expected_target = h_base + delta
                fidelity = float(np.max(np.abs(captured_target - expected_target)))
                max_fidelity = max(max_fidelity, fidelity)
                q_after = coordinate_value(captured_target, u)
                margin_before = clean_margin[sid]
                margin_after = _selected_margin(results[sid]["selected_logits"])
                if float(alpha) == 0.0:
                    alpha0_max_logit_dev = max(
                        alpha0_max_logit_dev, abs(margin_after - margin_before)
                    )
                oriented_before = margin_toward_label(
                    margin_before, expected_label
                )
                oriented_after = margin_toward_label(margin_after, expected_label)
                oriented_delta = oriented_after - oriented_before
                pred_before = _prediction(margin_before)
                pred_after = _prediction(margin_after)
                dq = q_after - q_base
                kappa = (
                    float(oriented_delta / max(abs(dq), 1e-12))
                    if condition == "truth_coordinate" and abs(dq) > 1e-12
                    else float("nan")
                )
                truth_delta_norm = float(
                    np.linalg.norm(
                        coordinate_transfer_delta(
                            h_base, h_matched, u, float(alpha)
                        )
                    )
                )
                if condition in {"random_direction", "orthogonal_random"}:
                    max_norm_mismatch = max(
                        max_norm_mismatch, abs(delta_norm - truth_delta_norm)
                    )
                raw_rows.append(
                    {
                        "model_id": cfg.model.id,
                        "resolved_revision": resolved_revision,
                        "base_sample_id": sid,
                        "sample_id": sid,
                        "pair_id": str(sample.pair_id),
                        "relation_family": str(sample.metadata.get("relation")),
                        "gold_label": int(label_map[sid]),
                        "expected_label": expected_label,
                        "matched_source_id": matched_id,
                        "source_sample_id": source_id,
                        "same_label_source_id": plan.same_label_source_id,
                        "shuffled_source_id": plan.shuffled_source_id,
                        "source_selection_seed": int(plan.selection_seed),
                        "site": SITE,
                        "layer": int(layer),
                        "token_selector": SELECTOR,
                        "token_index": int(token_indices[sid]),
                        "condition": condition,
                        "direction_seed": direction_seed,
                        "alpha": float(alpha),
                        "base_truth_coordinate": q_base,
                        "source_truth_coordinate": coordinate_value(
                            activations[int(layer)][source_id], u
                        ),
                        "matched_truth_coordinate": q_matched,
                        "intervened_truth_coordinate": q_after,
                        "delta_truth_coordinate": dq,
                        "base_yes_no_margin": margin_before,
                        "intervened_yes_no_margin": margin_after,
                        "delta_yes_no_margin": margin_after - margin_before,
                        "margin_toward_expected_before": oriented_before,
                        "margin_toward_expected_after": oriented_after,
                        "delta_margin_toward_expected": oriented_delta,
                        "activation_norm": activation_norm,
                        "delta_norm": delta_norm,
                        "delta_norm_ratio": delta_norm / max(activation_norm, 1e-12),
                        "truth_reference_delta_norm": truth_delta_norm,
                        "target_fidelity_max_abs": fidelity,
                        "prediction_before": pred_before,
                        "prediction_after": pred_after,
                        "counterfactual_flip": int(pred_after == expected_label),
                        "kappa": kappa,
                        "yes_token_id": int(output_token_ids[0]),
                        "no_token_id": int(output_token_ids[1]),
                        "yes_candidate_token_ids": json.dumps(candidate_lists[0]),
                        "no_candidate_token_ids": json.dumps(candidate_lists[1]),
                        "first_token_metric_only": bool(
                            len(candidate_lists[0]) != 1
                            or len(candidate_lists[1]) != 1
                        ),
                    }
                )

                for tl in traces:
                    vec_after = np.asarray(
                        results[sid]["captured"][tl], dtype=np.float64
                    )
                    q_trace_after = coordinate_value(
                        vec_after, probe_dirs[tl]
                    )
                    trace_rows.append(
                        {
                            "model_id": cfg.model.id,
                            "base_sample_id": sid,
                            "pair_id": str(sample.pair_id),
                            "condition": condition,
                            "direction_seed": direction_seed,
                            "alpha": float(alpha),
                            "trace_layer": int(tl),
                            "clean_truth_coordinate": clean_trace_coord[tl][sid],
                            "intervened_truth_coordinate": q_trace_after,
                            "delta_truth_coordinate": (
                                q_trace_after - clean_trace_coord[tl][sid]
                            ),
                            "clean_native_yes_no_margin": clean_trace_native[tl][sid],
                            "intervened_native_yes_no_margin": (
                                trace_native_after[tl][sid]
                            ),
                            "delta_native_yes_no_margin": (
                                trace_native_after[tl][sid]
                                - clean_trace_native[tl][sid]
                            ),
                        }
                    )

        for alpha in alphas:
            truth_deltas: dict[str, np.ndarray] = {}
            same_deltas: dict[str, np.ndarray] = {}
            shuffled_deltas: dict[str, np.ndarray] = {}
            full_deltas: dict[str, np.ndarray] = {}
            matched_sources: dict[str, str] = {}
            same_sources: dict[str, str] = {}
            shuffled_sources: dict[str, str] = {}
            for sid in base_ids:
                plan = plans[sid]
                h_base = activations[int(layer)][sid]
                h_match = activations[int(layer)][plan.matched_source_id]
                h_same = activations[int(layer)][plan.same_label_source_id]
                h_shuf = activations[int(layer)][plan.shuffled_source_id]
                truth_deltas[sid] = coordinate_transfer_delta(
                    h_base, h_match, u, float(alpha)
                )
                same_deltas[sid] = coordinate_transfer_delta(
                    h_base, h_same, u, float(alpha)
                )
                shuffled_deltas[sid] = coordinate_transfer_delta(
                    h_base, h_shuf, u, float(alpha)
                )
                full_deltas[sid] = full_residual_patch_delta(
                    h_base, h_match, float(alpha)
                )
                matched_sources[sid] = plan.matched_source_id
                same_sources[sid] = plan.same_label_source_id
                shuffled_sources[sid] = plan.shuffled_source_id

            evaluate_condition(
                condition="truth_coordinate",
                alpha=float(alpha),
                deltas=truth_deltas,
                source_ids=matched_sources,
            )
            evaluate_condition(
                condition="same_label_coordinate",
                alpha=float(alpha),
                deltas=same_deltas,
                source_ids=same_sources,
            )
            evaluate_condition(
                condition="shuffled_coordinate",
                alpha=float(alpha),
                deltas=shuffled_deltas,
                source_ids=shuffled_sources,
            )
            evaluate_condition(
                condition="full_residual_patch",
                alpha=float(alpha),
                deltas=full_deltas,
                source_ids=matched_sources,
            )

            for direction_seed, v in random_dirs.items():
                deltas = {}
                for sid in base_ids:
                    plan = plans[sid]
                    h_base = activations[int(layer)][sid]
                    h_match = activations[int(layer)][plan.matched_source_id]
                    q_gap = coordinate_value(h_match, u) - coordinate_value(
                        h_base, u
                    )
                    deltas[sid] = float(alpha) * abs(q_gap) * v
                evaluate_condition(
                    condition="random_direction",
                    alpha=float(alpha),
                    deltas=deltas,
                    source_ids=matched_sources,
                    direction_seed=int(direction_seed),
                )
            for direction_seed, v in ortho_dirs.items():
                deltas = {}
                for sid in base_ids:
                    plan = plans[sid]
                    h_base = activations[int(layer)][sid]
                    h_match = activations[int(layer)][plan.matched_source_id]
                    q_gap = coordinate_value(h_match, u) - coordinate_value(
                        h_base, u
                    )
                    deltas[sid] = float(alpha) * abs(q_gap) * v
                evaluate_condition(
                    condition="orthogonal_random",
                    alpha=float(alpha),
                    deltas=deltas,
                    source_ids=matched_sources,
                    direction_seed=int(direction_seed),
                )

            status.update(
                progress={
                    "completed_alpha": float(alpha),
                    "n_raw_rows": len(raw_rows),
                    "n_trace_rows": len(trace_rows),
                }
            )

        raw_df = pd.DataFrame(raw_rows)
        trace_df = pd.DataFrame(trace_rows)
        save_table(raw_df, run_dir / "intervention_rows.parquet")
        save_table(trace_df, run_dir / "trace_rows.parquet")

        aggregates = aggregate_intervention_rows(
            raw_df,
            n_bootstraps=int(cfg.statistics.bootstrap_samples),
            confidence_level=float(cfg.statistics.confidence_level),
            seed=int(cfg.reproducibility.bootstrap_seed),
        )
        medians = (
            raw_df.groupby(["condition", "alpha"], as_index=False)[
                "delta_margin_toward_expected"
            ]
            .median()
            .rename(
                columns={
                    "delta_margin_toward_expected": (
                        "median_delta_margin_toward_expected"
                    )
                }
            )
        )
        aggregates = aggregates.merge(
            medians, on=["condition", "alpha"], how="left"
        )
        save_table(aggregates, run_dir / "aggregate_metrics.parquet")

        contrasts: list[dict[str, Any]] = []
        for alpha in alphas:
            if np.isclose(alpha, 0.0):
                continue
            for control in (
                "random_direction",
                "orthogonal_random",
                "same_label_coordinate",
                "shuffled_coordinate",
            ):
                contrasts.append(
                    paired_control_contrast(
                        raw_df,
                        treatment="truth_coordinate",
                        control=control,
                        alpha=float(alpha),
                        n_bootstraps=int(cfg.statistics.bootstrap_samples),
                        confidence_level=float(cfg.statistics.confidence_level),
                        seed=(
                            int(cfg.reproducibility.bootstrap_seed)
                            + len(contrasts) * 101
                        ),
                    )
                )
        save_table(
            pd.DataFrame(contrasts), run_dir / "control_contrasts.parquet"
        )

        truth_dose = dose_response_summary(raw_df, "truth_coordinate")
        truth_alpha1 = aggregates[
            (aggregates["condition"] == "truth_coordinate")
            & np.isclose(aggregates["alpha"].to_numpy(float), 1.0)
        ]
        truth_alpha1_effect = (
            float(truth_alpha1.iloc[0]["mean_delta_margin_toward_expected"])
            if len(truth_alpha1)
            else None
        )
        intervention_probe = next(
            m for m in probe_metrics if int(m["layer"]) == int(layer)
        )
        summary = {
            "status": "complete",
            "version": E01A_VERSION,
            "model_id": cfg.model.id,
            "resolved_revision": resolved_revision,
            "profile": str(profile),
            "layer": int(layer),
            "trace_layers": traces,
            "alphas": list(map(float, alphas)),
            "random_directions": int(n_random),
            "n_pairs": int(len(selected_pair_ids)),
            "n_base_examples": int(len(base_ids)),
            "intervention_layer_probe_auroc": float(
                intervention_probe.get("auroc")
            ),
            "candidate_token_ids": candidate_lists,
            "first_token_metric_only": bool(
                len(candidate_lists[0]) != 1 or len(candidate_lists[1]) != 1
            ),
            "alpha0_max_logit_deviation": float(alpha0_max_logit_dev),
            "target_fidelity_max_abs": float(max_fidelity),
            "control_norm_match_max_abs": float(max_norm_mismatch),
            "truth_alpha1_mean_effect": truth_alpha1_effect,
            "truth_dose_response": truth_dose,
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "e01a_metrics.json")
        _write_summary(run_dir, summary)
        status.complete("E01A discovery run complete")
        manifest.finish(
            runs_summary=[
                {
                    "profile": str(profile),
                    "n_pairs": int(len(selected_pair_ids)),
                    "n_rows": int(len(raw_df)),
                    "truth_alpha1_mean_effect": truth_alpha1_effect,
                }
            ]
        )
        return run_dir
    except Exception as exc:
        logger.exception("E01A failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish(
            runs_summary=[{"status": "failed", "error": repr(exc)}]
        )
        raise
