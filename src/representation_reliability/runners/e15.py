"""E15 temporal causal half-life runner.

Executes the staged design of ``docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_DESIGN.md``
through the frozen protocol in
``docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_PROTOCOL.md``:

```text
stage 0  frozen stateful task, exact state-label validation
stage 1  D(k) and B(k) with no intervention
stage 2  intervention smoke at horizons {1, 4, 16}, engineering gate only
stage 3  full horizon curve, all arms, half-life estimation
stage 3b secondary distractor-density contrast (H15.3)
stage 4  replication on Qwen3-0.6B, only if the Stage 4 gate passes
```

The question is whether a state variable stays decodable at a future decision
point after the represented state has stopped governing that decision. Nothing
here selects a layer, a carrier, a horizon or an arm on an outcome.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.stateful_console import DENIED, GRANTED, LONG_DISTRACTORS, SHORT_DISTRACTORS
from ..interventions.setpoint import (
    norm_matched_direction_delta,
    setpoint_fidelity_tolerances,
    setpoint_identity_diagnostics,
    source_free_setpoint_delta,
)
from ..interventions.truth_coordinate import (
    coordinate_value,
    normalized_direction,
    random_unit_direction,
)
from ..metrics.causal import (
    cluster_bootstrap_mean_ci,
    counterfactual_outcome,
    margin_toward_label,
)
from ..metrics.setpoint import validation_setpoint_targets
from ..metrics.temporal_half_life import (
    half_life,
    horizon_condition_summary,
    paired_horizon_contrast,
    relative_curve,
    shuffled_decision_null,
)
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.status import StatusFile, atomic_write_json
from .e00c import candidate_first_token_ids
from .e01a import _prediction
from .e15_support import (
    BOOTSTRAP_SEED,
    CARRIER_LAYER,
    CORPUS_SPECS,
    DIRECTION_SEED_BASE,
    E15_VERSION,
    HORIZONS,
    K0,
    MIN_BEHAVIOR,
    MIN_DECODABILITY,
    N_ORTHOGONAL_DIRECTIONS,
    N_RANDOM_DIRECTIONS,
    NO_OP_TOLERANCE,
    PERMUTATION_SEED,
    PROBE_SEED,
    PROPAGATION_LAYERS,
    RANDOM_LABEL_BAND,
    SITE,
    SMOKE_HORIZONS,
    SMOKE_PAIRS,
    behavior_metrics,
    campaign_dir,
    corpus_bundle,
    fit_site_probe,
    horizon_view,
    probe_metrics_record,
    repo_root,
    resolve_site_table,
    run_clean_batches,
    run_edit_batches,
    selected_margin,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "qwen3_1.7b"
REPLICATION_MODEL = "qwen3_0.6b"          # frozen before any causal outcome
DECISION_LAYERS: tuple[int, ...] = (CARRIER_LAYER, *PROPAGATION_LAYERS)
N_PERMUTATIONS = 1000

CONTROL_ARMS = (
    "random_normmatched",
    "orthogonal_normmatched",
    "irrelevant_state",
    "late_position",
)


# --------------------------------------------------------------------- setup
def _config(model_name: str):
    root = repo_root()
    cfg, provenance = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / f"configs/models/{model_name}.yaml",
        experiment_path=root / "configs/experiments/E15_temporal_causal_half_life.yaml",
        overrides=(),
    )
    return cfg, provenance


def _stage_dir(stage: str, model_name: str) -> Path:
    path = campaign_dir() / f"{stage}_{model_name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _start_run(stage: str, model_name: str, cfg, provenance, extra: dict[str, Any]):
    run_dir = _stage_dir(stage, model_name)
    status = StatusFile.create(run_dir, run_id=f"E15-{stage}-{model_name}", experiment_id="E15")
    manifest = RunManifest(run_dir)
    manifest.set_start(
        config_hash(cfg),
        {**(provenance or {}), "e15_version": E15_VERSION, "stage": stage, **extra},
        {
            "probe": PROBE_SEED,
            "direction_base": DIRECTION_SEED_BASE,
            "bootstrap": BOOTSTRAP_SEED,
            "permutation": PERMUTATION_SEED,
            "corpus": {name: seed for name, _n, seed in CORPUS_SPECS},
        },
    )
    save_resolved_config(cfg, run_dir / "resolved_config.yaml", provenance)
    return run_dir, status, manifest


def _release(adapter) -> None:
    try:
        del adapter
    except Exception:  # pragma: no cover - defensive
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _episode_key(row: Any) -> str:
    return f"{row['split']}|{int(row['episode_index'])}|{int(row['target_label'])}"


# ------------------------------------------------------------------- stage 0
def run_e15_stage0(model_name: str = PRIMARY_MODEL) -> Path:
    """Freeze the task and validate exact state labels (protocol 1.6)."""
    from transformers import AutoTokenizer

    cfg, provenance = _config(model_name)
    run_dir, status, manifest = _start_run("stage0", model_name, cfg, provenance, {})
    try:
        samples, frame, stats, labels, by_id = corpus_bundle()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)

        granted_ids = tokenizer(" " + GRANTED, add_special_tokens=False)["input_ids"]
        denied_ids = tokenizer(" " + DENIED, add_special_tokens=False)["input_ids"]
        parity = len(granted_ids) == len(denied_ids)

        # Carrier index invariance across horizons, and twin token-length parity.
        class _TokenizerAdapter:
            pass

        holder = _TokenizerAdapter()
        holder.tokenizer = tokenizer  # type: ignore[attr-defined]
        sites = resolve_site_table(holder, samples)

        index_by_episode: dict[str, dict[int, int]] = {}
        length_by_episode: dict[str, dict[int, int]] = {}
        for row in frame.to_dict("records"):
            sid = str(row["sample_id"])
            key = _episode_key(row)
            horizon = int(row["horizon"])
            index_by_episode.setdefault(key, {})[horizon] = int(
                sites[sid]["carrier"]["token_index"]
            )
            length_by_episode.setdefault(key, {})[horizon] = int(
                sites[sid]["carrier"]["sequence_length"]
            )
        invariant = all(len(set(v.values())) == 1 for v in index_by_episode.values())

        pair_length_mismatch = 0
        for pair_id, block in frame.groupby("pair_id"):
            ids = sorted(block["sample_id"].astype(str).tolist())
            lengths = {sites[sid]["decision"]["sequence_length"] for sid in ids}
            if len(lengths) != 1:
                pair_length_mismatch += 1

        gate = {
            "state_word_token_parity": bool(parity),
            "granted_token_ids": list(map(int, granted_ids)),
            "denied_token_ids": list(map(int, denied_ids)),
            "carrier_index_horizon_invariant": bool(invariant),
            "twin_token_length_mismatches": int(pair_length_mismatch),
            "n_samples": int(len(frame)),
            "n_pairs": int(frame["pair_id"].nunique()),
            "confirmation_split_exists": False,
        }
        gate["passed"] = bool(
            parity and invariant and pair_length_mismatch == 0 and len(frame) > 0
        )

        manifest.update_dataset_info(
            split_hash=dataset_split_hash(
                dict(zip(frame["sample_id"].astype(str), frame["split"].astype(str)))
            ),
            prompt_hash_sample=prompt_hash(str(frame["prompt"].iloc[0])),
        )
        save_json(stats, run_dir / "corpus_statistics.json")
        save_json(gate, run_dir / "stage0_gate.json")
        save_table(
            frame[
                [
                    "sample_id", "pair_id", "split", "horizon", "target_label",
                    "episode_index", "target_state", "other_state", "target_slot",
                    "n_steps",
                ]
            ],
            run_dir / "corpus_index.parquet",
        )
        example = frame.iloc[0]
        (run_dir / "example_prompt.txt").write_text(str(example["prompt"]), encoding="utf-8")
        manifest.finish([{"stage": "stage0", "gate": gate}])
        if not gate["passed"]:
            status.fail("E15 Stage 0 gate G0 failed")
            raise RuntimeError(f"E15 Stage 0 gate failed: {gate}")
        status.complete("Stage 0 complete: task frozen and state labels validated")
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise


# ------------------------------------------------------------------- stage 1
def run_e15_stage1(model_name: str = PRIMARY_MODEL) -> Path:
    """Establish D(k) and B(k) with no intervention (protocol section 3)."""
    cfg, provenance = _config(model_name)
    run_dir, status, manifest = _start_run("stage1", model_name, cfg, provenance, {})
    adapter = None
    try:
        samples, frame, _stats, labels, by_id = corpus_bundle()
        adapter = load_adapter(cfg)
        manifest.update_model_info(
            id=adapter.display_model_id,
            dtype=str(cfg.model.dtype),
            num_layers=adapter.num_layers,
            hidden_size=adapter.hidden_size,
            resolved_native_modules={
                f"{SITE}:{layer}": adapter.resolve_site(SITE, layer).native_module_name
                for layer in DECISION_LAYERS
            },
            notes={"revisions": adapter.resolved_revisions()},
        )
        if max(DECISION_LAYERS) >= adapter.num_layers:
            raise RuntimeError(
                f"frozen E15 layers {DECISION_LAYERS} exceed {adapter.num_layers} blocks"
            )
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        sites = resolve_site_table(adapter, samples)
        save_json(
            {sid: sites[sid] for sid in list(sites)[:20]},
            run_dir / "resolved_sites_sample.json",
        )

        probe_records: list[dict[str, Any]] = []
        behavior_rows: list[dict[str, Any]] = []
        reference_arrays: dict[str, np.ndarray] = {}
        carrier_reference: dict[str, np.ndarray] = {}
        carrier_drift: list[float] = []
        setpoints: dict[str, Any] | None = None
        d_primary: dict[int, float] = {}
        d_carry: dict[int, float] = {}
        d_layer: dict[str, float] = {}

        for horizon in HORIZONS:
            ids = (
                frame[frame["horizon"].astype(int) == int(horizon)]["sample_id"]
                .astype(str)
                .sort_values()
                .tolist()
            )
            clean = run_clean_batches(
                adapter, by_id, ids, sites,
                output_token_ids=output_token_ids,
                decision_layers=PROPAGATION_LAYERS,
                batch_size=int(cfg.runtime.batch_size),
            )
            states = {sid: clean[sid]["sites"] for sid in ids}

            # Carrier horizon-invariance: identical prefix must give identical state.
            row_key = {
                str(r["sample_id"]): _episode_key(r)
                for r in frame[frame["horizon"].astype(int) == int(horizon)].to_dict("records")
            }
            for sid in ids:
                key = row_key[sid]
                vector = states[sid]["carrier"]
                if key not in carrier_reference:
                    carrier_reference[key] = vector.copy()
                else:
                    reference = carrier_reference[key]
                    denominator = max(float(np.linalg.norm(reference)), 1e-12)
                    carrier_drift.append(
                        float(np.linalg.norm(vector - reference) / denominator)
                    )

            site_keys = ["late_carrier"] + [f"decision_l{layer}" for layer in DECISION_LAYERS]
            if int(horizon) == K0:
                site_keys = ["carrier", *site_keys]
            for site_key in site_keys:
                result = fit_site_probe(
                    states, frame, labels,
                    site_key=site_key, horizon=int(horizon),
                    c_grid=cfg.probe.C_grid, seed=PROBE_SEED,
                )
                probe_records.append(probe_metrics_record(result))
                reference_arrays[f"dir_h{int(horizon)}_{site_key}"] = np.asarray(
                    result["direction"], dtype=np.float64
                )
                auroc = float(result["test_metrics"]["auroc"])
                if site_key == f"decision_l{CARRIER_LAYER}":
                    d_primary[int(horizon)] = auroc
                elif site_key == "late_carrier":
                    d_carry[int(horizon)] = auroc
                elif site_key == "carrier":
                    d_layer["carrier"] = auroc
                else:
                    d_layer[f"h{int(horizon)}_{site_key}"] = auroc

                # Frozen source-free targets come from the carrier probe only.
                if site_key == "carrier" and int(horizon) == K0:
                    u = normalized_direction(result["direction"])
                    val_ids = horizon_view(frame, "validation", K0)["sample_id"].astype(str).tolist()
                    coordinates = [coordinate_value(states[sid]["carrier"], u) for sid in val_ids]
                    val_margins = [
                        selected_margin(clean[sid]["selected_logits"]) for sid in val_ids
                    ]
                    setpoints = validation_setpoint_targets(
                        coordinates,
                        [int(labels[sid]) for sid in val_ids],
                        val_margins,
                    )
                    reference_arrays["carrier_direction"] = np.asarray(u, dtype=np.float64)

            for split in ("validation", "discovery_test"):
                split_ids = horizon_view(frame, split, int(horizon))["sample_id"].astype(str).tolist()
                behavior_rows.append(
                    {
                        "horizon": int(horizon),
                        "split": split,
                        **behavior_metrics(clean, labels, split_ids),
                    }
                )
            del clean, states
            gc.collect()

        if setpoints is None:
            raise RuntimeError("carrier setpoint reference was never constructed")

        behavior = pd.DataFrame(behavior_rows)
        behavior["margin_auroc"] = [
            row["margin_metrics"].get("auroc") for row in behavior_rows
        ]
        behavior = behavior.drop(columns=["margin_metrics"])
        test_behavior = behavior[behavior["split"] == "discovery_test"].set_index("horizon")
        val_behavior = behavior[behavior["split"] == "validation"].set_index("horizon")

        b_by_horizon = {int(k): float(test_behavior.loc[k, "accuracy"]) for k in HORIZONS}
        sigma_m_val = {int(k): float(val_behavior.loc[k, "margin_sd"]) for k in HORIZONS}

        # G1c truncation is decided on non-causal behaviour only.
        interpretable: list[int] = []
        for horizon in HORIZONS:
            if b_by_horizon[int(horizon)] >= MIN_BEHAVIOR:
                interpretable.append(int(horizon))
            else:
                break

        random_label_aurocs = [
            value
            for record in probe_records
            for value in record["random_label_auroc"]
            if value is not None
        ]
        gate = {
            "G1a_carrier_decodability": d_layer.get("carrier"),
            "G1a_passed": bool((d_layer.get("carrier") or 0.0) >= MIN_DECODABILITY),
            "G1b_decision_decodability_k0": d_primary.get(K0),
            "G1b_passed": bool((d_primary.get(K0) or 0.0) >= MIN_DECODABILITY),
            "G1c_behavior_by_horizon": b_by_horizon,
            "G1c_min_behavior": MIN_BEHAVIOR,
            "G1c_interpretable_horizons": interpretable,
            "G1c_passed": bool(interpretable and interpretable[0] == K0),
            "G1d_random_label_auroc_range": [
                float(min(random_label_aurocs)), float(max(random_label_aurocs)),
            ],
            "G1d_passed": bool(
                RANDOM_LABEL_BAND[0] <= min(random_label_aurocs)
                and max(random_label_aurocs) <= RANDOM_LABEL_BAND[1]
            ),
            "carrier_horizon_invariance_max_relative_l2": (
                float(max(carrier_drift)) if carrier_drift else 0.0
            ),
        }
        gate["passed"] = bool(
            gate["G1a_passed"] and gate["G1b_passed"]
            and gate["G1c_passed"] and gate["G1d_passed"]
        )

        summary = {
            "model": adapter.display_model_id,
            "site": SITE,
            "carrier_layer": CARRIER_LAYER,
            "horizons": list(map(int, HORIZONS)),
            "D_carrier": d_layer.get("carrier"),
            "D_primary_by_horizon": d_primary,
            "D_carry_by_horizon": d_carry,
            "D_other_layers": d_layer,
            "B_by_horizon": b_by_horizon,
            "sigma_margin_validation_by_horizon": sigma_m_val,
            "validation_setpoints": setpoints,
            "gate": gate,
        }
        np.savez(run_dir / "stage1_reference.npz", **reference_arrays)
        save_json(probe_records, run_dir / "probe_metrics.json")
        save_table(behavior, run_dir / "behavior_by_horizon.parquet")
        save_json(summary, run_dir / "stage1_summary.json")
        manifest.finish([{"stage": "stage1", "gate": gate}])
        if not gate["passed"]:
            status.fail("E15 Stage 1 gate G1 failed")
            raise RuntimeError(f"E15 Stage 1 gate failed: {gate}")
        status.complete("Stage 1 complete: D(k) and B(k) established")
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            _release(adapter)


# --------------------------------------------------------- intervention core
def _load_stage1(model_name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stage1 = _stage_dir("stage1", model_name)
    summary_path = stage1 / "stage1_summary.json"
    if not summary_path.exists():
        raise RuntimeError("E15 Stage 1 must complete before any intervention stage")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(stage1 / "stage1_reference.npz") as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    return summary, arrays


def _arm_plan(hidden_size: int, u: np.ndarray) -> list[dict[str, Any]]:
    """Frozen arm list (protocol section 6). Order is fixed, not outcome-driven."""
    plan: list[dict[str, Any]] = [
        {"condition": "no_op", "edit_site": "carrier", "kind": "zero", "direction_index": -1},
        {"condition": "setpoint", "edit_site": "carrier", "kind": "setpoint", "direction_index": -1},
    ]
    for index in range(N_RANDOM_DIRECTIONS):
        plan.append(
            {
                "condition": "random_normmatched",
                "edit_site": "carrier",
                "kind": "norm_matched",
                "direction_index": index,
                "direction": random_unit_direction(
                    hidden_size, DIRECTION_SEED_BASE + index
                ),
            }
        )
    for index in range(N_ORTHOGONAL_DIRECTIONS):
        plan.append(
            {
                "condition": "orthogonal_normmatched",
                "edit_site": "carrier",
                "kind": "norm_matched",
                "direction_index": index,
                "direction": random_unit_direction(
                    hidden_size, DIRECTION_SEED_BASE + 100 + index, orthogonal_to=u
                ),
            }
        )
    plan.append(
        {
            "condition": "irrelevant_state",
            "edit_site": "irrelevant_carrier",
            "kind": "setpoint",
            "direction_index": -1,
        }
    )
    plan.append(
        {
            "condition": "late_position",
            "edit_site": "late_carrier",
            "kind": "setpoint",
            "direction_index": -1,
        }
    )
    return plan


def _intervention_sweep(
    adapter,
    cfg,
    *,
    frame: pd.DataFrame,
    by_id: dict[str, Any],
    labels: dict[str, int],
    sites: dict[str, Any],
    output_token_ids: list[int],
    u: np.ndarray,
    setpoints: dict[str, Any],
    horizons: list[int],
    pair_limit: int | None,
    arms: list[dict[str, Any]] | None = None,
    decision_directions: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every frozen arm at every requested horizon; return raw evidence."""
    plan = arms if arms is not None else _arm_plan(adapter.hidden_size, u)
    tolerances = setpoint_fidelity_tolerances(str(cfg.model.dtype))
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "tolerances": tolerances,
        "no_op_max_margin_deviation": 0.0,
        "max_projection_relative_deviation": 0.0,
        "max_orthogonal_relative_deviation": 0.0,
        "max_norm_match_relative_deviation": 0.0,
        "hook_leak_detected": False,
        "horizons": list(map(int, horizons)),
        "arms": sorted({str(item["condition"]) for item in plan}),
    }

    for horizon in horizons:
        view = horizon_view(frame, "discovery_test", int(horizon))
        pair_ids = sorted(view["pair_id"].astype(str).unique().tolist())
        if pair_limit is not None:
            pair_ids = pair_ids[: int(pair_limit)]
        keep = set(pair_ids)
        ids = (
            view[view["pair_id"].astype(str).isin(keep)]["sample_id"]
            .astype(str)
            .tolist()
        )
        meta = view.set_index("sample_id")

        clean = run_clean_batches(
            adapter, by_id, ids, sites,
            output_token_ids=output_token_ids,
            decision_layers=PROPAGATION_LAYERS,
            batch_size=int(cfg.runtime.batch_size),
        )
        clean_margin = {sid: selected_margin(clean[sid]["selected_logits"]) for sid in ids}
        q_target = {
            sid: (
                float(setpoints["q0_star"])
                if int(labels[sid]) == 1
                else float(setpoints["q1_star"])
            )
            for sid in ids
        }
        treatment_delta = {
            sid: source_free_setpoint_delta(clean[sid]["sites"]["carrier"], u, q_target[sid])
            for sid in ids
        }

        for arm in plan:
            condition = str(arm["condition"])
            edit_site = str(arm["edit_site"])
            kind = str(arm["kind"])
            deltas: dict[str, np.ndarray] = {}
            targets: dict[str, float | None] = {}
            for sid in ids:
                base = clean[sid]["sites"][edit_site]
                if kind == "zero":
                    deltas[sid] = np.zeros_like(base)
                    targets[sid] = None
                elif kind == "setpoint":
                    deltas[sid] = source_free_setpoint_delta(base, u, q_target[sid])
                    targets[sid] = q_target[sid]
                elif kind == "norm_matched":
                    deltas[sid] = norm_matched_direction_delta(
                        treatment_delta[sid], arm["direction"]
                    )
                    targets[sid] = None
                else:  # pragma: no cover - frozen plan only
                    raise ValueError(f"unknown arm kind {kind!r}")

            edited = run_edit_batches(
                adapter, by_id, ids, sites,
                edit_site=edit_site,
                deltas_by_id=deltas,
                output_token_ids=output_token_ids,
                propagation_layers=PROPAGATION_LAYERS,
                batch_size=int(cfg.runtime.batch_size),
            )

            for sid in ids:
                base = clean[sid]["sites"][edit_site]
                after_state = edited[sid]["edited_carrier_state"]
                delta = np.asarray(deltas[sid], dtype=np.float64)
                margin_before = clean_margin[sid]
                margin_after = selected_margin(edited[sid]["selected_logits"])
                label = int(labels[sid])
                expected = 1 - label
                oriented_before = margin_toward_label(margin_before, expected)
                oriented_after = margin_toward_label(margin_after, expected)
                prediction_before = _prediction(margin_before)
                prediction_after = _prediction(margin_after)
                outcome = counterfactual_outcome(
                    prediction_before, prediction_after, expected
                )
                delta_norm = float(np.linalg.norm(delta))
                activation_norm = float(np.linalg.norm(base))

                record: dict[str, Any] = {
                    "horizon": int(horizon),
                    "condition": condition,
                    "direction_index": int(arm["direction_index"]),
                    "edit_site": edit_site,
                    "base_sample_id": sid,
                    "pair_id": str(meta.loc[sid, "pair_id"]),
                    "episode_index": int(meta.loc[sid, "episode_index"]),
                    "target_label": label,
                    "expected_label": expected,
                    "margin_before": margin_before,
                    "margin_after": margin_after,
                    "delta_margin_raw": float(margin_after - margin_before),
                    "margin_toward_expected_before": oriented_before,
                    "margin_toward_expected_after": oriented_after,
                    "delta_margin_toward_expected": float(oriented_after - oriented_before),
                    "prediction_before": prediction_before,
                    "prediction_after": prediction_after,
                    "expected_label_after": outcome["expected_label_after"],
                    "counterfactual_flip": outcome["counterfactual_flip"],
                    "delta_norm": delta_norm,
                    "activation_norm": activation_norm,
                    "delta_over_activation_norm": delta_norm / max(activation_norm, 1e-12),
                    "q_base": coordinate_value(base, u),
                    "q_target": targets[sid],
                    "q_after": coordinate_value(after_state, u),
                }

                if kind == "setpoint":
                    fidelity = setpoint_identity_diagnostics(
                        base, after_state, u, float(targets[sid])
                    )
                    record["projection_relative_deviation"] = fidelity[
                        "projection_relative_deviation"
                    ]
                    record["orthogonal_relative_deviation"] = fidelity[
                        "orthogonal_relative_deviation"
                    ]
                    diagnostics["max_projection_relative_deviation"] = max(
                        diagnostics["max_projection_relative_deviation"],
                        abs(fidelity["q_after"] - fidelity["q_target"])
                        / max(float(setpoints["sigma_q_validation"]), 1e-12),
                    )
                    diagnostics["max_orthogonal_relative_deviation"] = max(
                        diagnostics["max_orthogonal_relative_deviation"],
                        fidelity["orthogonal_relative_deviation"],
                    )
                else:
                    record["projection_relative_deviation"] = None
                    record["orthogonal_relative_deviation"] = None

                if kind == "zero":
                    diagnostics["no_op_max_margin_deviation"] = max(
                        diagnostics["no_op_max_margin_deviation"],
                        abs(margin_after - margin_before),
                    )
                if kind == "norm_matched":
                    reference = float(np.linalg.norm(treatment_delta[sid]))
                    diagnostics["max_norm_match_relative_deviation"] = max(
                        diagnostics["max_norm_match_relative_deviation"],
                        abs(delta_norm - reference) / max(reference, 1e-12),
                    )

                for layer in PROPAGATION_LAYERS:
                    clean_state = clean[sid]["sites"][f"decision_l{layer}"]
                    edited_state = edited[sid]["propagation"][int(layer)]
                    denominator = max(float(np.linalg.norm(clean_state)), 1e-12)
                    record[f"p_norm_l{layer}"] = float(
                        np.linalg.norm(edited_state - clean_state) / denominator
                    )
                    if decision_directions is not None:
                        key = f"dir_h{int(horizon)}_decision_l{layer}"
                        direction = decision_directions.get(key)
                        if direction is not None:
                            unit = normalized_direction(direction)
                            record[f"p_q_l{layer}"] = float(
                                coordinate_value(edited_state, unit)
                                - coordinate_value(clean_state, unit)
                            )
                rows.append(record)

            del edited
        del clean
        gc.collect()

    frame_rows = pd.DataFrame(rows)
    if not np.isfinite(frame_rows["delta_margin_toward_expected"].to_numpy(float)).all():
        raise RuntimeError("non-finite intervention evidence produced")
    return frame_rows, diagnostics


def _hook_leak(adapter) -> bool:
    from ..adapters.intervention import resid_post_hook_count

    return any(
        resid_post_hook_count(adapter, layer=layer) for layer in DECISION_LAYERS
    )


# ------------------------------------------------------------------- stage 2
def run_e15_stage2(model_name: str = PRIMARY_MODEL) -> Path:
    """Bounded intervention smoke; gate G2 is engineering-only (protocol 7)."""
    cfg, provenance = _config(model_name)
    summary1, arrays = _load_stage1(model_name)
    run_dir, status, manifest = _start_run(
        "stage2", model_name, cfg, provenance,
        {"stage1_gate": summary1["gate"], "smoke_horizons": list(SMOKE_HORIZONS)},
    )
    adapter = None
    try:
        samples, frame, _stats, labels, by_id = corpus_bundle()
        adapter = load_adapter(cfg)
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        sites = resolve_site_table(adapter, samples)
        u = normalized_direction(arrays["carrier_direction"])
        rows, diagnostics = _intervention_sweep(
            adapter, cfg,
            frame=frame, by_id=by_id, labels=labels, sites=sites,
            output_token_ids=output_token_ids, u=u,
            setpoints=summary1["validation_setpoints"],
            horizons=[int(k) for k in SMOKE_HORIZONS],
            pair_limit=SMOKE_PAIRS,
            decision_directions=arrays,
        )
        diagnostics["hook_leak_detected"] = _hook_leak(adapter)
        tolerances = diagnostics["tolerances"]
        gate = {
            "no_op_max_margin_deviation": diagnostics["no_op_max_margin_deviation"],
            "no_op_tolerance": NO_OP_TOLERANCE,
            "no_op_passed": bool(
                diagnostics["no_op_max_margin_deviation"] <= NO_OP_TOLERANCE
            ),
            "hook_leak_detected": diagnostics["hook_leak_detected"],
            "projection_validation_sigma": diagnostics["max_projection_relative_deviation"],
            "orthogonal_relative": diagnostics["max_orthogonal_relative_deviation"],
            "norm_match_relative": diagnostics["max_norm_match_relative_deviation"],
            "tolerances": tolerances,
        }
        gate["fidelity_passed"] = bool(
            diagnostics["max_projection_relative_deviation"]
            <= tolerances["projection_validation_sigma"]
            and diagnostics["max_orthogonal_relative_deviation"]
            <= tolerances["orthogonal_relative"]
        )
        gate["norm_match_passed"] = bool(
            diagnostics["max_norm_match_relative_deviation"] <= 1e-6
        )
        gate["passed"] = bool(
            gate["no_op_passed"]
            and not gate["hook_leak_detected"]
            and gate["fidelity_passed"]
            and gate["norm_match_passed"]
        )
        save_table(rows, run_dir / "smoke_rows.parquet")
        save_json(diagnostics, run_dir / "smoke_diagnostics.json")
        save_json(gate, run_dir / "stage2_gate.json")
        manifest.finish([{"stage": "stage2", "gate": gate}])
        if not gate["passed"]:
            status.fail("E15 Stage 2 engineering gate G2 failed")
            raise RuntimeError(f"E15 Stage 2 gate failed: {gate}")
        status.complete("Stage 2 complete: intervention contract validated")
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            _release(adapter)


# ------------------------------------------------------------------- stage 3
def _curves_and_gate(
    rows: pd.DataFrame,
    summary1: dict[str, Any],
    horizons: list[int],
    *,
    bootstraps: int,
    confidence: float,
) -> dict[str, Any]:
    """Build D/C curves, half-lives, contrasts, nulls and the Stage 4 gate."""
    summary = horizon_condition_summary(
        rows, n_bootstraps=bootstraps, confidence_level=confidence, seed=BOOTSTRAP_SEED
    )
    treat = summary[summary["condition"] == "setpoint"].set_index("horizon")
    sigma_m = {int(k): float(v) for k, v in summary1["sigma_margin_validation_by_horizon"].items()}
    d_primary = {int(k): float(v) for k, v in summary1["D_primary_by_horizon"].items()}

    c_raw = [float(treat.loc[k, "mean_effect"]) for k in horizons]
    c_z = [float(treat.loc[k, "mean_effect"]) / max(sigma_m[k], 1e-12) for k in horizons]
    d_values = [d_primary[k] for k in horizons]

    k0_ci = (
        float(treat.loc[horizons[0], "ci_low"]),
        float(treat.loc[horizons[0], "ci_high"]),
    )
    k0_excludes_zero = bool(k0_ci[0] > 0.0 or k0_ci[1] < 0.0)

    curves: dict[str, Any] = {}
    for name, values, floor in (
        ("C_raw", c_raw, 0.0),
        ("C_z", c_z, 0.0),
        ("D_primary", d_values, 0.5),
    ):
        curve = relative_curve(horizons, values, floor=floor)
        baseline_ci = (
            k0_excludes_zero if name.startswith("C") else bool(d_values[0] - 0.5 > 0.0)
        )
        curves[name] = {
            **curve,
            "half_life": half_life(
                horizons,
                curve["relative"],
                baseline_positive=curve["baseline_positive"],
                baseline_ci_excludes_zero=baseline_ci,
            ),
        }

    contrasts = []
    for horizon in horizons:
        for control in CONTROL_ARMS:
            contrasts.append(
                paired_horizon_contrast(
                    rows, horizon=int(horizon), treatment="setpoint", control=control,
                    n_bootstraps=bootstraps, confidence_level=confidence,
                    seed=BOOTSTRAP_SEED + 7 * int(horizon) + len(control),
                )
            )

    nulls = {}
    for horizon in horizons:
        block = rows[
            (rows["horizon"].astype(int) == int(horizon))
            & (rows["condition"] == "setpoint")
        ]
        nulls[str(int(horizon))] = shuffled_decision_null(
            block["delta_margin_raw"].to_numpy(float),
            block["expected_label"].to_numpy(int),
            n_permutations=N_PERMUTATIONS,
            seed=PERMUTATION_SEED + int(horizon),
        )

    hc_raw = curves["C_raw"]["half_life"]
    hc_z = curves["C_z"]["half_life"]
    hd = curves["D_primary"]["half_life"]

    d_rel_at_hc: float | None = None
    if hc_raw["status"] == "estimated":
        d_rel_at_hc = float(
            np.interp(
                float(hc_raw["value"]),
                np.asarray(horizons, dtype=float),
                np.asarray(curves["D_primary"]["relative"], dtype=float),
            )
        )

    k0 = horizons[0]
    k0_contrasts = [c for c in contrasts if c["horizon"] == k0]
    gate = {
        "G3.1_k0_effect": float(treat.loc[k0, "mean_effect"]),
        "G3.1_k0_ci": list(k0_ci),
        "G3.1_k0_ci_excludes_zero": k0_excludes_zero,
        "G3.1_control_contrasts": {
            c["control"]: {
                "mean_difference": c["mean_difference"],
                "ci": [c["ci_low"], c["ci_high"]],
                "ci_excludes_zero": c["ci_excludes_zero"],
                "treatment_greater": bool(c["ci_low"] > 0.0),
            }
            for c in k0_contrasts
        },
        "G3.1_passed": bool(
            k0_excludes_zero
            and float(treat.loc[k0, "mean_effect"]) > 0.0
            and all(c["ci_low"] > 0.0 for c in k0_contrasts)
        ),
        "G3.2_H_C_raw": hc_raw,
        "G3.2_H_C_z": hc_z,
        "G3.2_passed": bool(
            hc_raw["status"] == "estimated" and hc_z["status"] == "estimated"
        ),
        "G3.3_H_D": hd,
        "G3.3_D_rel_at_H_C": d_rel_at_hc,
        "G3.3_passed": bool(d_rel_at_hc is not None and d_rel_at_hc >= 0.90),
        "G3.5_null_at_k0": nulls[str(int(k0))],
        "G3.5_passed": bool(nulls[str(int(k0))]["exceeds_null"]),
    }
    gate["passed"] = bool(
        gate["G3.1_passed"] and gate["G3.2_passed"]
        and gate["G3.3_passed"] and gate["G3.5_passed"]
    )
    return {
        "horizon_condition_summary": summary,
        "curves": curves,
        "contrasts": contrasts,
        "shuffled_decision_nulls": nulls,
        "gate": gate,
        "C_raw_by_horizon": {int(k): v for k, v in zip(horizons, c_raw)},
        "C_z_by_horizon": {int(k): v for k, v in zip(horizons, c_z)},
        "D_primary_by_horizon": {int(k): v for k, v in zip(horizons, d_values)},
    }


def run_e15_stage3(model_name: str = PRIMARY_MODEL) -> Path:
    """Full horizon curve for one model (protocol section 8)."""
    cfg, provenance = _config(model_name)
    summary1, arrays = _load_stage1(model_name)
    stage2_gate = _stage_dir("stage2", model_name) / "stage2_gate.json"
    if not stage2_gate.exists():
        raise RuntimeError("E15 Stage 2 must complete before Stage 3")
    if not json.loads(stage2_gate.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("E15 Stage 2 gate did not pass; Stage 3 is not authorized")

    horizons = [int(k) for k in summary1["gate"]["G1c_interpretable_horizons"]]
    if not horizons:
        raise RuntimeError("no interpretable horizons survived Stage 1 gate G1c")

    run_dir, status, manifest = _start_run(
        "stage3", model_name, cfg, provenance,
        {"interpretable_horizons": horizons, "stage1_gate": summary1["gate"]},
    )
    adapter = None
    started = time.time()
    try:
        samples, frame, _stats, labels, by_id = corpus_bundle()
        adapter = load_adapter(cfg)
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        sites = resolve_site_table(adapter, samples)
        u = normalized_direction(arrays["carrier_direction"])
        rows, diagnostics = _intervention_sweep(
            adapter, cfg,
            frame=frame, by_id=by_id, labels=labels, sites=sites,
            output_token_ids=output_token_ids, u=u,
            setpoints=summary1["validation_setpoints"],
            horizons=horizons, pair_limit=None,
            decision_directions=arrays,
        )
        diagnostics["hook_leak_detected"] = _hook_leak(adapter)
        save_table(rows, run_dir / "intervention_rows.parquet")
        save_json(diagnostics, run_dir / "intervention_diagnostics.json")

        analysis = _curves_and_gate(
            rows, summary1, horizons,
            bootstraps=int(cfg.statistics.bootstrap_samples),
            confidence=float(cfg.statistics.confidence_level),
        )
        tolerances = diagnostics["tolerances"]
        fidelity_ok = bool(
            diagnostics["max_projection_relative_deviation"]
            <= tolerances["projection_validation_sigma"]
            and diagnostics["max_orthogonal_relative_deviation"]
            <= tolerances["orthogonal_relative"]
            and diagnostics["no_op_max_margin_deviation"] <= NO_OP_TOLERANCE
            and not diagnostics["hook_leak_detected"]
        )
        analysis["gate"]["G3.4_fidelity_passed"] = fidelity_ok
        analysis["gate"]["passed"] = bool(analysis["gate"]["passed"] and fidelity_ok)

        propagation = (
            rows[rows["condition"] == "setpoint"]
            .groupby("horizon")[
                [c for c in rows.columns if c.startswith(("p_norm_l", "p_q_l"))]
            ]
            .mean()
            .reset_index()
        )
        save_table(analysis["horizon_condition_summary"], run_dir / "horizon_condition_summary.parquet")
        save_table(propagation, run_dir / "propagation_by_horizon.parquet")
        save_json(
            {
                "model": adapter.display_model_id,
                "horizons": horizons,
                "curves": analysis["curves"],
                "contrasts": analysis["contrasts"],
                "shuffled_decision_nulls": analysis["shuffled_decision_nulls"],
                "C_raw_by_horizon": analysis["C_raw_by_horizon"],
                "C_z_by_horizon": analysis["C_z_by_horizon"],
                "D_primary_by_horizon": analysis["D_primary_by_horizon"],
                "gate": analysis["gate"],
                "wall_time_s": time.time() - started,
            },
            run_dir / "stage3_summary.json",
        )
        manifest.finish([{"stage": "stage3", "gate": analysis["gate"]}])
        status.complete(
            "Stage 3 complete; Stage 4 gate "
            + ("passed" if analysis["gate"]["passed"] else "not passed")
        )
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            _release(adapter)


# ------------------------------------------------------------------ stage 3b
def run_e15_stage3b(model_name: str = PRIMARY_MODEL) -> Path:
    """Secondary distractor-density contrast for H15.3 (protocol 8.2)."""
    stage3 = _stage_dir("stage3", model_name) / "stage3_summary.json"
    if not stage3.exists():
        raise RuntimeError("E15 Stage 3 must complete before the secondary contrast")
    cfg, provenance = _config(model_name)
    summary1, arrays = _load_stage1(model_name)
    run_dir, status, manifest = _start_run(
        "stage3b", model_name, cfg, provenance, {"secondary": True, "hypothesis": "H15.3"},
    )
    adapter = None
    try:
        results: dict[str, Any] = {"secondary": True, "hypothesis": "H15.3", "conditions": {}}
        adapter = load_adapter(cfg)
        u = normalized_direction(arrays["carrier_direction"])
        arms = [
            {"condition": "setpoint", "edit_site": "carrier", "kind": "setpoint", "direction_index": -1},
        ] + [
            {
                "condition": "random_normmatched",
                "edit_site": "carrier",
                "kind": "norm_matched",
                "direction_index": index,
                "direction": random_unit_direction(
                    adapter.hidden_size, DIRECTION_SEED_BASE + index
                ),
            }
            for index in range(N_RANDOM_DIRECTIONS)
        ]
        density_rows: list[pd.DataFrame] = []
        for label, pool, horizon in (
            ("sparse_long_steps", LONG_DISTRACTORS, 6),
            ("dense_short_steps", SHORT_DISTRACTORS, 16),
        ):
            samples, frame, _stats, labels, by_id = _density_bundle(pool, horizon, label)
            sites = resolve_site_table(adapter, samples)
            output_token_ids = candidate_first_token_ids(
                adapter, cfg.behavior.candidates_primary
            )
            clean_ids = horizon_view(frame, "discovery_test", horizon)["sample_id"].astype(str).tolist()
            clean = run_clean_batches(
                adapter, by_id, clean_ids, sites,
                output_token_ids=output_token_ids,
                decision_layers=PROPAGATION_LAYERS,
                batch_size=int(cfg.runtime.batch_size),
            )
            token_distance = float(
                np.mean(
                    [
                        sites[sid]["decision"]["token_index"]
                        - sites[sid]["carrier"]["token_index"]
                        for sid in clean_ids
                    ]
                )
            )
            rows, _diagnostics = _intervention_sweep(
                adapter, cfg,
                frame=frame, by_id=by_id, labels=labels, sites=sites,
                output_token_ids=output_token_ids, u=u,
                setpoints=summary1["validation_setpoints"],
                horizons=[horizon], pair_limit=None, arms=arms,
                decision_directions=None,
            )
            rows["density_condition"] = label
            density_rows.append(rows)
            treat = rows[rows["condition"] == "setpoint"]
            ci = cluster_bootstrap_mean_ci(
                treat["delta_margin_toward_expected"].to_numpy(float),
                treat["pair_id"].astype(str).tolist(),
                n_bootstraps=int(cfg.statistics.bootstrap_samples),
                confidence_level=float(cfg.statistics.confidence_level),
                seed=BOOTSTRAP_SEED + 91,
            )
            results["conditions"][label] = {
                "gap_steps": int(horizon),
                "mean_token_distance": token_distance,
                "C_raw": ci["mean"],
                "C_ci": [ci["ci_low"], ci["ci_high"]],
                "B": behavior_metrics(clean, labels, clean_ids)["accuracy"],
                "n_pairs": ci["n_clusters"],
            }
            del clean
            gc.collect()
        save_table(pd.concat(density_rows, ignore_index=True), run_dir / "density_rows.parquet")
        save_json(results, run_dir / "stage3b_summary.json")
        manifest.finish([{"stage": "stage3b", "result": results}])
        status.complete("Stage 3b secondary contrast complete")
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            _release(adapter)


def _density_bundle(pool, horizon: int, label: str):
    from .e15_support import build_e15_corpus

    samples, frame, stats = build_e15_corpus(
        horizons=[int(horizon)],
        distractor_pool=list(pool),
        namespace_suffix=f"-{label}",
    )
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    by_id = {str(s.sample_id): s for s in samples}
    return samples, frame, stats, labels, by_id


# ------------------------------------------------------------------- stage 4
def run_e15_stage4() -> Path:
    """Replication on the frozen second checkpoint, only if the gate passed."""
    stage3 = _stage_dir("stage3", PRIMARY_MODEL) / "stage3_summary.json"
    if not stage3.exists():
        raise RuntimeError("E15 Stage 3 must complete before Stage 4")
    gate = json.loads(stage3.read_text(encoding="utf-8"))["gate"]
    if not gate.get("passed"):
        raise RuntimeError(
            "E15 Stage 4 is not authorized: the frozen Stage 4 gate did not pass. "
            f"gate={json.dumps(gate, default=str)[:800]}"
        )
    run_e15_stage0(REPLICATION_MODEL)
    run_e15_stage1(REPLICATION_MODEL)
    run_e15_stage2(REPLICATION_MODEL)
    return run_e15_stage3(REPLICATION_MODEL)


# ------------------------------------------------------------------ analysis
def analyze_e15(model_name: str = PRIMARY_MODEL) -> Path:
    """Assemble the E15 discovery report from completed stage artifacts."""
    stage0 = _stage_dir("stage0", model_name)
    stage1 = _stage_dir("stage1", model_name)
    stage2 = _stage_dir("stage2", model_name)
    stage3 = _stage_dir("stage3", model_name)

    def read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    payload = {
        "e15_version": E15_VERSION,
        "model": model_name,
        "site": f"{SITE}/L{CARRIER_LAYER}",
        "stage0": read(stage0 / "stage0_gate.json"),
        "stage1": read(stage1 / "stage1_summary.json"),
        "stage2": read(stage2 / "stage2_gate.json"),
        "stage3": read(stage3 / "stage3_summary.json"),
        "stage3b": read(_stage_dir("stage3b", model_name) / "stage3b_summary.json"),
    }
    stage3_payload = payload["stage3"] or {}
    gate = stage3_payload.get("gate", {})
    if not stage3_payload:
        verdict = "incomplete"
    elif gate.get("passed"):
        verdict = "supported_pending_replication"
    elif not gate.get("G3.1_passed"):
        verdict = "unsupported_no_causal_effect_at_k0"
    elif not gate.get("G3.2_passed"):
        verdict = "unresolved_causal_curve_not_summarisable"
    elif not gate.get("G3.3_passed"):
        verdict = "unsupported_representation_decays_with_utilization"
    else:
        verdict = "unresolved"
    payload["verdict"] = verdict
    out = campaign_dir() / f"E15_ANALYSIS_{model_name}.json"
    atomic_write_json(out, payload)
    return out
