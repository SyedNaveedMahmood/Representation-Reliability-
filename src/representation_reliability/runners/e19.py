"""E19 temporal persistence and reorganization of causal organization.

Frozen by `docs/E19_TEMPORAL_CAUSAL_ORGANIZATION_PROTOCOL.md`.

Measures, at each horizon, the componentwise profile ``O(k) = (Q, A, G)`` plus
``D``, ``P`` and ``B``, at two loci taken from E18's map:

```text
locus S   state_word_last @ L8    k grows -> state age AND remaining distance grow
locus D   decision @ L24          k grows -> state age grows, distance stays ~0
```

Two things E15 lacked are built in: a per-horizon carrier-sufficiency gate, so a
`Q/A/G` decomposition of an absent causal effect is recorded but never
interpreted; and both the horizon-local and the frozen-reference estimands, so
code rotation can be told apart from pathway loss.
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

from ..adapters.intervention import (
    forward_multi_capture,
    forward_resid_post_edit,
    resid_post_hook_count,
)
from ..config import config_hash, resolve_config, save_resolved_config
from ..interventions.orthogonal_context import orthogonal_component
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
from ..metrics.causal import counterfactual_outcome, margin_toward_label
from ..metrics.decoding import classification_metrics
from ..metrics.setpoint import validation_setpoint_targets
from ..metrics.temporal_half_life import (
    bootstrap_two_sided_p,
    curve_cluster_bootstrap,
    half_life,
    holm_adjust,
)
from ..probes.linear import evaluate_probe, fit_probe, raw_probe_direction
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.status import StatusFile, atomic_write_json
from .e00c import candidate_first_token_ids
from .e01a import _prediction
from .e15_support import build_e15_corpus, horizon_view, selected_margin
from .e18 import resolve_site_tokens
from .extract import load_adapter

logger = logging.getLogger(__name__)

E19_VERSION = "e19-temporal-causal-organization-v1"

MODEL = "qwen3_1.7b"
CORPUS_SPECS: tuple[tuple[str, int, int], ...] = (
    ("train", 400, 20261901),
    ("validation", 200, 20261902),
    ("discovery_test", 150, 20261903),
)
HORIZONS: tuple[int, ...] = (1, 2, 4, 8)
K0 = 1

# Frozen loci, taken from E18's map; neither is searched over here.
LOCI: tuple[dict[str, Any], ...] = (
    {
        "name": "S_source",
        "site": "state_word_last",
        "layer": 8,
        "propagation": (12, 17, 21, 27),
        "grows": "age_and_distance",
    },
    {
        "name": "D_decision",
        "site": "decision",
        "layer": 24,
        "propagation": (27,),
        "grows": "age_only",
    },
)

N_RANDOM = 3
DIRECTION_SEED_BASE = 20261920
PROBE_SEED = 20261910
BOOTSTRAP_SEED = 20261930

NO_OP_TOLERANCE = 1e-6
MIN_BEHAVIOR = 0.70
MIN_D_NONINFERIORITY = -0.05
SESOI_RELATIVE = 0.25
SUFFICIENCY_FLIP = 0.10
MAGNITUDE_STABILITY_MIN = 0.80
CONTEXT_EPSILON = 1e-8

COMPONENTS = ("Q", "A", "G")
ESTIMANDS = ("native", "ref")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def campaign_dir() -> Path:
    path = repo_root() / "runs" / "E19_TEMPORAL_ORG"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config():
    root = repo_root()
    return resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / f"configs/models/{MODEL}.yaml",
        experiment_path=root / "configs/experiments/E19_temporal_causal_organization.yaml",
        overrides=(),
    )


def _corpus():
    samples, frame, stats = build_e15_corpus(
        specs=CORPUS_SPECS, horizons=HORIZONS, namespace_suffix="-e19"
    )
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    by_id = {str(s.sample_id): s for s in samples}
    return samples, frame, stats, labels, by_id


def _twin_map(view: pd.DataFrame) -> dict[str, str]:
    by_pair: dict[str, list[str]] = {}
    for row in view.to_dict("records"):
        by_pair.setdefault(str(row["pair_id"]), []).append(str(row["sample_id"]))
    twins: dict[str, str] = {}
    for pair_id, ids in by_pair.items():
        if len(ids) != 2:
            raise RuntimeError(f"pair {pair_id} is incomplete")
        first, second = sorted(ids)
        twins[first] = second
        twins[second] = first
    return twins


def _clean(
    adapter, by_id, ids, sites, *, site: str, layer: int, propagation, output_token_ids,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Clean forward: the locus state, the decision-token propagation states."""
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(ids), stride):
        chunk = ids[start : start + stride]
        specs: list[tuple[str, int, list[int]]] = [
            ("site", int(layer), [sites[sid][site][0] for sid in chunk])
        ]
        for prop_layer in propagation:
            specs.append(
                (f"dec_l{prop_layer}", int(prop_layer), [sites[sid]["decision"][0] for sid in chunk])
            )
        out = forward_multi_capture(
            adapter,
            [by_id[sid].prompt for sid in chunk],
            readout_token_indices=[sites[sid]["_readout"][0] for sid in chunk],
            output_token_ids=list(map(int, output_token_ids)),
            capture_specs=specs,
        )
        logits = np.asarray(out["selected_logits"], dtype=np.float64)
        for row, sid in enumerate(chunk):
            result[sid] = {
                "selected_logits": logits[row].copy(),
                "site": np.asarray(out["captured"]["site"][row], dtype=np.float64).copy(),
                "propagation": {
                    int(p): np.asarray(out["captured"][f"dec_l{p}"][row], dtype=np.float64).copy()
                    for p in propagation
                },
            }
    if set(result) != set(ids):
        raise RuntimeError("clean forward identity mismatch")
    return result


def _edit(
    adapter, by_id, ids, sites, *, site: str, layer: int, propagation,
    deltas_by_id, output_token_ids, batch_size: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(ids), stride):
        chunk = ids[start : start + stride]
        out = forward_resid_post_edit(
            adapter,
            [by_id[sid].prompt for sid in chunk],
            edit_layer=int(layer),
            edit_token_indices=[sites[sid][site][0] for sid in chunk],
            deltas=np.stack([np.asarray(deltas_by_id[sid], dtype=np.float64) for sid in chunk]),
            readout_token_indices=[sites[sid]["_readout"][0] for sid in chunk],
            output_token_ids=list(map(int, output_token_ids)),
            capture_layers=list(map(int, propagation)),
            capture_token_indices=[sites[sid]["decision"][0] for sid in chunk],
        )
        logits = np.asarray(out["selected_logits"], dtype=np.float64)
        edited = np.asarray(out["edited_carrier_state"], dtype=np.float64)
        for row, sid in enumerate(chunk):
            result[sid] = {
                "selected_logits": logits[row].copy(),
                "edited_state": edited[row].copy(),
                "propagation": {
                    int(p): np.asarray(v[row], dtype=np.float64).copy()
                    for p, v in out["captured"].items()
                },
            }
    if set(result) != set(ids):
        raise RuntimeError("edited forward identity mismatch")
    return result


def _fit_axis(states_by_split, labels, c_grid, seed):
    x_tr, y_tr = states_by_split["train"]
    x_va, y_va = states_by_split["validation"]
    x_te, y_te = states_by_split["discovery_test"]
    fit = fit_probe(x_tr, y_tr, x_va, y_va, c_grid=list(c_grid), seed=int(seed))
    metrics = evaluate_probe(fit, x_te, y_te)
    return normalized_direction(raw_probe_direction(fit)), fit, metrics


def run_e19() -> Path:
    """Execute the frozen E19 temporal causal-organization sweep."""
    cfg, provenance = _config()
    run_dir = campaign_dir() / MODEL
    run_dir.mkdir(parents=True, exist_ok=True)
    status = StatusFile.create(run_dir, run_id=f"E19-{MODEL}", experiment_id="E19")
    manifest = RunManifest(run_dir)
    adapter = None
    try:
        samples, frame, stats, labels, by_id = _corpus()
        manifest.set_start(
            config_hash(cfg),
            {**provenance, "e19_version": E19_VERSION,
             "loci": [dict(l) for l in LOCI], "horizons": list(HORIZONS)},
            {"probe": PROBE_SEED, "direction_base": DIRECTION_SEED_BASE,
             "bootstrap": BOOTSTRAP_SEED,
             "corpus": {n: s for n, _c, s in CORPUS_SPECS}},
        )
        save_resolved_config(cfg, run_dir / "resolved_config.yaml", provenance)
        save_json(stats, run_dir / "corpus_statistics.json")

        adapter = load_adapter(cfg)
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        sites = {str(s.sample_id): resolve_site_tokens(adapter.tokenizer, s) for s in samples}
        manifest.update_model_info(
            id=adapter.display_model_id, dtype=str(cfg.model.dtype),
            num_layers=adapter.num_layers, hidden_size=adapter.hidden_size,
            candidate_token_ids=output_token_ids,
        )
        manifest.update_dataset_info(
            split_hash=dataset_split_hash(
                dict(zip(frame["sample_id"].astype(str), frame["split"].astype(str)))
            ),
            prompt_hash_sample=prompt_hash(str(frame["prompt"].iloc[0])),
            confirmation_accessed=False,
        )

        batch_size = int(cfg.runtime.batch_size)
        hidden = int(adapter.hidden_size)
        tolerances = setpoint_fidelity_tolerances(str(cfg.model.dtype))
        diagnostics: dict[str, Any] = {
            "no_op_max_margin_deviation": 0.0,
            "max_norm_match_relative_deviation": 0.0,
            "max_projection_relative_deviation": 0.0,
            "max_orthogonal_relative_deviation": 0.0,
            "hook_leak_detected": False,
            "tolerances": tolerances,
        }
        rows: list[dict[str, Any]] = []
        probe_records: list[dict[str, Any]] = []
        behaviour_records: list[dict[str, Any]] = []
        axes: dict[tuple[str, int], np.ndarray] = {}
        targets: dict[tuple[str, int], dict[str, float]] = {}

        # ---- pass A: horizon-local axes and validation setpoint targets.
        for locus in LOCI:
            for horizon in HORIZONS:
                ids_by_split = {
                    split: horizon_view(frame, split, horizon)["sample_id"].astype(str).tolist()
                    for split in ("train", "validation", "discovery_test")
                }
                every = [sid for split in ids_by_split for sid in ids_by_split[split]]
                clean = _clean(
                    adapter, by_id, every, sites, site=locus["site"], layer=locus["layer"],
                    propagation=(), output_token_ids=output_token_ids, batch_size=batch_size,
                )
                blocks = {
                    split: (
                        np.stack([clean[sid]["site"] for sid in ids_by_split[split]]),
                        np.asarray([labels[sid] for sid in ids_by_split[split]], dtype=int),
                    )
                    for split in ids_by_split
                }
                axis, _fit, metrics = _fit_axis(blocks, labels, cfg.probe.C_grid, PROBE_SEED)
                axes[(locus["name"], horizon)] = axis
                probe_records.append({
                    "locus": locus["name"], "site": locus["site"], "layer": locus["layer"],
                    "horizon": int(horizon), "test_auroc": metrics.get("auroc"),
                    "test_balanced_accuracy": metrics.get("balanced_accuracy"),
                })
                val_ids = ids_by_split["validation"]
                targets[(locus["name"], horizon)] = validation_setpoint_targets(
                    [coordinate_value(clean[sid]["site"], axis) for sid in val_ids],
                    [int(labels[sid]) for sid in val_ids],
                    [selected_margin(clean[sid]["selected_logits"]) for sid in val_ids],
                )
                test_ids = ids_by_split["discovery_test"]
                margins = np.asarray(
                    [selected_margin(clean[sid]["selected_logits"]) for sid in test_ids]
                )
                y_true = np.asarray([labels[sid] for sid in test_ids], dtype=int)
                behaviour_records.append({
                    "locus": locus["name"], "horizon": int(horizon),
                    "accuracy": float(((margins >= 0).astype(int) == y_true).mean()),
                    "margin_abs_mean": float(np.abs(margins).mean()),
                    "margin_auroc": classification_metrics(y_true, margins).get("auroc"),
                })
                del clean
                gc.collect()

        behaviour = pd.DataFrame(behaviour_records)
        b_by_k = (
            behaviour.groupby("horizon")["accuracy"].min().to_dict()
        )
        interpretable: list[int] = []
        for horizon in HORIZONS:
            if float(b_by_k[int(horizon)]) >= MIN_BEHAVIOR:
                interpretable.append(int(horizon))
            else:
                break
        if not interpretable or interpretable[0] != K0:
            raise RuntimeError(f"E19 gate G2 failed at k0: {b_by_k}")

        # ---- pass B: the factorial sweep on discovery_test.
        for locus in LOCI:
            for horizon in interpretable:
                view = horizon_view(frame, "discovery_test", horizon)
                ids = view["sample_id"].astype(str).tolist()
                meta = view.set_index("sample_id")
                twins = _twin_map(view)
                clean = _clean(
                    adapter, by_id, ids, sites, site=locus["site"], layer=locus["layer"],
                    propagation=locus["propagation"], output_token_ids=output_token_ids,
                    batch_size=batch_size,
                )
                clean_margin = {sid: selected_margin(clean[sid]["selected_logits"]) for sid in ids}
                twin_state = {sid: clean[twins[sid]]["site"] for sid in ids}

                plan: list[dict[str, Any]] = [
                    {"condition": "no_op", "estimand": "none", "index": -1,
                     "deltas": {sid: np.zeros(hidden) for sid in ids}},
                    {"condition": "full_state_patch", "estimand": "none", "index": -1,
                     "deltas": {sid: twin_state[sid] - clean[sid]["site"] for sid in ids}},
                ]
                setpoint_reference: dict[str, np.ndarray] = {}
                context_reference: dict[str, np.ndarray] = {}
                for estimand in ESTIMANDS:
                    key = (locus["name"], horizon if estimand == "native" else K0)
                    axis = axes[key]
                    target = targets[key]
                    q_target = {
                        sid: float(
                            target["q0_star"] if int(labels[sid]) == 1 else target["q1_star"]
                        )
                        for sid in ids
                    }
                    scalar = {
                        sid: source_free_setpoint_delta(clean[sid]["site"], axis, q_target[sid])
                        for sid in ids
                    }
                    context = {
                        sid: orthogonal_component(twin_state[sid], clean[sid]["site"], axis)
                        for sid in ids
                    }
                    if estimand == "native":
                        setpoint_reference = scalar
                        context_reference = context
                    plan.extend([
                        {"condition": "Y10_scalar", "estimand": estimand, "index": -1,
                         "deltas": scalar, "axis": axis, "q_target": q_target},
                        {"condition": "Y01_context", "estimand": estimand, "index": -1,
                         "deltas": context, "axis": axis},
                        {"condition": "Y11_both", "estimand": estimand, "index": -1,
                         "deltas": {sid: scalar[sid] + context[sid] for sid in ids},
                         "axis": axis},
                    ])
                native_axis = axes[(locus["name"], horizon)]
                for i in range(N_RANDOM):
                    direction = random_unit_direction(hidden, DIRECTION_SEED_BASE + i)
                    plan.append({
                        "condition": "random_norm_matched", "estimand": "none", "index": i,
                        "deltas": {
                            sid: norm_matched_direction_delta(setpoint_reference[sid], direction)
                            for sid in ids
                        },
                    })
                for i in range(N_RANDOM):
                    direction = random_unit_direction(
                        hidden, DIRECTION_SEED_BASE + 100 + i, orthogonal_to=native_axis
                    )
                    plan.append({
                        "condition": "orthogonal_random", "estimand": "none", "index": i,
                        "deltas": {
                            sid: norm_matched_direction_delta(context_reference[sid], direction)
                            for sid in ids
                        },
                    })

                for arm in plan:
                    edited = _edit(
                        adapter, by_id, ids, sites, site=locus["site"], layer=locus["layer"],
                        propagation=locus["propagation"], deltas_by_id=arm["deltas"],
                        output_token_ids=output_token_ids, batch_size=batch_size,
                    )
                    for sid in ids:
                        base = clean[sid]["site"]
                        after = edited[sid]["edited_state"]
                        delta = np.asarray(arm["deltas"][sid], dtype=np.float64)
                        margin_before = clean_margin[sid]
                        margin_after = selected_margin(edited[sid]["selected_logits"])
                        label = int(labels[sid])
                        expected = 1 - label
                        outcome = counterfactual_outcome(
                            _prediction(margin_before), _prediction(margin_after), expected
                        )
                        delta_norm = float(np.linalg.norm(delta))
                        activation_norm = float(np.linalg.norm(base))
                        record: dict[str, Any] = {
                            "locus": locus["name"], "site": locus["site"],
                            "layer": int(locus["layer"]), "grows": locus["grows"],
                            "horizon": int(horizon), "condition": str(arm["condition"]),
                            "estimand": str(arm["estimand"]), "direction_index": int(arm["index"]),
                            "base_sample_id": sid,
                            "pair_id": str(meta.loc[sid, "pair_id"]),
                            "episode_index": int(meta.loc[sid, "episode_index"]),
                            "target_label": label, "expected_label": expected,
                            "margin_before": margin_before, "margin_after": margin_after,
                            "delta_margin_toward_expected": float(
                                margin_toward_label(margin_after, expected)
                                - margin_toward_label(margin_before, expected)
                            ),
                            "prediction_before": _prediction(margin_before),
                            "prediction_after": _prediction(margin_after),
                            "counterfactual_flip": outcome["counterfactual_flip"],
                            "delta_norm": delta_norm,
                            "activation_norm": activation_norm,
                            "delta_over_activation_norm": delta_norm / max(activation_norm, 1e-12),
                        }
                        if arm["condition"] == "no_op":
                            diagnostics["no_op_max_margin_deviation"] = max(
                                diagnostics["no_op_max_margin_deviation"],
                                abs(margin_after - margin_before),
                            )
                        if arm["condition"] == "random_norm_matched":
                            reference = float(np.linalg.norm(setpoint_reference[sid]))
                            diagnostics["max_norm_match_relative_deviation"] = max(
                                diagnostics["max_norm_match_relative_deviation"],
                                abs(delta_norm - reference) / max(reference, 1e-12),
                            )
                        if arm["condition"] == "orthogonal_random":
                            reference = float(np.linalg.norm(context_reference[sid]))
                            diagnostics["max_norm_match_relative_deviation"] = max(
                                diagnostics["max_norm_match_relative_deviation"],
                                abs(delta_norm - reference) / max(reference, 1e-12),
                            )
                        if arm["condition"] == "Y10_scalar":
                            fidelity = setpoint_identity_diagnostics(
                                base, after, arm["axis"], float(arm["q_target"][sid])
                            )
                            sigma_q = float(
                                targets[
                                    (locus["name"], horizon if arm["estimand"] == "native" else K0)
                                ]["sigma_q_validation"]
                            )
                            diagnostics["max_projection_relative_deviation"] = max(
                                diagnostics["max_projection_relative_deviation"],
                                abs(fidelity["q_after"] - fidelity["q_target"])
                                / max(sigma_q, 1e-12),
                            )
                            diagnostics["max_orthogonal_relative_deviation"] = max(
                                diagnostics["max_orthogonal_relative_deviation"],
                                fidelity["orthogonal_relative_deviation"],
                            )
                        if arm["condition"] == "Y01_context":
                            record["context_dot_axis"] = float(
                                np.dot(delta, arm["axis"])
                            )
                        for prop_layer in locus["propagation"]:
                            clean_state = clean[sid]["propagation"][int(prop_layer)]
                            edited_state = edited[sid]["propagation"][int(prop_layer)]
                            record[f"p_norm_l{prop_layer}"] = float(
                                np.linalg.norm(edited_state - clean_state)
                                / max(float(np.linalg.norm(clean_state)), 1e-12)
                            )
                        rows.append(record)
                    del edited
                del clean
                gc.collect()

        diagnostics["hook_leak_detected"] = any(
            resid_post_hook_count(adapter, layer=int(l["layer"])) for l in LOCI
        )
        raw = pd.DataFrame(rows)
        if not np.isfinite(raw["delta_margin_toward_expected"].to_numpy(float)).all():
            raise RuntimeError("non-finite E19 evidence produced")
        np.savez(
            run_dir / "probe_axes.npz",
            **{f"{name}__k{horizon}": axis for (name, horizon), axis in axes.items()},
        )
        save_table(raw, run_dir / "intervention_rows.parquet")
        save_table(pd.DataFrame(probe_records), run_dir / "probe_metrics.parquet")
        save_table(behaviour, run_dir / "behaviour_by_horizon.parquet")
        save_json(diagnostics, run_dir / "diagnostics.json")

        analysis = analyze_rows(
            raw, probe_records, axes, interpretable,
            bootstraps=int(cfg.statistics.bootstrap_samples),
            confidence=float(cfg.statistics.confidence_level),
        )
        numerics_ok = bool(
            diagnostics["no_op_max_margin_deviation"] <= NO_OP_TOLERANCE
            and diagnostics["max_norm_match_relative_deviation"] <= 1e-6
            and diagnostics["max_projection_relative_deviation"]
            <= tolerances["projection_validation_sigma"]
            and diagnostics["max_orthogonal_relative_deviation"]
            <= tolerances["orthogonal_relative"]
            and not diagnostics["hook_leak_detected"]
        )
        analysis["gates"] = {
            "G1_numerics_passed": numerics_ok,
            "G2_behaviour_by_horizon": {str(k): float(v) for k, v in b_by_k.items()},
            "G2_interpretable_horizons": interpretable,
            **analysis.get("gates", {}),
        }
        summary = {
            "e19_version": E19_VERSION,
            "model": adapter.display_model_id,
            "loci": [dict(l) for l in LOCI],
            "horizons": interpretable,
            "diagnostics": diagnostics,
            "confirmation_accessed": False,
            **analysis,
        }
        save_json(summary, run_dir / "e19_summary.json")
        atomic_write_json(campaign_dir() / "E19_TEMPORAL_ORG.json", summary)
        manifest.finish([{"stage": "e19", "outcome": summary.get("outcome")}])
        if not numerics_ok:
            status.fail("E19 numerics gate G1 failed")
            raise RuntimeError(f"E19 G1 failed: {diagnostics}")
        status.complete(f"E19 complete: {summary.get('outcome')}")
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _component_frame(raw: pd.DataFrame, locus: str, estimand: str) -> pd.DataFrame:
    """Per-episode Q, A, G at every horizon for one locus and estimand."""
    block = raw[
        (raw["locus"] == locus)
        & (raw["estimand"] == estimand)
        & (raw["condition"].isin(["Y10_scalar", "Y01_context", "Y11_both"]))
    ]
    wide = block.pivot_table(
        index=["episode_index", "horizon", "base_sample_id"],
        columns="condition", values="delta_margin_toward_expected",
    ).reset_index()
    wide["Q"] = wide["Y10_scalar"]
    wide["A"] = wide["Y01_context"]
    wide["G"] = (wide["Y11_both"] - wide["Y10_scalar"]) - wide["Y01_context"]
    return wide


def analyze_rows(
    raw: pd.DataFrame,
    probe_records: list[dict[str, Any]],
    axes: dict[tuple[str, int], np.ndarray],
    horizons: list[int],
    *,
    bootstraps: int,
    confidence: float,
    enforce_magnitude_gate: bool = False,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Curves, gates, hypotheses and the frozen outcome label.

    ``enforce_magnitude_gate`` is E20's preregistered G4. When True a curve whose
    scalar edit shrank below ``MAGNITUDE_STABILITY_MIN`` of its k0 residual
    fraction is excluded from every hypothesis test and from any half-life, not
    merely from the outcome label. E19 ran with it False, which is what its
    recorded results reflect; the control was post-hoc there.
    """
    out: dict[str, Any] = {"curves": {}, "sufficiency": {}, "hypotheses": {}, "rotation": {}}
    probes = pd.DataFrame(probe_records)

    # --- G3 carrier sufficiency per locus and horizon.
    for locus in raw["locus"].unique():
        for horizon in horizons:
            patch = raw[
                (raw["locus"] == locus)
                & (raw["horizon"] == horizon)
                & (raw["condition"] == "full_state_patch")
            ]
            control = raw[
                (raw["locus"] == locus)
                & (raw["horizon"] == horizon)
                & (raw["condition"] == "random_norm_matched")
            ]
            boot = curve_cluster_bootstrap(
                patch.assign(horizon=horizon),
                cluster_col="episode_index", horizon_col="horizon",
                value_col="delta_margin_toward_expected",
                n_bootstraps=bootstraps, confidence_level=confidence,
                seed=bootstrap_seed + 11 * int(horizon),
            )
            flip = float(patch["counterfactual_flip"].mean())
            effect_excludes = bool(boot["ci_low"][0] > 0.0 or boot["ci_high"][0] < 0.0)
            beats = bool(
                float(patch["delta_margin_toward_expected"].mean())
                > float(control["delta_margin_toward_expected"].mean())
            )
            out["sufficiency"][f"{locus}@k{horizon}"] = {
                "flip_rate": flip,
                "effect": boot["point"][0],
                "effect_ci": [boot["ci_low"][0], boot["ci_high"][0]],
                "effect_ci_excludes_zero": effect_excludes,
                "exceeds_random": beats,
                "passes_G3": bool(
                    flip >= SUFFICIENCY_FLIP and effect_excludes and beats
                ),
            }

    # --- component curves, per locus and estimand.
    for locus in sorted(raw["locus"].unique()):
        for estimand in ESTIMANDS:
            wide = _component_frame(raw, locus, estimand)
            if wide.empty:
                continue
            key = f"{locus}/{estimand}"
            entry: dict[str, Any] = {}
            draws: dict[str, np.ndarray] = {}
            for component in COMPONENTS:
                boot = curve_cluster_bootstrap(
                    wide, cluster_col="episode_index", horizon_col="horizon",
                    value_col=component, n_bootstraps=bootstraps,
                    confidence_level=confidence,
                    seed=bootstrap_seed + 3 * COMPONENTS.index(component),
                )
                draws[component] = boot["draws"]
                # A component whose k0 value is itself indistinguishable from
                # zero has no meaningful relative curve: normalising by a
                # near-zero baseline manufactures huge ratios out of noise.
                # Report it as not assessable instead (protocol section 9).
                assessable = bool(boot["ci_low"][0] > 0.0 or boot["ci_high"][0] < 0.0)
                rel = (
                    [
                        (v / boot["point"][0]) if abs(boot["point"][0]) > 1e-9 else float("nan")
                        for v in boot["point"]
                    ]
                    if assessable
                    else [float("nan")] * len(boot["point"])
                )
                entry[component] = {
                    "horizons": boot["horizons"],
                    "point": boot["point"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "relative": rel,
                    "n_clusters": boot["n_clusters"],
                    "k0_assessable": assessable,
                }
                if all(np.isfinite(rel)) and len(rel) >= 2:
                    entry[component]["half_life"] = half_life(
                        boot["horizons"], rel,
                        baseline_positive=bool(boot["point"][0] > 0),
                        baseline_ci_excludes_zero=entry[component]["k0_assessable"],
                    )
            out["curves"][key] = entry
            out.setdefault("_draws", {})[key] = draws

    # --- G4 edit-magnitude stability, computed BEFORE any hypothesis so it can
    # gate them. The native estimand recomputes its validation setpoint targets
    # at every horizon, so if the class medians converge the edit shrinks and a
    # falling Q would be a magnitude artifact rather than pathway loss. The ref
    # estimand holds the k0 targets and is the magnitude-stable comparator.
    magnitude = (
        raw[raw["condition"] == "Y10_scalar"]
        .groupby(["locus", "estimand", "horizon"])[
            ["delta_norm", "activation_norm", "delta_over_activation_norm"]
        ]
        .mean()
        .reset_index()
    )
    out["edit_magnitude"] = magnitude.to_dict(orient="records")
    stability: dict[str, Any] = {}
    for (locus, estimand), block in magnitude.groupby(["locus", "estimand"]):
        ordered = block.sort_values("horizon")["delta_over_activation_norm"].to_numpy(float)
        ratio = float(ordered[-1] / ordered[0]) if ordered[0] > 0 else float("nan")
        stability[f"{locus}/{estimand}"] = {
            "residual_fraction_k0": float(ordered[0]),
            "residual_fraction_kstar": float(ordered[-1]),
            "kstar_over_k0": ratio,
            "magnitude_stable": bool(
                np.isfinite(ratio) and ratio >= MAGNITUDE_STABILITY_MIN
            ),
        }
    out["edit_magnitude_stability"] = stability
    stable_keys = {k for k, v in stability.items() if v["magnitude_stable"]}
    out["magnitude_stable_curves"] = sorted(stable_keys)
    out["magnitude_gate_enforced"] = bool(enforce_magnitude_gate)

    def _excluded(curve_key: str) -> bool:
        return bool(enforce_magnitude_gate and curve_key not in stable_keys)

    # Under G4 an excluded curve may not carry a half-life either.
    if enforce_magnitude_gate:
        for curve_key, entry in out["curves"].items():
            if _excluded(curve_key):
                for component in COMPONENTS:
                    entry[component].pop("half_life", None)
                    entry[component]["excluded_magnitude_unstable"] = True

    # --- H19.1 representational persistence.
    persistence = {}
    for locus in sorted(probes["locus"].unique()):
        block = probes[probes["locus"] == locus].set_index("horizon")
        k_star = horizons[-1]
        delta_d = float(block.loc[k_star, "test_auroc"]) - float(block.loc[K0, "test_auroc"])
        persistence[locus] = {
            "D_k0": float(block.loc[K0, "test_auroc"]),
            "D_kstar": float(block.loc[k_star, "test_auroc"]),
            "delta": delta_d,
            "non_inferior": bool(delta_d > MIN_D_NONINFERIORITY),
        }
    out["hypotheses"]["H19.1"] = {
        "margin": MIN_D_NONINFERIORITY,
        "per_locus": persistence,
        "supported": bool(all(v["non_inferior"] for v in persistence.values())),
    }

    # --- H19.2 componentwise change at k*, Holm-corrected.
    k_star = horizons[-1]
    h2: dict[str, Any] = {}
    for key, draws in out.get("_draws", {}).items():
        if _excluded(key):
            h2[key] = {"status": "excluded_magnitude_unstable"}
            continue
        entry = out["curves"][key]
        k_index = entry["Q"]["horizons"].index(k_star)
        raw_p: dict[str, float] = {}
        detail: dict[str, Any] = {}
        for component in COMPONENTS:
            if not entry[component]["k0_assessable"]:
                detail[component] = {"status": "not_assessable_k0_includes_zero"}
                raw_p[component] = float("nan")
                continue
            contrast = draws[component][:, k_index] - draws[component][:, 0]
            point = entry[component]["point"][k_index] - entry[component]["point"][0]
            sesoi = SESOI_RELATIVE * abs(entry[component]["point"][0])
            lo, hi = np.nanquantile(contrast, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
            p = bootstrap_two_sided_p(contrast)
            raw_p[component] = p
            detail[component] = {
                "change_k0_to_kstar": float(point),
                "ci": [float(lo), float(hi)],
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "sesoi": float(sesoi),
                "exceeds_sesoi": bool(abs(point) >= sesoi),
                "raw_p": p,
            }
        adjusted = holm_adjust(raw_p)
        for component in COMPONENTS:
            if component in detail and "status" not in detail[component]:
                holm_p = adjusted.get(component)
                detail[component]["holm_p"] = holm_p
                # A Holm-adjusted p of exactly 0.0 is the STRONGEST possible
                # result, so it must not be short-circuited by falsy-zero truth
                # testing. Compare explicitly against None.
                significant = holm_p is not None and np.isfinite(holm_p) and holm_p < 0.05
                detail[component]["supported"] = bool(
                    detail[component]["exceeds_sesoi"]
                    and detail[component]["ci_excludes_zero"]
                    and significant
                )
        h2[key] = {
            "k_star": k_star,
            "components": detail,
            "any_component_supported": bool(
                any(v.get("supported") for v in detail.values())
            ),
        }
    out["hypotheses"]["H19.2"] = h2

    # --- H19.3 differential pathway persistence at k*.
    h3: dict[str, Any] = {}
    for key, draws in out.get("_draws", {}).items():
        if _excluded(key):
            h3[key] = {"status": "excluded_magnitude_unstable"}
            continue
        entry = out["curves"][key]
        k_index = entry["Q"]["horizons"].index(k_star)
        pairs: dict[str, Any] = {}
        for i, first in enumerate(COMPONENTS):
            for second in COMPONENTS[i + 1 :]:
                if not (entry[first]["k0_assessable"] and entry[second]["k0_assessable"]):
                    pairs[f"{first}_vs_{second}"] = {"status": "not_assessable"}
                    continue
                rel_a = draws[first][:, k_index] / draws[first][:, 0]
                rel_b = draws[second][:, k_index] / draws[second][:, 0]
                contrast = rel_a - rel_b
                point = float(
                    entry[first]["relative"][k_index] - entry[second]["relative"][k_index]
                )
                lo, hi = np.nanquantile(
                    contrast, [(1 - confidence) / 2, 1 - (1 - confidence) / 2]
                )
                pairs[f"{first}_vs_{second}"] = {
                    "relative_difference": point,
                    "ci": [float(lo), float(hi)],
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                    "exceeds_sesoi": bool(abs(point) >= SESOI_RELATIVE),
                    "supported": bool(
                        (lo > 0.0 or hi < 0.0) and abs(point) >= SESOI_RELATIVE
                    ),
                }
        h3[key] = {
            "pairs": pairs,
            "any_pair_supported": bool(
                any(v.get("supported") for v in pairs.values())
            ),
        }
    out["hypotheses"]["H19.3"] = h3

    # --- H19.4 age versus remaining distance.
    h4: dict[str, Any] = {}
    for estimand in ESTIMANDS:
        source = out["curves"].get(f"S_source/{estimand}")
        decision = out["curves"].get(f"D_decision/{estimand}")
        if not source or not decision:
            continue
        if _excluded(f"S_source/{estimand}") or _excluded(f"D_decision/{estimand}"):
            h4[estimand] = {"status": "excluded_magnitude_unstable"}
            continue
        per_component = {}
        for component in COMPONENTS:
            k_index = source[component]["horizons"].index(k_star)
            per_component[component] = {
                "S_relative_at_kstar": source[component]["relative"][k_index],
                "D_relative_at_kstar": decision[component]["relative"][k_index],
                "distance_contribution": (
                    float(
                        decision[component]["relative"][k_index]
                        - source[component]["relative"][k_index]
                    )
                    if np.isfinite(source[component]["relative"][k_index])
                    and np.isfinite(decision[component]["relative"][k_index])
                    else None
                ),
            }
        h4[estimand] = per_component
    out["hypotheses"]["H19.4"] = {
        "note": (
            "locus D grows state age only; locus S grows age and remaining "
            "distance. The gap is the frozen descriptive estimate of the "
            "propagation-distance contribution."
        ),
        "per_estimand": h4,
    }

    # --- axis rotation.
    for locus in sorted({k[0] for k in axes}):
        base = axes[(locus, K0)]
        out["rotation"][locus] = {
            str(k): float(np.dot(base, axes[(locus, k)]))
            for k in horizons
            if (locus, k) in axes
        }

    # --- frozen outcome label.
    interpretable_cells = [
        key for key, value in out["sufficiency"].items() if value["passes_G3"]
    ]
    # Only magnitude-stable curves may drive the outcome label.
    h2_any = any(
        v.get("any_component_supported") for k, v in h2.items() if k in stable_keys
    )
    h3_any = any(
        v.get("any_pair_supported") for k, v in h3.items() if k in stable_keys
    )
    if not interpretable_cells:
        outcome = "no_cell_passed_carrier_sufficiency"
    elif not out["hypotheses"]["H19.1"]["supported"]:
        outcome = "representation_did_not_persist"
    elif h2_any and h3_any:
        outcome = "componentwise_reorganization"
    elif h2_any:
        outcome = "causal_organization_changed_uniformly"
    else:
        outcome = "no_material_causal_change_over_horizon"
    out["outcome"] = outcome
    out.pop("_draws", None)
    return out


def reanalyze_e19(model_name: str = MODEL) -> Path:
    """Re-run the analysis from persisted rows, probes and axes. No GPU."""
    run_dir = campaign_dir() / model_name
    raw = pd.read_parquet(run_dir / "intervention_rows.parquet")
    probes = pd.read_parquet(run_dir / "probe_metrics.parquet").to_dict(orient="records")
    with np.load(run_dir / "probe_axes.npz") as archive:
        axes = {}
        for key in archive.files:
            name, _, horizon = key.partition("__k")
            axes[(name, int(horizon))] = np.asarray(archive[key], dtype=np.float64)
    previous = json.loads((run_dir / "e19_summary.json").read_text(encoding="utf-8"))
    horizons = [int(k) for k in previous["gates"]["G2_interpretable_horizons"]]
    cfg, _provenance = _config()
    analysis = analyze_rows(
        raw, probes, axes, horizons,
        bootstraps=int(cfg.statistics.bootstrap_samples),
        confidence=float(cfg.statistics.confidence_level),
    )
    summary = {**previous, **analysis}
    summary["gates"] = {**previous["gates"], **analysis.get("gates", {})}
    save_json(summary, run_dir / "e19_summary.json")
    atomic_write_json(campaign_dir() / "E19_TEMPORAL_ORG.json", summary)
    return run_dir


def read_e19() -> dict[str, Any] | None:
    path = campaign_dir() / "E19_TEMPORAL_ORG.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
