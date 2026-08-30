"""E20 long-horizon extension of the temporal causal-organization profile.

Frozen by `docs/E20_LONG_HORIZON_PROTOCOL.md`.

E19 established the signature but every half-life was right-censored: nothing
halved by ``k=8``. E20 changes exactly one thing - how far the horizon reaches -
and inherits the carrier, the two-locus design, the components, the estimands,
the arms, the inference and the claim boundary from E19 unchanged.

Two structural points:

* the horizon range is chosen in a **non-causal Phase 1**. Both distractor pools
  are measured on behaviour and decodability only, with every intervention
  forbidden, and the frozen rule picks whichever reaches further at ``B >= 0.70``;
* E19's post-hoc magnitude control is now the preregistered gate **G4**, which
  excludes a failing curve from every hypothesis test and from any half-life.
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

from ..adapters.intervention import resid_post_hook_count
from ..config import config_hash, resolve_config, save_resolved_config
from ..data.stateful_console import LONG_DISTRACTORS, SHORT_DISTRACTORS
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
from ..probes.linear import evaluate_probe, fit_probe, raw_probe_direction
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.status import StatusFile, atomic_write_json
from .e00c import candidate_first_token_ids
from .e01a import _prediction
from .e15_support import build_e15_corpus, horizon_view, selected_margin
from .e18 import resolve_site_tokens
from .e19 import (
    ESTIMANDS,
    LOCI,
    MIN_BEHAVIOR,
    N_RANDOM,
    NO_OP_TOLERANCE,
    _clean,
    _edit,
    _twin_map,
    analyze_rows,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)

E20_VERSION = "e20-long-horizon-v1"

MODEL = "qwen3_1.7b"
CORPUS_SPECS: tuple[tuple[str, int, int], ...] = (
    ("train", 300, 20262001),
    ("validation", 150, 20262002),
    ("discovery_test", 150, 20262003),
)
CANDIDATE_HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32)
K0 = 1

POOLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("short", tuple(SHORT_DISTRACTORS)),
    ("long", tuple(LONG_DISTRACTORS)),
)
TIE_BREAK_POOL = "short"          # continuity with E15, E18 and E19

PROBE_SEED = 20262010
DIRECTION_SEED_BASE = 20262020
BOOTSTRAP_SEED = 20262030


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def campaign_dir() -> Path:
    path = repo_root() / "runs" / "E20_LONG_HORIZON"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config():
    root = repo_root()
    return resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / f"configs/models/{MODEL}.yaml",
        experiment_path=root / "configs/experiments/E20_long_horizon.yaml",
        overrides=(),
    )


def _corpus(pool_name: str, pool: tuple[str, ...], horizons: tuple[int, ...]):
    samples, frame, stats = build_e15_corpus(
        specs=CORPUS_SPECS, horizons=horizons,
        distractor_pool=list(pool), namespace_suffix=f"-e20-{pool_name}",
    )
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    by_id = {str(s.sample_id): s for s in samples}
    return samples, frame, stats, labels, by_id


def select_grid(reach_by_pool: dict[str, list[int]]) -> tuple[str, list[int]]:
    """Frozen Phase 1 rule: greater reach wins, ties go to the short pool."""
    best_name = TIE_BREAK_POOL
    best = reach_by_pool.get(TIE_BREAK_POOL, [])
    for name, reach in reach_by_pool.items():
        if len(reach) > len(best):
            best_name, best = name, reach
    return best_name, list(best)


def _admissible_prefix(behaviour: dict[int, float], horizons: tuple[int, ...]) -> list[int]:
    reach: list[int] = []
    for horizon in horizons:
        if float(behaviour.get(int(horizon), 0.0)) >= MIN_BEHAVIOR:
            reach.append(int(horizon))
        else:
            break
    return reach


# ------------------------------------------------------------------- phase one
def _phase_one_pool(
    adapter, cfg, pool_name: str, pool: tuple[str, ...], output_token_ids
) -> dict[str, Any]:
    """Behaviour and decodability only. No intervention is run here."""
    samples, frame, _stats, labels, by_id = _corpus(pool_name, pool, CANDIDATE_HORIZONS)
    sites = {str(s.sample_id): resolve_site_tokens(adapter.tokenizer, s) for s in samples}
    batch_size = int(cfg.runtime.batch_size)
    behaviour: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for horizon in CANDIDATE_HORIZONS:
        ids_by_split = {
            split: horizon_view(frame, split, horizon)["sample_id"].astype(str).tolist()
            for split in ("train", "validation", "discovery_test")
        }
        every = [sid for split in ids_by_split for sid in ids_by_split[split]]
        per_locus_d: dict[str, float] = {}
        accuracy = None
        distance = None
        for locus in LOCI:
            clean = _clean(
                adapter, by_id, every, sites, site=locus["site"], layer=locus["layer"],
                propagation=(), output_token_ids=output_token_ids, batch_size=batch_size,
            )
            blocks = {}
            for split, ids in ids_by_split.items():
                blocks[split] = (
                    np.stack([clean[sid]["site"] for sid in ids]),
                    np.asarray([labels[sid] for sid in ids], dtype=int),
                )
            fit = fit_probe(
                *blocks["train"], *blocks["validation"],
                c_grid=list(cfg.probe.C_grid), seed=PROBE_SEED,
            )
            per_locus_d[locus["name"]] = float(
                evaluate_probe(fit, *blocks["discovery_test"]).get("auroc") or float("nan")
            )
            if accuracy is None:
                test_ids = ids_by_split["discovery_test"]
                margins = np.asarray(
                    [selected_margin(clean[sid]["selected_logits"]) for sid in test_ids]
                )
                y_true = np.asarray([labels[sid] for sid in test_ids], dtype=int)
                accuracy = float(((margins >= 0).astype(int) == y_true).mean())
                distance = float(
                    np.mean(
                        [
                            sites[sid]["decision"][0] - sites[sid]["state_word_last"][0]
                            for sid in test_ids
                        ]
                    )
                )
                margin_auroc = classification_metrics(y_true, margins).get("auroc")
            del clean
            gc.collect()
        behaviour[int(horizon)] = float(accuracy)
        rows.append({
            "pool": pool_name, "horizon": int(horizon), "B": float(accuracy),
            "margin_auroc": margin_auroc, "mean_token_distance": distance,
            **{f"D_{name}": value for name, value in per_locus_d.items()},
        })
    reach = _admissible_prefix(behaviour, CANDIDATE_HORIZONS)
    return {"pool": pool_name, "rows": rows, "behaviour": behaviour, "reach": reach}


# ------------------------------------------------------------------- phase two
def _axes_and_targets(
    adapter, cfg, *, frame, by_id, labels, sites, horizons, output_token_ids
):
    """Horizon-local probe axis and validation setpoint targets at each locus."""
    axes: dict[tuple[str, int], np.ndarray] = {}
    targets: dict[tuple[str, int], dict[str, float]] = {}
    probe_records: list[dict[str, Any]] = []
    behaviour_records: list[dict[str, Any]] = []
    batch_size = int(cfg.runtime.batch_size)
    for locus in LOCI:
        for horizon in horizons:
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
                    np.stack([clean[sid]["site"] for sid in ids]),
                    np.asarray([labels[sid] for sid in ids], dtype=int),
                )
                for split, ids in ids_by_split.items()
            }
            fit = fit_probe(
                *blocks["train"], *blocks["validation"],
                c_grid=list(cfg.probe.C_grid), seed=PROBE_SEED,
            )
            axis = normalized_direction(raw_probe_direction(fit))
            axes[(locus["name"], int(horizon))] = axis
            metrics = evaluate_probe(fit, *blocks["discovery_test"])
            probe_records.append({
                "locus": locus["name"], "site": locus["site"], "layer": locus["layer"],
                "horizon": int(horizon), "test_auroc": metrics.get("auroc"),
                "test_balanced_accuracy": metrics.get("balanced_accuracy"),
            })
            val_ids = ids_by_split["validation"]
            targets[(locus["name"], int(horizon))] = validation_setpoint_targets(
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
    return axes, targets, probe_records, behaviour_records


def _factorial_sweep(
    adapter, cfg, *, frame, by_id, labels, sites, horizons, axes, targets,
    output_token_ids, diagnostics,
) -> list[dict[str, Any]]:
    """The frozen E19 arm set at every (locus, horizon)."""
    batch_size = int(cfg.runtime.batch_size)
    hidden = int(adapter.hidden_size)
    rows: list[dict[str, Any]] = []
    for locus in LOCI:
        for horizon in horizons:
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
                key = (locus["name"], int(horizon) if estimand == "native" else K0)
                axis = axes[key]
                target = targets[key]
                q_target = {
                    sid: float(target["q0_star"] if int(labels[sid]) == 1 else target["q1_star"])
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
                    setpoint_reference, context_reference = scalar, context
                plan.extend([
                    {"condition": "Y10_scalar", "estimand": estimand, "index": -1,
                     "deltas": scalar, "axis": axis, "q_target": q_target, "key": key},
                    {"condition": "Y01_context", "estimand": estimand, "index": -1,
                     "deltas": context, "axis": axis},
                    {"condition": "Y11_both", "estimand": estimand, "index": -1,
                     "deltas": {sid: scalar[sid] + context[sid] for sid in ids}, "axis": axis},
                ])
            native_axis = axes[(locus["name"], int(horizon))]
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
                        "delta_norm": delta_norm, "activation_norm": activation_norm,
                        "delta_over_activation_norm": delta_norm / max(activation_norm, 1e-12),
                    }
                    if arm["condition"] == "no_op":
                        diagnostics["no_op_max_margin_deviation"] = max(
                            diagnostics["no_op_max_margin_deviation"],
                            abs(margin_after - margin_before),
                        )
                    if arm["condition"] in ("random_norm_matched", "orthogonal_random"):
                        reference = float(np.linalg.norm(
                            setpoint_reference[sid]
                            if arm["condition"] == "random_norm_matched"
                            else context_reference[sid]
                        ))
                        diagnostics["max_norm_match_relative_deviation"] = max(
                            diagnostics["max_norm_match_relative_deviation"],
                            abs(delta_norm - reference) / max(reference, 1e-12),
                        )
                    if arm["condition"] == "Y10_scalar":
                        fidelity = setpoint_identity_diagnostics(
                            base, after, arm["axis"], float(arm["q_target"][sid])
                        )
                        sigma_q = float(targets[arm["key"]]["sigma_q_validation"])
                        diagnostics["max_projection_relative_deviation"] = max(
                            diagnostics["max_projection_relative_deviation"],
                            abs(fidelity["q_after"] - fidelity["q_target"]) / max(sigma_q, 1e-12),
                        )
                        diagnostics["max_orthogonal_relative_deviation"] = max(
                            diagnostics["max_orthogonal_relative_deviation"],
                            fidelity["orthogonal_relative_deviation"],
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
    return rows


def run_e20() -> Path:
    """Phase 1 non-causal range selection, then the frozen E19 sweep."""
    cfg, provenance = _config()
    run_dir = campaign_dir() / MODEL
    run_dir.mkdir(parents=True, exist_ok=True)
    status = StatusFile.create(run_dir, run_id=f"E20-{MODEL}", experiment_id="E20")
    manifest = RunManifest(run_dir)
    adapter = None
    try:
        manifest.set_start(
            config_hash(cfg),
            {**provenance, "e20_version": E20_VERSION,
             "candidate_horizons": list(CANDIDATE_HORIZONS),
             "pools": [name for name, _p in POOLS]},
            {"probe": PROBE_SEED, "direction_base": DIRECTION_SEED_BASE,
             "bootstrap": BOOTSTRAP_SEED,
             "corpus": {n: s for n, _c, s in CORPUS_SPECS}},
        )
        save_resolved_config(cfg, run_dir / "resolved_config.yaml", provenance)
        adapter = load_adapter(cfg)
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        manifest.update_model_info(
            id=adapter.display_model_id, dtype=str(cfg.model.dtype),
            num_layers=adapter.num_layers, hidden_size=adapter.hidden_size,
            candidate_token_ids=output_token_ids,
        )

        # ---------------- Phase 1: non-causal only.
        phase_one = {}
        for pool_name, pool in POOLS:
            logger.info("E20 phase 1: %s pool", pool_name)
            phase_one[pool_name] = _phase_one_pool(
                adapter, cfg, pool_name, pool, output_token_ids
            )
        reach_by_pool = {name: entry["reach"] for name, entry in phase_one.items()}
        selected_pool, grid = select_grid(reach_by_pool)
        phase_one_table = pd.DataFrame(
            [row for entry in phase_one.values() for row in entry["rows"]]
        )
        save_table(phase_one_table, run_dir / "phase1_non_causal.parquet")
        save_json(
            {
                "candidate_horizons": list(CANDIDATE_HORIZONS),
                "reach_by_pool": reach_by_pool,
                "selected_pool": selected_pool,
                "selected_grid": grid,
                "min_behavior": MIN_BEHAVIOR,
                "causal_quantities_inspected": False,
                "table": phase_one_table.to_dict(orient="records"),
            },
            run_dir / "phase1_selection.json",
        )
        logger.info("E20 selected pool=%s grid=%s", selected_pool, grid)

        extends_beyond_e19 = bool(len(grid) > 4)
        if not extends_beyond_e19:
            summary = {
                "e20_version": E20_VERSION,
                "model": adapter.display_model_id,
                "phase1": phase_one_table.to_dict(orient="records"),
                "reach_by_pool": reach_by_pool,
                "selected_pool": selected_pool,
                "selected_grid": grid,
                "outcome": "stopped_after_phase1_no_longer_interpretable_horizon",
                "confirmation_accessed": False,
            }
            save_json(summary, run_dir / "e20_summary.json")
            atomic_write_json(campaign_dir() / "E20_LONG_HORIZON.json", summary)
            manifest.finish([{"stage": "phase1", "outcome": summary["outcome"]}])
            status.complete("E20 stopped after Phase 1 by the frozen stop rule")
            return run_dir

        # ---------------- Phase 2: the frozen E19 sweep on the selected grid.
        pool = dict(POOLS)[selected_pool]
        samples, frame, stats, labels, by_id = _corpus(selected_pool, pool, tuple(grid))
        sites = {str(s.sample_id): resolve_site_tokens(adapter.tokenizer, s) for s in samples}
        save_json(stats, run_dir / "corpus_statistics.json")
        manifest.update_dataset_info(
            split_hash=dataset_split_hash(
                dict(zip(frame["sample_id"].astype(str), frame["split"].astype(str)))
            ),
            prompt_hash_sample=prompt_hash(str(frame["prompt"].iloc[0])),
            confirmation_accessed=False,
        )

        axes, targets, probe_records, behaviour_records = _axes_and_targets(
            adapter, cfg, frame=frame, by_id=by_id, labels=labels, sites=sites,
            horizons=grid, output_token_ids=output_token_ids,
        )
        tolerances = setpoint_fidelity_tolerances(str(cfg.model.dtype))
        diagnostics: dict[str, Any] = {
            "no_op_max_margin_deviation": 0.0,
            "max_norm_match_relative_deviation": 0.0,
            "max_projection_relative_deviation": 0.0,
            "max_orthogonal_relative_deviation": 0.0,
            "hook_leak_detected": False,
            "tolerances": tolerances,
        }
        rows = _factorial_sweep(
            adapter, cfg, frame=frame, by_id=by_id, labels=labels, sites=sites,
            horizons=grid, axes=axes, targets=targets,
            output_token_ids=output_token_ids, diagnostics=diagnostics,
        )
        diagnostics["hook_leak_detected"] = any(
            resid_post_hook_count(adapter, layer=int(l["layer"])) for l in LOCI
        )
        raw = pd.DataFrame(rows)
        if not np.isfinite(raw["delta_margin_toward_expected"].to_numpy(float)).all():
            raise RuntimeError("non-finite E20 evidence produced")
        np.savez(
            run_dir / "probe_axes.npz",
            **{f"{name}__k{horizon}": axis for (name, horizon), axis in axes.items()},
        )
        save_table(raw, run_dir / "intervention_rows.parquet")
        save_table(pd.DataFrame(probe_records), run_dir / "probe_metrics.parquet")
        save_table(pd.DataFrame(behaviour_records), run_dir / "behaviour_by_horizon.parquet")
        save_json(diagnostics, run_dir / "diagnostics.json")

        analysis = analyze_rows(
            raw, probe_records, axes, list(grid),
            bootstraps=int(cfg.statistics.bootstrap_samples),
            confidence=float(cfg.statistics.confidence_level),
            enforce_magnitude_gate=True,
            bootstrap_seed=BOOTSTRAP_SEED,
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
            "G2_selected_grid": list(grid),
            "G2_reach_by_pool": reach_by_pool,
            "G4_enforced": True,
            **analysis.get("gates", {}),
        }
        summary = {
            "e20_version": E20_VERSION,
            "model": adapter.display_model_id,
            "selected_pool": selected_pool,
            "selected_grid": list(grid),
            "phase1": phase_one_table.to_dict(orient="records"),
            "reach_by_pool": reach_by_pool,
            "loci": [dict(l) for l in LOCI],
            "horizons": list(grid),
            "diagnostics": diagnostics,
            "confirmation_accessed": False,
            **analysis,
        }
        save_json(summary, run_dir / "e20_summary.json")
        atomic_write_json(campaign_dir() / "E20_LONG_HORIZON.json", summary)
        manifest.finish([{"stage": "e20", "outcome": summary.get("outcome")}])
        if not numerics_ok:
            status.fail("E20 numerics gate G1 failed")
            raise RuntimeError(f"E20 G1 failed: {diagnostics}")
        status.complete(f"E20 complete: {summary.get('outcome')}")
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


def reanalyze_e20(model_name: str = MODEL) -> Path:
    """Re-run the E20 analysis from persisted rows, probes and axes. No GPU."""
    run_dir = campaign_dir() / model_name
    raw = pd.read_parquet(run_dir / "intervention_rows.parquet")
    probes = pd.read_parquet(run_dir / "probe_metrics.parquet").to_dict(orient="records")
    with np.load(run_dir / "probe_axes.npz") as archive:
        axes = {}
        for key in archive.files:
            name, _, horizon = key.partition("__k")
            axes[(name, int(horizon))] = np.asarray(archive[key], dtype=np.float64)
    previous = json.loads((run_dir / "e20_summary.json").read_text(encoding="utf-8"))
    grid = [int(k) for k in previous["selected_grid"]]
    cfg, _provenance = _config()
    analysis = analyze_rows(
        raw, probes, axes, grid,
        bootstraps=int(cfg.statistics.bootstrap_samples),
        confidence=float(cfg.statistics.confidence_level),
        enforce_magnitude_gate=True,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    summary = {**previous, **analysis}
    summary["gates"] = {**previous.get("gates", {}), **analysis.get("gates", {})}
    save_json(summary, run_dir / "e20_summary.json")
    atomic_write_json(campaign_dir() / "E20_LONG_HORIZON.json", summary)
    return run_dir


def read_e20() -> dict[str, Any] | None:
    path = campaign_dir() / "E20_LONG_HORIZON.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
