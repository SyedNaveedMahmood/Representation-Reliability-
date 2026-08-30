"""E15 Gate 1 - carrier causal sufficiency.

`docs/Reproduction_Reliability_Next_Direction_Review.md` section 10.2 requires a
carrier to pass a sufficiency gate *before* any `Q/A/G` interpretation:

    full-state counterfactual patching at the bottleneck must change the delayed
    decision strongly; a no-op patch must be numerically null; a random
    same-norm patch must be weak.

E15's Stage 3 produced a null for the source-free scalar setpoint. Section 10.10
of that review lists two very different diagnoses that the Stage 3 evidence alone
cannot separate:

    full patch STRONG, Q/A/G weak -> the causal code is outside the linear
                                     factorial decomposition
    full patch WEAK               -> the carrier is not causally sufficient;
                                     redesign rather than interpret

This runner adds exactly the missing arm. It is a **diagnostic addendum** to E15,
not a redesign: same frozen corpus, same frozen site (`resid_post` L17), same
frozen carrier token, same discovery-test episodes, same readout. Nothing about
the E15 protocol is re-opened and no new hypothesis is introduced.

Arms (frozen before execution):

```text
no_op                       zero delta                       numerical contract
setpoint                    the frozen E15 treatment         in-run reference
full_state_patch            h_twin_carrier - h_base_carrier  UPPER BOUND
full_patch_random           same-norm random direction, 5 seeds
```

The primary gate is evaluated at `k0 = 1`; the remaining interpretable horizons
are reported descriptively.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..interventions.setpoint import norm_matched_direction_delta, source_free_setpoint_delta
from ..interventions.truth_coordinate import (
    coordinate_value,
    normalized_direction,
    random_unit_direction,
)
from ..metrics.causal import counterfactual_outcome, margin_toward_label
from ..metrics.temporal_half_life import horizon_condition_summary, paired_horizon_contrast
from ..reporting.tables import save_json, save_table
from .e00c import candidate_first_token_ids
from .e01a import _prediction
from .e15 import DECISION_LAYERS, PRIMARY_MODEL, _config, _load_stage1, _release, _start_run
from .e15_support import (
    BOOTSTRAP_SEED,
    CARRIER_LAYER,
    DIRECTION_SEED_BASE,
    K0,
    NO_OP_TOLERANCE,
    PROPAGATION_LAYERS,
    behavior_metrics,
    corpus_bundle,
    horizon_view,
    resolve_site_table,
    run_clean_batches,
    run_edit_batches,
    selected_margin,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)

GATE1_VERSION = "e15-gate1-carrier-sufficiency-v1"

# Frozen: the primary gate horizon, and the descriptive ones.
GATE_HORIZON = K0
DESCRIPTIVE_HORIZONS: tuple[int, ...] = (2, 4, 8)
N_FULL_PATCH_RANDOM = 5
# Frozen seed block, disjoint from the E15 Stage 3 control seeds
# (DIRECTION_SEED_BASE + 0..4 and +100..104) so no control direction is reused.
FULL_PATCH_RANDOM_SEED_BASE = DIRECTION_SEED_BASE + 200


def _twin_carrier_states(
    clean: dict[str, dict[str, Any]],
    view: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Map each base sample to its matched twin's carrier state.

    Twins share a ``pair_id``, differ only in the clearance word, and E15 Stage 0
    verified that they have identical token length and identical carrier index.
    The difference of their carrier states is therefore the exact full-state
    counterfactual displacement at the frozen carrier.
    """
    by_pair: dict[str, list[str]] = {}
    for row in view.to_dict("records"):
        by_pair.setdefault(str(row["pair_id"]), []).append(str(row["sample_id"]))
    twins: dict[str, np.ndarray] = {}
    for pair_id, ids in by_pair.items():
        if len(ids) != 2:
            raise RuntimeError(f"pair {pair_id} is not complete in the gate view")
        first, second = sorted(ids)
        twins[first] = clean[second]["sites"]["carrier"].copy()
        twins[second] = clean[first]["sites"]["carrier"].copy()
    if set(twins) != set(view["sample_id"].astype(str)):
        raise RuntimeError("twin carrier map lost sample identity")
    return twins


def run_e15_gate1(model_name: str = PRIMARY_MODEL) -> Path:
    """Run the carrier-sufficiency arms and emit the frozen Gate 1 verdict."""
    cfg, provenance = _config(model_name)
    summary1, arrays = _load_stage1(model_name)
    horizons = [int(k) for k in summary1["gate"]["G1c_interpretable_horizons"]]
    if GATE_HORIZON not in horizons:
        raise RuntimeError("the Gate 1 horizon is not in the interpretable grid")
    run_horizons = [GATE_HORIZON] + [k for k in DESCRIPTIVE_HORIZONS if k in horizons]

    run_dir, status, manifest = _start_run(
        "gate1", model_name, cfg, provenance,
        {
            "gate1_version": GATE1_VERSION,
            "gate_horizon": GATE_HORIZON,
            "run_horizons": run_horizons,
            "purpose": "carrier causal sufficiency; diagnostic addendum to E15",
        },
    )
    adapter = None
    try:
        samples, frame, _stats, labels, by_id = corpus_bundle()
        adapter = load_adapter(cfg)
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        sites = resolve_site_table(adapter, samples)
        u = normalized_direction(arrays["carrier_direction"])
        setpoints = summary1["validation_setpoints"]

        rows: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {
            "no_op_max_margin_deviation": 0.0,
            "max_norm_match_relative_deviation": 0.0,
            "hook_leak_detected": False,
        }
        behaviour: dict[str, Any] = {}

        for horizon in run_horizons:
            view = horizon_view(frame, "discovery_test", int(horizon))
            ids = view["sample_id"].astype(str).tolist()
            meta = view.set_index("sample_id")

            clean = run_clean_batches(
                adapter, by_id, ids, sites,
                output_token_ids=output_token_ids,
                decision_layers=PROPAGATION_LAYERS,
                batch_size=int(cfg.runtime.batch_size),
            )
            clean_margin = {sid: selected_margin(clean[sid]["selected_logits"]) for sid in ids}
            behaviour[str(int(horizon))] = behavior_metrics(clean, labels, ids)
            twin_carrier = _twin_carrier_states(clean, view)

            q_target = {
                sid: (
                    float(setpoints["q0_star"])
                    if int(labels[sid]) == 1
                    else float(setpoints["q1_star"])
                )
                for sid in ids
            }
            full_patch_delta = {
                sid: twin_carrier[sid] - clean[sid]["sites"]["carrier"] for sid in ids
            }
            setpoint_delta = {
                sid: source_free_setpoint_delta(
                    clean[sid]["sites"]["carrier"], u, q_target[sid]
                )
                for sid in ids
            }

            plan: list[dict[str, Any]] = [
                {"condition": "no_op", "direction_index": -1, "kind": "zero"},
                {"condition": "setpoint", "direction_index": -1, "kind": "setpoint"},
                {"condition": "full_state_patch", "direction_index": -1, "kind": "full"},
            ]
            for index in range(N_FULL_PATCH_RANDOM):
                plan.append(
                    {
                        "condition": "full_patch_random",
                        "direction_index": index,
                        "kind": "full_norm_matched",
                        "direction": random_unit_direction(
                            adapter.hidden_size, FULL_PATCH_RANDOM_SEED_BASE + index
                        ),
                    }
                )

            for arm in plan:
                kind = str(arm["kind"])
                deltas: dict[str, np.ndarray] = {}
                for sid in ids:
                    if kind == "zero":
                        deltas[sid] = np.zeros_like(clean[sid]["sites"]["carrier"])
                    elif kind == "setpoint":
                        deltas[sid] = setpoint_delta[sid]
                    elif kind == "full":
                        deltas[sid] = full_patch_delta[sid]
                    else:
                        deltas[sid] = norm_matched_direction_delta(
                            full_patch_delta[sid], arm["direction"]
                        )

                edited = run_edit_batches(
                    adapter, by_id, ids, sites,
                    edit_site="carrier",
                    deltas_by_id=deltas,
                    output_token_ids=output_token_ids,
                    propagation_layers=PROPAGATION_LAYERS,
                    batch_size=int(cfg.runtime.batch_size),
                )

                for sid in ids:
                    base = clean[sid]["sites"]["carrier"]
                    after_state = edited[sid]["edited_carrier_state"]
                    delta = np.asarray(deltas[sid], dtype=np.float64)
                    margin_before = clean_margin[sid]
                    margin_after = selected_margin(edited[sid]["selected_logits"])
                    label = int(labels[sid])
                    expected = 1 - label
                    outcome = counterfactual_outcome(
                        _prediction(margin_before), _prediction(margin_after), expected
                    )
                    delta_norm = float(np.linalg.norm(delta))
                    activation_norm = float(np.linalg.norm(base))
                    record = {
                        "horizon": int(horizon),
                        "condition": str(arm["condition"]),
                        "direction_index": int(arm["direction_index"]),
                        "base_sample_id": sid,
                        "pair_id": str(meta.loc[sid, "pair_id"]),
                        "episode_index": int(meta.loc[sid, "episode_index"]),
                        "target_label": label,
                        "expected_label": expected,
                        "margin_before": margin_before,
                        "margin_after": margin_after,
                        "delta_margin_raw": float(margin_after - margin_before),
                        "delta_margin_toward_expected": float(
                            margin_toward_label(margin_after, expected)
                            - margin_toward_label(margin_before, expected)
                        ),
                        "prediction_before": _prediction(margin_before),
                        "prediction_after": _prediction(margin_after),
                        "expected_label_after": outcome["expected_label_after"],
                        "counterfactual_flip": outcome["counterfactual_flip"],
                        "delta_norm": delta_norm,
                        "activation_norm": activation_norm,
                        "delta_over_activation_norm": delta_norm / max(activation_norm, 1e-12),
                        "q_base": coordinate_value(base, u),
                        "q_after": coordinate_value(after_state, u),
                        "full_patch_norm": float(np.linalg.norm(full_patch_delta[sid])),
                    }
                    for layer in PROPAGATION_LAYERS:
                        clean_state = clean[sid]["sites"][f"decision_l{layer}"]
                        edited_state = edited[sid]["propagation"][int(layer)]
                        record[f"p_norm_l{layer}"] = float(
                            np.linalg.norm(edited_state - clean_state)
                            / max(float(np.linalg.norm(clean_state)), 1e-12)
                        )
                    if kind == "zero":
                        diagnostics["no_op_max_margin_deviation"] = max(
                            diagnostics["no_op_max_margin_deviation"],
                            abs(margin_after - margin_before),
                        )
                    if kind == "full_norm_matched":
                        reference = float(np.linalg.norm(full_patch_delta[sid]))
                        diagnostics["max_norm_match_relative_deviation"] = max(
                            diagnostics["max_norm_match_relative_deviation"],
                            abs(delta_norm - reference) / max(reference, 1e-12),
                        )
                    rows.append(record)
                del edited
            del clean
            gc.collect()

        from ..adapters.intervention import resid_post_hook_count

        diagnostics["hook_leak_detected"] = any(
            resid_post_hook_count(adapter, layer=layer) for layer in DECISION_LAYERS
        )

        frame_rows = pd.DataFrame(rows)
        if not np.isfinite(frame_rows["delta_margin_toward_expected"].to_numpy(float)).all():
            raise RuntimeError("non-finite Gate 1 evidence produced")
        save_table(frame_rows, run_dir / "gate1_rows.parquet")

        bootstraps = int(cfg.statistics.bootstrap_samples)
        confidence = float(cfg.statistics.confidence_level)
        summary = horizon_condition_summary(
            frame_rows, n_bootstraps=bootstraps, confidence_level=confidence,
            seed=BOOTSTRAP_SEED + 500,
        )
        save_table(summary, run_dir / "gate1_condition_summary.parquet")

        contrasts = [
            paired_horizon_contrast(
                frame_rows, horizon=int(horizon), treatment="full_state_patch",
                control=control, n_bootstraps=bootstraps, confidence_level=confidence,
                seed=BOOTSTRAP_SEED + 600 + 7 * int(horizon) + len(control),
            )
            for horizon in run_horizons
            for control in ("full_patch_random", "setpoint")
        ]

        indexed = summary.set_index(["horizon", "condition"])
        full_k0 = indexed.loc[(GATE_HORIZON, "full_state_patch")]
        random_k0 = indexed.loc[(GATE_HORIZON, "full_patch_random")]
        vs_random_k0 = next(
            c for c in contrasts
            if c["horizon"] == GATE_HORIZON and c["control"] == "full_patch_random"
        )

        gate = {
            "gate_horizon": GATE_HORIZON,
            "full_patch_effect": float(full_k0["mean_effect"]),
            "full_patch_ci": [float(full_k0["ci_low"]), float(full_k0["ci_high"])],
            "full_patch_ci_excludes_zero": bool(
                float(full_k0["ci_low"]) > 0.0 or float(full_k0["ci_high"]) < 0.0
            ),
            "full_patch_flip_rate": float(
                frame_rows[
                    (frame_rows["horizon"] == GATE_HORIZON)
                    & (frame_rows["condition"] == "full_state_patch")
                ]["counterfactual_flip"].mean()
            ),
            "same_norm_random_effect": float(random_k0["mean_effect"]),
            "full_patch_minus_random": vs_random_k0["mean_difference"],
            "full_patch_minus_random_ci": [vs_random_k0["ci_low"], vs_random_k0["ci_high"]],
            "mean_full_patch_residual_fraction": float(
                frame_rows[
                    (frame_rows["horizon"] == GATE_HORIZON)
                    & (frame_rows["condition"] == "full_state_patch")
                ]["delta_over_activation_norm"].mean()
            ),
            "no_op_max_margin_deviation": diagnostics["no_op_max_margin_deviation"],
            "no_op_passed": bool(
                diagnostics["no_op_max_margin_deviation"] <= NO_OP_TOLERANCE
            ),
            "norm_match_passed": bool(
                diagnostics["max_norm_match_relative_deviation"] <= 1e-6
            ),
            "hook_leak_detected": diagnostics["hook_leak_detected"],
        }
        gate["G1a_full_patch_effective"] = bool(
            gate["full_patch_effect"] > 0.0 and gate["full_patch_ci_excludes_zero"]
        )
        gate["G1b_exceeds_same_norm_random"] = bool(vs_random_k0["ci_low"] > 0.0)
        gate["G1c_numerics_passed"] = bool(
            gate["no_op_passed"] and gate["norm_match_passed"]
            and not gate["hook_leak_detected"]
        )
        gate["carrier_causally_sufficient"] = bool(
            gate["G1a_full_patch_effective"]
            and gate["G1b_exceeds_same_norm_random"]
            and gate["G1c_numerics_passed"]
        )
        gate["diagnosis"] = (
            "causal_code_outside_linear_factorial_decomposition"
            if gate["carrier_causally_sufficient"]
            else "carrier_not_causally_sufficient_redesign_required"
        )

        payload = {
            "gate1_version": GATE1_VERSION,
            "model": adapter.display_model_id,
            "site": f"resid_post/L{CARRIER_LAYER}",
            "horizons": run_horizons,
            "gate": gate,
            "contrasts": contrasts,
            "clean_behaviour": behaviour,
            "diagnostics": diagnostics,
            "condition_means": summary.to_dict(orient="records"),
        }
        save_json(payload, run_dir / "gate1_summary.json")
        manifest.finish([{"stage": "gate1", "gate": gate}])
        status.complete(
            "Gate 1 complete: carrier "
            + ("IS" if gate["carrier_causally_sufficient"] else "is NOT")
            + " causally sufficient"
        )
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            _release(adapter)


def read_gate1(model_name: str = PRIMARY_MODEL) -> dict[str, Any] | None:
    """Return the Gate 1 verdict for a model, or None if it has not been run."""
    from .e15 import _stage_dir

    summary = _stage_dir("gate1", model_name) / "gate1_summary.json"
    if not summary.exists():
        return None
    return json.loads(summary.read_text(encoding="utf-8"))
