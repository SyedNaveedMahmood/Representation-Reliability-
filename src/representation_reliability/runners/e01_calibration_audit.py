"""E01 cross-checkpoint intervention calibration audit.

Required by `docs/Reproduction_Reliability_Next_Direction_Review.md` section 8.
Wu, Zhao and Chen (2026) show that raw-unit interventions at a fixed layer and
coefficient can manufacture an apparent cross-model scaling trend. E01's headline
`Q0` contrast (0.0144 in Qwen3-0.6B versus 0.7013 in Qwen3-1.7B) is measured at a
single operating point defined by validation class medians, so it inherits that
risk. This audit re-parameterizes the same frozen intervention by **residual
fraction**

```text
r = ||delta h|| / ||h||
```

and reports effect curves `Q(r)`, `A(r)`, `G(r)` for both checkpoints instead of
one operating point.

Nothing is retuned and the consumed E01 confirmation holdout is never touched:

* frozen dataset, frozen split assignment, discovery rows only;
* frozen site `resid_post` L17, frozen selector `last_prompt`;
* frozen probe recipe (train fit, validation-only C selection);
* frozen semantic direction (opposite-class median target defines the *sign*);
* frozen matched-orthogonal context construction from the counterfactual twin;
* frozen 150-pair discovery-test evaluation set.

Only the intervention **magnitude parameterization** changes, which is exactly
the audit's purpose. The audit is descriptive; it makes no confirmatory claim.
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

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.splits import build_discovery_label_map, discovery_view
from ..interventions.orthogonal_context import orthogonal_component
from ..interventions.truth_coordinate import (
    coordinate_value,
    normalized_direction,
    random_unit_direction,
)
from ..metrics.causal import cluster_bootstrap_mean_ci, margin_toward_label
from ..metrics.setpoint import validation_setpoint_targets
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.status import StatusFile, atomic_write_json
from .e00c import _generate_dataset, candidate_token_id_lists
from .e01a import _layer_probe, _prediction, _selected_margin
from .e01a_support import (
    deterministic_subset_pair_ids,
    extract_resid_post_layers,
    run_intervention_batches,
    run_unintervened_batches,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)

AUDIT_VERSION = "e01-calibration-audit-v1"

SITE = "resid_post"
SELECTOR = "last_prompt"
LAYER = 17                       # frozen E01 site; no layer search
MODELS: tuple[str, ...] = ("qwen3_0.6b", "qwen3_1.7b")

# Predeclared residual-fraction grid (review section 8.2). Chosen to bracket the
# residual fractions the original E01 operating point actually produced in both
# checkpoints, on a roughly logarithmic scale. Frozen before execution.
RESIDUAL_FRACTIONS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40)

N_RANDOM_DIRECTIONS = 5
RANDOM_SEED_BASE = 20260901
BOOTSTRAP_SEED = 20260911
KNN_K = 10


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def audit_dir() -> Path:
    path = repo_root() / "runs" / "E01_CALIBRATION_AUDIT"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config(model_name: str):
    root = repo_root()
    return resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / f"configs/models/{model_name}.yaml",
        experiment_path=root / "configs/experiments/E01B3_factorial_decomposition.yaml",
        overrides=(),
    )


# ------------------------------------------------------------------ diagnostics
def _pairwise_distance_scales(
    states: np.ndarray, labels: np.ndarray, *, seed: int, n_pairs: int = 4000
) -> dict[str, float]:
    """Within-class and between-class activation distance scales (review 8.2.2)."""
    rng = np.random.default_rng(int(seed))
    n = len(states)
    same: list[float] = []
    diff: list[float] = []
    for _ in range(int(n_pairs)):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        distance = float(np.linalg.norm(states[i] - states[j]))
        (same if labels[i] == labels[j] else diff).append(distance)
    return {
        "within_class_mean_distance": float(np.mean(same)) if same else float("nan"),
        "between_class_mean_distance": float(np.mean(diff)) if diff else float("nan"),
        "activation_norm_mean": float(np.linalg.norm(states, axis=1).mean()),
    }


def _diag_mahalanobis(state: np.ndarray, mean: np.ndarray, inv_var: np.ndarray) -> float:
    """Diagonal-covariance Mahalanobis distance.

    A full covariance is not estimable here (hidden width exceeds the row count),
    so the diagnostic is deliberately the diagonal form and is labelled as such.
    """
    delta = np.asarray(state, dtype=np.float64) - mean
    return float(np.sqrt(np.sum(delta * delta * inv_var)))


def _knn_distance(state: np.ndarray, cloud: np.ndarray, k: int) -> float:
    distances = np.linalg.norm(cloud - state[None, :], axis=1)
    partition = np.partition(distances, k)[:k]
    return float(partition.mean())


# --------------------------------------------------------------------- one model
def _audit_one_model(model_name: str) -> Path:
    cfg, provenance = _config(model_name)
    run_dir = audit_dir() / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    status = StatusFile.create(run_dir, run_id=f"E01AUDIT-{model_name}", experiment_id="E01AUDIT")
    manifest = RunManifest(run_dir)

    adapter = None
    try:
        samples, full_df = _generate_dataset(cfg)
        # discovery_view REMOVES the confirmation rows entirely; they are never
        # loaded, labelled, or evaluated anywhere in this audit.
        discovery_df = discovery_view(full_df).reset_index(drop=True)
        if "confirmation" in set(discovery_df["split"].astype(str)):
            raise RuntimeError("confirmation rows leaked into the audit view")
        label_map = build_discovery_label_map(discovery_df)
        discovery_ids = discovery_df["sample_id"].astype(str).tolist()
        keep = set(discovery_ids)
        samples_by_id = {
            str(s.sample_id): s for s in samples if str(s.sample_id) in keep
        }
        split_of = dict(
            zip(discovery_df["sample_id"].astype(str), discovery_df["split"].astype(str))
        )
        selected_pairs = deterministic_subset_pair_ids(
            discovery_df, max_pairs=None, seed=int(cfg.reproducibility.control_seed)
        )
        base_df = discovery_df[
            (discovery_df["split"].astype(str) == "discovery_test")
            & discovery_df["pair_id"].astype(str).isin(set(selected_pairs))
        ].copy()
        base_ids = base_df["sample_id"].astype(str).tolist()
        meta = base_df.set_index("sample_id")

        manifest.set_start(
            config_hash(cfg),
            {**provenance, "audit_version": AUDIT_VERSION, "model": model_name,
             "residual_fractions": list(RESIDUAL_FRACTIONS)},
            cfg.effective_seeds(),
        )
        save_resolved_config(cfg, run_dir / "resolved_config.yaml", provenance)

        adapter = load_adapter(cfg)
        if not 0 <= LAYER < adapter.num_layers:
            raise RuntimeError(f"frozen layer {LAYER} out of range for {model_name}")
        candidates = list(cfg.behavior.candidates_primary)
        candidate_lists = candidate_token_id_lists(adapter, candidates)
        if len(candidates) != 2 or any(len(ids) != 1 for ids in candidate_lists):
            raise RuntimeError("the audit requires single-token Yes/No candidates")
        output_token_ids = [int(ids[0]) for ids in candidate_lists]
        revisions = adapter.resolved_revisions()
        manifest.update_model_info(
            id=adapter.display_model_id,
            dtype=str(cfg.model.dtype),
            num_layers=adapter.num_layers,
            hidden_size=adapter.hidden_size,
            candidate_token_ids=output_token_ids,
            resolved_native_modules={
                f"{SITE}:{LAYER}": adapter.resolve_site(SITE, LAYER).native_module_name
            },
            notes={"revisions": revisions},
        )
        manifest.update_dataset_info(
            split_hash=dataset_split_hash(split_of),
            prompt_hash_sample=prompt_hash(str(discovery_df.iloc[0]["prompt"])),
            confirmation_accessed=False,
        )

        activations, token_indices, token_sites = extract_resid_post_layers(
            adapter,
            [samples_by_id[sid] for sid in discovery_ids],
            layers=[LAYER],
            token_selector=SELECTOR,
            batch_size=int(cfg.runtime.batch_size),
        )
        if any(
            token_indices[sid] != int(token_sites[sid]["sequence_length"]) - 1
            for sid in base_ids
        ):
            raise RuntimeError("last_prompt did not resolve to the final prompt token")

        _fit, u, probe_metrics = _layer_probe(
            LAYER, activations, discovery_df, label_map,
            c_grid=cfg.probe.C_grid, seed=int(cfg.reproducibility.probe_seed),
        )
        u = normalized_direction(u)

        validation_ids = discovery_df.loc[
            discovery_df["split"].astype(str) == "validation", "sample_id"
        ].astype(str).tolist()
        validation_states = np.stack([activations[LAYER][sid] for sid in validation_ids])
        validation_clean = run_unintervened_batches(
            adapter, samples_by_id, validation_ids,
            token_indices=token_indices, output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        setpoints = validation_setpoint_targets(
            validation_states @ u,
            [int(label_map[sid]) for sid in validation_ids],
            [_selected_margin(validation_clean[sid]) for sid in validation_ids],
        )
        sigma_q = float(setpoints["sigma_q_validation"])
        sigma_m = float(setpoints["sigma_margin_validation"])

        # On-manifold reference cloud and scales, from validation only.
        cloud_mean = validation_states.mean(axis=0)
        cloud_var = validation_states.var(axis=0, ddof=1)
        inv_var = 1.0 / np.maximum(cloud_var, 1e-8)
        scales = _pairwise_distance_scales(
            validation_states,
            np.asarray([int(label_map[sid]) for sid in validation_ids]),
            seed=int(cfg.reproducibility.control_seed),
        )

        # Clean pass at the exact intervention batch shape.
        hidden_dim = int(adapter.hidden_size)
        zeros = {sid: np.zeros(hidden_dim, dtype=np.float64) for sid in base_ids}
        clean = run_intervention_batches(
            adapter, samples_by_id, base_ids, layer=LAYER,
            token_indices=token_indices, deltas_by_id=zeros,
            output_token_ids=output_token_ids, capture_layers=[LAYER],
            batch_size=int(cfg.runtime.batch_size),
        )
        unhooked = run_unintervened_batches(
            adapter, samples_by_id, base_ids,
            token_indices=token_indices, output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        bases = {sid: clean[sid]["captured"][LAYER].copy() for sid in base_ids}
        clean_margin = {sid: _selected_margin(clean[sid]["selected_logits"]) for sid in base_ids}
        no_op_deviation = max(
            abs(clean_margin[sid] - _selected_margin(unhooked[sid])) for sid in base_ids
        )

        # Frozen context construction: the matched counterfactual twin's
        # displacement, projected orthogonal to the frozen probe direction.
        matched_context: dict[str, np.ndarray] = {}
        same_label_context: dict[str, np.ndarray] = {}
        shuffled_context: dict[str, np.ndarray] = {}
        full_patch_direction: dict[str, np.ndarray] = {}
        by_label: dict[int, list[str]] = {0: [], 1: []}
        for sid in base_ids:
            by_label[int(label_map[sid])].append(sid)
        rng = np.random.default_rng(int(cfg.reproducibility.control_seed))
        for sid in base_ids:
            twin = str(samples_by_id[sid].counterfactual_id)
            if twin not in activations[LAYER]:
                raise RuntimeError(f"counterfactual twin missing for {sid}")
            matched_context[sid] = orthogonal_component(
                activations[LAYER][twin], bases[sid], u
            )
            full_patch_direction[sid] = activations[LAYER][twin] - bases[sid]
            pool_same = [x for x in by_label[int(label_map[sid])] if x != sid]
            pool_diff = [
                x for x in by_label[1 - int(label_map[sid])]
                if str(samples_by_id[sid].counterfactual_id) != x
            ]
            same_label_context[sid] = orthogonal_component(
                activations[LAYER][pool_same[int(rng.integers(len(pool_same)))]], bases[sid], u
            )
            shuffled_context[sid] = orthogonal_component(
                activations[LAYER][pool_diff[int(rng.integers(len(pool_diff)))]], bases[sid], u
            )

        def unit(vec: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 1e-9:
                raise RuntimeError("degenerate direction in the audit plan")
            return vec / norm

        random_dirs = [
            random_unit_direction(hidden_dim, RANDOM_SEED_BASE + i)
            for i in range(N_RANDOM_DIRECTIONS)
        ]
        random_orth_dirs = [
            random_unit_direction(hidden_dim, RANDOM_SEED_BASE + 100 + i, orthogonal_to=u)
            for i in range(N_RANDOM_DIRECTIONS)
        ]

        # The clean-state manifold diagnostics do not depend on the arm, so they
        # are computed once per example rather than once per intervention row.
        base_mahalanobis = {
            sid: _diag_mahalanobis(bases[sid], cloud_mean, inv_var) for sid in base_ids
        }
        base_knn = {
            sid: _knn_distance(bases[sid], validation_states, KNN_K) for sid in base_ids
        }

        rows: list[dict[str, Any]] = []
        max_residual_error = 0.0

        for r in RESIDUAL_FRACTIONS:
            # Frozen per-example magnitude: exactly r * ||h|| for every one-component
            # arm, so every arm at a given r is norm-matched by construction.
            scale = {sid: float(r) * float(np.linalg.norm(bases[sid])) for sid in base_ids}
            # The semantic sign is the frozen opposite-class-median direction.
            sign = {
                sid: float(
                    np.sign(
                        (
                            float(setpoints["q0_star"])
                            if int(label_map[sid]) == 1
                            else float(setpoints["q1_star"])
                        )
                        - coordinate_value(bases[sid], u)
                    )
                )
                for sid in base_ids
            }

            plan: list[dict[str, Any]] = [
                {"condition": "Y10_scalar", "index": -1,
                 "delta": {sid: sign[sid] * scale[sid] * u for sid in base_ids}},
                {"condition": "Y01_matched_context", "index": -1,
                 "delta": {sid: scale[sid] * unit(matched_context[sid]) for sid in base_ids}},
                {"condition": "Y11_both", "index": -1,
                 "delta": {
                     sid: sign[sid] * scale[sid] * u + scale[sid] * unit(matched_context[sid])
                     for sid in base_ids
                 }},
                {"condition": "same_label_context", "index": -1,
                 "delta": {sid: scale[sid] * unit(same_label_context[sid]) for sid in base_ids}},
                {"condition": "shuffled_context", "index": -1,
                 "delta": {sid: scale[sid] * unit(shuffled_context[sid]) for sid in base_ids}},
                {"condition": "full_patch_direction", "index": -1,
                 "delta": {sid: scale[sid] * unit(full_patch_direction[sid]) for sid in base_ids}},
            ]
            for i, direction in enumerate(random_dirs):
                plan.append({
                    "condition": "random", "index": i,
                    "delta": {sid: scale[sid] * direction for sid in base_ids},
                })
            for i, direction in enumerate(random_orth_dirs):
                plan.append({
                    "condition": "random_orthogonal", "index": i,
                    "delta": {sid: scale[sid] * direction for sid in base_ids},
                })

            for arm in plan:
                condition = str(arm["condition"])
                deltas = arm["delta"]
                results = run_intervention_batches(
                    adapter, samples_by_id, base_ids, layer=LAYER,
                    token_indices=token_indices, deltas_by_id=deltas,
                    output_token_ids=output_token_ids, capture_layers=[LAYER],
                    batch_size=int(cfg.runtime.batch_size),
                )
                for sid in base_ids:
                    base = bases[sid]
                    after = results[sid]["captured"][LAYER]
                    delta = np.asarray(deltas[sid], dtype=np.float64)
                    delta_norm = float(np.linalg.norm(delta))
                    activation_norm = float(np.linalg.norm(base))
                    achieved_r = delta_norm / max(activation_norm, 1e-12)
                    if condition != "Y11_both":
                        max_residual_error = max(
                            max_residual_error, abs(achieved_r - float(r))
                        )
                    margin_before = clean_margin[sid]
                    margin_after = _selected_margin(results[sid]["selected_logits"])
                    label = int(label_map[sid])
                    expected = 1 - label
                    q_before = coordinate_value(base, u)
                    q_after = coordinate_value(after, u)
                    rows.append({
                        "model": model_name,
                        "residual_fraction": float(r),
                        "condition": condition,
                        "direction_index": int(arm["index"]),
                        "base_sample_id": sid,
                        "pair_id": str(meta.loc[sid, "pair_id"]),
                        "relation": str(meta.loc[sid, "relation"]),
                        "target_label": label,
                        "expected_label": expected,
                        # 1. residual fraction
                        "delta_norm": delta_norm,
                        "activation_norm": activation_norm,
                        "achieved_residual_fraction": achieved_r,
                        # 2. relative to natural distance scales
                        "delta_over_within_class_distance": delta_norm
                        / max(scales["within_class_mean_distance"], 1e-12),
                        "delta_over_between_class_distance": delta_norm
                        / max(scales["between_class_mean_distance"], 1e-12),
                        # 3. achieved semantic displacement
                        "q_before": q_before,
                        "q_after": q_after,
                        "semantic_sign": sign[sid],
                        "delta_q": float(q_after - q_before),
                        "delta_q_z": float((q_after - q_before) / max(sigma_q, 1e-12)),
                        "delta_q_z_oriented": float(
                            sign[sid] * (q_after - q_before) / max(sigma_q, 1e-12)
                        ),
                        # 4. on-manifold diagnostics
                        "mahalanobis_diag_before": base_mahalanobis[sid],
                        "mahalanobis_diag_after": _diag_mahalanobis(after, cloud_mean, inv_var),
                        "knn_distance_before": base_knn[sid],
                        "knn_distance_after": _knn_distance(after, validation_states, KNN_K),
                        # 5. margin change, raw and standardized
                        "margin_before": margin_before,
                        "margin_after": margin_after,
                        "delta_margin_toward_expected": float(
                            margin_toward_label(margin_after, expected)
                            - margin_toward_label(margin_before, expected)
                        ),
                        "delta_margin_toward_expected_z": float(
                            (
                                margin_toward_label(margin_after, expected)
                                - margin_toward_label(margin_before, expected)
                            )
                            / max(sigma_m, 1e-12)
                        ),
                        "prediction_before": _prediction(margin_before),
                        "prediction_after": _prediction(margin_after),
                    })
                del results
            gc.collect()

        rows_df = pd.DataFrame(rows)
        if not np.isfinite(rows_df["delta_margin_toward_expected"].to_numpy(float)).all():
            raise RuntimeError("non-finite audit evidence produced")
        save_table(rows_df, run_dir / "audit_rows.parquet")

        # The residual fraction that E01's own operating point actually produced.
        # This is the number the review's confound argument turns on: the frozen
        # opposite-class-median setpoint is a different residual perturbation in
        # each checkpoint, so the headline Q0 contrast was never norm-matched.
        original_point = {
            sid: abs(
                (
                    float(setpoints["q0_star"])
                    if int(label_map[sid]) == 1
                    else float(setpoints["q1_star"])
                )
                - coordinate_value(bases[sid], u)
            )
            / max(float(np.linalg.norm(bases[sid])), 1e-12)
            for sid in base_ids
        }

        payload = {
            "audit_version": AUDIT_VERSION,
            "model": adapter.display_model_id,
            "original_e01_operating_point_residual_fraction": float(
                np.mean(list(original_point.values()))
            ),
            "site": f"{SITE}/L{LAYER}/{SELECTOR}",
            "n_base_examples": len(base_ids),
            "n_pairs": int(base_df["pair_id"].nunique()),
            "residual_fractions": list(RESIDUAL_FRACTIONS),
            "probe_metrics": probe_metrics,
            "validation_setpoints": {
                k: v for k, v in setpoints.items() if k != "grid"
            },
            "activation_scales": scales,
            "no_op_max_margin_deviation": no_op_deviation,
            "max_residual_fraction_error": max_residual_error,
            "confirmation_accessed": False,
        }
        save_json(payload, run_dir / "audit_model_summary.json")
        manifest.finish([{"model": model_name, "n_rows": len(rows_df)}])
        status.complete(f"calibration audit complete for {model_name}")
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


def run_e01_calibration_audit(models: tuple[str, ...] = MODELS) -> Path:
    """Run the audit for each frozen checkpoint, one model resident at a time."""
    logging.basicConfig(level=logging.INFO)
    for model_name in models:
        _audit_one_model(model_name)
    return analyze_e01_calibration_audit(models)


# ------------------------------------------------------------------- analysis
def _curve_for(
    rows: pd.DataFrame, *, value_col: str, n_bootstraps: int, confidence: float
) -> dict[str, Any]:
    """Q(r), A(r), G(r) with pair-cluster CIs at every residual fraction."""
    out: dict[str, Any] = {}
    for r, block in rows.groupby("residual_fraction", sort=True):
        wide = (
            block.groupby(["base_sample_id", "pair_id", "condition"], as_index=False)[value_col]
            .mean()
            .pivot(index=["base_sample_id", "pair_id"], columns="condition", values=value_col)
            .reset_index()
        )
        needed = {"Y10_scalar", "Y01_matched_context", "Y11_both"}
        if not needed.issubset(wide.columns):
            raise RuntimeError(f"missing factorial arms at r={r}")
        clusters = wide["pair_id"].astype(str).tolist()
        estimands = {
            "Q": wide["Y10_scalar"].to_numpy(float),
            "A": wide["Y01_matched_context"].to_numpy(float),
            "G": (
                wide["Y11_both"].to_numpy(float) - wide["Y10_scalar"].to_numpy(float)
            )
            - wide["Y01_matched_context"].to_numpy(float),
        }
        for name, values in estimands.items():
            ci = cluster_bootstrap_mean_ci(
                values, clusters, n_bootstraps=n_bootstraps,
                confidence_level=confidence, seed=BOOTSTRAP_SEED + int(1000 * float(r)),
            )
            out.setdefault(name, {})[f"{float(r):g}"] = {
                "mean": ci["mean"], "ci": [ci["ci_low"], ci["ci_high"]],
                "n_pairs": ci["n_clusters"],
                "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
            }
        for control in ("random", "random_orthogonal", "same_label_context",
                        "shuffled_context", "full_patch_direction"):
            if control not in wide.columns:
                continue
            ci = cluster_bootstrap_mean_ci(
                wide[control].to_numpy(float), clusters, n_bootstraps=n_bootstraps,
                confidence_level=confidence, seed=BOOTSTRAP_SEED + 7 + int(1000 * float(r)),
            )
            out.setdefault(control, {})[f"{float(r):g}"] = {
                "mean": ci["mean"], "ci": [ci["ci_low"], ci["ci_high"]],
            }
    return out


def _matched_semantic_shift_comparison(
    rows: pd.DataFrame,
    curves: dict[str, Any],
    models: tuple[str, ...],
    *,
    estimand: str = "Q",
    n_points: int = 5,
) -> dict[str, Any]:
    """Second ruler: compare checkpoints at matched achieved semantic shift.

    Section 8.2 of the review asks for the conclusion to be robust across
    reasonable normalizations, so the checkpoints are compared both at matched
    residual fraction and, here, at matched validation-standardized coordinate
    displacement. Each model's effect curve is linearly interpolated onto the
    overlapping ``delta_q_z`` range; no extrapolation is performed.
    """
    per_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in models:
        block = rows[
            (rows["model"] == model_name) & (rows["condition"] == "Y10_scalar")
        ]
        shift = (
            block.groupby("residual_fraction")["delta_q_z_oriented"].mean().sort_index()
        )
        xs = shift.to_numpy(float)
        ys = np.asarray(
            [curves[model_name][estimand][f"{float(r):g}"]["mean"] for r in shift.index],
            dtype=float,
        )
        if np.any(np.diff(xs) <= 0):
            raise RuntimeError(
                f"achieved semantic shift is not increasing in r for {model_name}"
            )
        per_model[model_name] = (xs, ys)

    low = max(float(per_model[m][0].min()) for m in models)
    high = min(float(per_model[m][0].max()) for m in models)
    if not high > low:
        return {
            "estimand": estimand,
            "overlap": None,
            "note": "the checkpoints share no common achieved delta_q_z range",
            "achieved_ranges": {
                m: [float(per_model[m][0].min()), float(per_model[m][0].max())]
                for m in models
            },
        }
    grid = np.linspace(low, high, int(n_points))
    points: dict[str, Any] = {}
    for value in grid:
        interpolated = {
            m: float(np.interp(value, per_model[m][0], per_model[m][1])) for m in models
        }
        record = dict(interpolated)
        if len(models) == 2:
            small, large = models
            record["difference_large_minus_small"] = float(
                interpolated[large] - interpolated[small]
            )
            record["ratio_large_over_small"] = (
                float(interpolated[large] / interpolated[small])
                if abs(interpolated[small]) > 1e-12
                else None
            )
        points[f"{float(value):.3f}"] = record
    ordering = None
    if len(models) == 2:
        ordering = all(
            v["difference_large_minus_small"] > 0.0 for v in points.values()
        )
    return {
        "estimand": estimand,
        "overlap": [low, high],
        "achieved_ranges": {
            m: [float(per_model[m][0].min()), float(per_model[m][0].max())]
            for m in models
        },
        "points": points,
        "ordering_preserved_across_overlap": ordering,
    }


# Frozen on-manifold acceptance band. An edit whose k-NN distance to the
# validation cloud grows by more than this factor is treated as divergent, and
# any effect measured there is reported as magnitude-artifact-suspect rather
# than as a model property (review section 8.3, final row).
MAX_ONMANIFOLD_KNN_RATIO = 1.35


def _manifold_diagnostics(rows: pd.DataFrame) -> dict[str, Any]:
    """Per model and residual fraction, how far the edited states drifted."""
    block = rows[rows["condition"] == "Y10_scalar"]
    out: dict[str, Any] = {}
    for (model_name, r), group in block.groupby(["model", "residual_fraction"], sort=True):
        knn_ratio = float(
            group["knn_distance_after"].mean() / max(group["knn_distance_before"].mean(), 1e-12)
        )
        out.setdefault(str(model_name), {})[f"{float(r):g}"] = {
            "knn_ratio": knn_ratio,
            "mahalanobis_diag_ratio": float(
                group["mahalanobis_diag_after"].mean()
                / max(group["mahalanobis_diag_before"].mean(), 1e-12)
            ),
            "delta_over_within_class_distance": float(
                group["delta_over_within_class_distance"].mean()
            ),
            "delta_over_between_class_distance": float(
                group["delta_over_between_class_distance"].mean()
            ),
            "on_manifold": bool(knn_ratio <= MAX_ONMANIFOLD_KNN_RATIO),
        }
    return out


def analyze_e01_calibration_audit(models: tuple[str, ...] = MODELS) -> Path:
    """Build the cross-checkpoint calibrated comparison and the verdict."""
    frames: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for model_name in models:
        model_dir = audit_dir() / model_name
        rows_path = model_dir / "audit_rows.parquet"
        if not rows_path.exists():
            raise RuntimeError(f"audit rows missing for {model_name}")
        frames.append(pd.read_parquet(rows_path))
        summaries[model_name] = json.loads(
            (model_dir / "audit_model_summary.json").read_text(encoding="utf-8")
        )
    rows = pd.concat(frames, ignore_index=True)
    save_table(rows, audit_dir() / "audit_rows_all_models.parquet")

    curves: dict[str, Any] = {}
    curves_z: dict[str, Any] = {}
    for model_name in models:
        block = rows[rows["model"] == model_name]
        curves[model_name] = _curve_for(
            block, value_col="delta_margin_toward_expected",
            n_bootstraps=2000, confidence=0.95,
        )
        curves_z[model_name] = _curve_for(
            block, value_col="delta_margin_toward_expected_z",
            n_bootstraps=2000, confidence=0.95,
        )

    # Achieved standardized semantic displacement per model and r, so the two
    # checkpoints can also be compared at matched delta_q_z rather than matched r.
    achieved = (
        rows[rows["condition"] == "Y10_scalar"]
        .groupby(["model", "residual_fraction"], as_index=False)["delta_q_z_oriented"]
        .mean()
    )

    matched_r: dict[str, Any] = {}
    if len(models) == 2:
        small, large = models
        for estimand in ("Q", "A", "G"):
            matched_r[estimand] = {}
            for r_key in curves[small][estimand]:
                a = curves[small][estimand][r_key]["mean"]
                b = curves[large][estimand][r_key]["mean"]
                matched_r[estimand][r_key] = {
                    small: a, large: b, "difference_large_minus_small": float(b - a),
                    "ratio_large_over_small": (
                        float(b / a) if abs(a) > 1e-12 else None
                    ),
                }

    ordering_preserved = None
    if matched_r:
        q_rows = matched_r["Q"]
        ordering_preserved = all(
            v["difference_large_minus_small"] > 0.0 for v in q_rows.values()
        )

    matched_shift = {
        estimand: _matched_semantic_shift_comparison(rows, curves, tuple(models), estimand=estimand)
        for estimand in ("Q", "A", "G")
    }

    manifold = _manifold_diagnostics(rows)
    on_manifold_fractions = sorted(
        {
            float(r)
            for per_model in manifold.values()
            for r, entry in per_model.items()
            if entry["on_manifold"]
        }
        .intersection(
            *[
                {float(r) for r, entry in per_model.items() if entry["on_manifold"]}
                for per_model in manifold.values()
            ]
        )
    )
    trustworthy = {
        estimand: {
            f"{r:g}": matched_r[estimand][f"{r:g}"]
            for r in on_manifold_fractions
            if f"{r:g}" in matched_r.get(estimand, {})
        }
        for estimand in ("Q", "A", "G")
        if matched_r
    }

    payload = {
        "audit_version": AUDIT_VERSION,
        "models": list(models),
        "residual_fractions": list(RESIDUAL_FRACTIONS),
        "manifold_diagnostics": manifold,
        "on_manifold_residual_fractions": on_manifold_fractions,
        "matched_r_comparison_on_manifold_only": trustworthy,
        "original_e01_operating_point_residual_fraction": {
            m: summaries[m].get("original_e01_operating_point_residual_fraction")
            for m in models
        },
        "matched_semantic_shift_comparison": matched_shift,
        "per_model_summary": summaries,
        "curves_raw_margin": curves,
        "curves_standardized_margin": curves_z,
        "achieved_delta_q_z_by_model_and_r": achieved.to_dict(orient="records"),
        "matched_residual_fraction_comparison": matched_r,
        "Q_ordering_preserved_at_every_matched_r": ordering_preserved,
        "confirmation_accessed": False,
        "interpretation_rule": (
            "If the checkpoint ordering of Q survives at every matched residual "
            "fraction, the E01 contrast is not an artifact of raw-unit operating "
            "points; report the calibrated curve rather than the single-point "
            "ratio. If it does not survive, withdraw the scale wording and keep "
            "the within-checkpoint distributed Q/A/G result."
        ),
    }
    out = audit_dir() / "E01_CALIBRATION_AUDIT.json"
    atomic_write_json(out, payload)
    return out
