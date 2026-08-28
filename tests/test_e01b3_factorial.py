from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from representation_reliability.interventions.orthogonal_context import (
    fixed_setpoint_context_edit,
    standardize_orthogonal_context,
)
from representation_reliability.metrics.factorial import (
    aggregate_factorial_metrics,
    build_factorial_rows,
    build_factorial_trace_rows,
    factorial_estimands,
    paired_factorial_contrast,
    relation_family_factorial_metrics,
)
from representation_reliability.runners.e01b3_support import (
    probe_scaler_digest,
    validate_e01b3_artifact_shape,
    validate_source_plan_identity,
)


@pytest.mark.parametrize(
    ("arms", "expected_a", "expected_g"),
    [
        ((0.0, 2.0, 3.0, 5.0), 3.0, 0.0),  # pure additive
        ((0.0, 2.0, 0.0, 5.0), 0.0, 3.0),  # pure gating
        ((0.0, 2.0, 3.0, 7.0), 3.0, 2.0),  # mixed
        ((0.0, 2.0, 1.0, 2.0), 1.0, -1.0),  # suppression
    ],
)
def test_factorial_estimands_cover_predeclared_mechanisms(arms, expected_a, expected_g):
    result = factorial_estimands(*arms)
    assert result["Q0"] == pytest.approx(2.0)
    assert result["A_context"] == pytest.approx(expected_a)
    assert result["G_interaction"] == pytest.approx(expected_g)
    assert result["Q_context"] - result["Q0"] == pytest.approx(expected_g)
    assert (arms[3] - arms[1]) - (arms[2] - arms[0]) == pytest.approx(expected_g)


def test_context_only_preserves_q_and_combined_arm_hits_target():
    u = np.array([1.0, 0.0, 0.0])
    base = np.array([0.25, 2.0, -1.0])
    context, diagnostics = standardize_orthogonal_context(
        np.array([7.0, 3.0, 4.0]), u, 2.5, epsilon=1e-12
    )
    semantic, scaled_context, combined = fixed_setpoint_context_edit(base, u, 1.75, context, 0.5)
    context_only = base + scaled_context
    y11 = base + combined
    assert diagnostics["context_applied_norm"] == pytest.approx(2.5)
    assert np.dot(context, u) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(context_only, u) == pytest.approx(np.dot(base, u), abs=1e-12)
    assert np.dot(y11, u) == pytest.approx(1.75, abs=1e-12)
    assert np.allclose(combined, semantic + scaled_context)


def _arm_row(
    *,
    sid: str,
    pair: str,
    condition: str,
    strength: float,
    seed: float,
    before: float,
    after: float,
    q_after: float,
) -> dict:
    return {
        "model_id": "model",
        "resolved_revision": "revision",
        "base_sample_id": sid,
        "pair_id": pair,
        "relation_family": "north_south",
        "target_label": 1 if sid.endswith("1") else 0,
        "q_base": 0.2,
        "q_target": 1.2,
        "q_after": q_after,
        "condition": condition,
        "lambda_context": strength,
        "direction_seed": seed,
        "context_source_id": None if condition == "random_orthogonal" else "source",
        "context_selection_seed": 17,
        "context_applied_norm": 3.0,
        "token_id": 99,
        "token_index": 4,
        "margin_toward_target_before": before,
        "margin_toward_target_after": after,
    }


def _factorial_inputs():
    prior = []
    y01 = []
    for index, sid in enumerate(("s0", "s1")):
        pair = f"p{index}"
        prior.append(
            _arm_row(
                sid=sid,
                pair=pair,
                condition="coordinate_only",
                strength=0.0,
                seed=np.nan,
                before=1.0,
                after=3.0,
                q_after=1.2,
            )
        )
        for condition, seed in (
            ("matched_orthogonal", np.nan),
            ("random_orthogonal", 40001.0),
        ):
            y01_value = 2.0 + index
            y11_value = 5.0 + index
            prior.append(
                _arm_row(
                    sid=sid,
                    pair=pair,
                    condition=condition,
                    strength=1.0,
                    seed=seed,
                    before=1.0,
                    after=y11_value,
                    q_after=1.2,
                )
            )
            y01.append(
                _arm_row(
                    sid=sid,
                    pair=pair,
                    condition=condition,
                    strength=1.0,
                    seed=seed,
                    before=1.0,
                    after=y01_value,
                    q_after=0.2,
                )
            )
    return pd.DataFrame(prior), pd.DataFrame(y01)


def test_factorial_merge_orientation_identity_and_metrics():
    prior, y01 = _factorial_inputs()
    rows, audit = build_factorial_rows(prior, y01, y11_run_id="prior", y01_run_id="current")
    assert audit["compatibility_mismatches"] == 0
    assert set(rows["A_context"]) == {1.0, 2.0}
    assert set(rows["Q0"]) == {2.0}
    assert set(rows["G_interaction"]) == {1.0}
    additive, interaction = aggregate_factorial_metrics(
        rows, n_bootstraps=100, confidence_level=0.95, seed=7
    )
    assert len(additive) == len(interaction) == 2
    contrast = paired_factorial_contrast(
        rows,
        metric="G_interaction",
        left_condition="matched_orthogonal",
        right_condition="random_orthogonal",
        context_strength=1.0,
        n_bootstraps=100,
        confidence_level=0.95,
        seed=7,
    )
    assert contrast["mean_difference"] == pytest.approx(0.0)
    family = relation_family_factorial_metrics(
        rows, n_bootstraps=100, confidence_level=0.95, seed=7
    )
    assert set(family["relation_family"]) == {"north_south"}


@pytest.mark.parametrize(
    "column",
    [
        "context_source_id",
        "context_selection_seed",
        "q_target",
        "context_applied_norm",
        "token_id",
    ],
)
def test_factorial_merge_rejects_prior_arm_identity_mismatch(column):
    prior, y01 = _factorial_inputs()
    index = y01.index[0]
    value = y01.loc[index, column]
    y01.loc[index, column] = value + 1 if isinstance(value, (int, float, np.number)) else "wrong"
    with pytest.raises(RuntimeError, match="compatibility mismatch"):
        build_factorial_rows(prior, y01, y11_run_id="prior", y01_run_id="current")


def _trace_row(sid, pair, condition, seed, layer, clean_q, after_q, clean_m, after_m):
    return {
        "model_id": "model",
        "base_sample_id": sid,
        "pair_id": pair,
        "relation_family": "north_south",
        "condition": condition,
        "lambda_context": 0.0 if condition == "coordinate_only" else 1.0,
        "direction_seed": seed,
        "trace_layer": layer,
        "clean_truth_coordinate": clean_q,
        "intervened_truth_coordinate": after_q,
        "clean_native_yes_no_margin": clean_m,
        "intervened_native_yes_no_margin": after_m,
    }


def test_trace_factorial_decomposition_and_target_orientation():
    prior = pd.DataFrame(
        [
            _trace_row("s0", "p0", "coordinate_only", np.nan, 17, 0, 2, 0, 2),
            _trace_row("s0", "p0", "matched_orthogonal", np.nan, 17, 0, 2, 0, 5),
            _trace_row("s1", "p1", "coordinate_only", np.nan, 17, 0, 2, 0, 2),
            _trace_row("s1", "p1", "matched_orthogonal", np.nan, 17, 0, 2, 0, 5),
        ]
    )
    y01 = pd.DataFrame(
        [
            _trace_row("s0", "p0", "matched_orthogonal", np.nan, 17, 0, 0, 0, 1),
            _trace_row("s1", "p1", "matched_orthogonal", np.nan, 17, 0, 0, 0, 1),
        ]
    )
    labels = pd.DataFrame({"base_sample_id": ["s0", "s1"], "target_label": [0, 1]})
    result = build_factorial_trace_rows(
        prior,
        y01,
        labels,
        layer_references={"17": {"sigma_q_validation": 2.0, "sigma_margin_validation": 4.0}},
    ).set_index("base_sample_id")
    assert result.loc["s0", "A_q_z"] == pytest.approx(0.0)
    assert result.loc["s0", "G_q_z"] == pytest.approx(0.0)
    assert result.loc["s0", "A_margin_z"] == pytest.approx(-0.25)
    assert result.loc["s1", "A_margin_z"] == pytest.approx(0.25)
    assert result.loc["s0", "G_margin_z"] == pytest.approx(-0.5)
    assert result.loc["s1", "G_margin_z"] == pytest.approx(0.5)


def _plan_frame():
    return pd.DataFrame(
        [
            {
                "base_sample_id": "s0",
                "condition": "matched_orthogonal",
                "direction_seed": np.nan,
                "base_pair_id": "p0",
                "base_relation_family": "north_south",
                "base_label": 0,
                "context_source_id": "s1",
                "context_source_pair_id": "p0",
                "context_source_relation_family": "north_south",
                "context_source_label": 1,
                "context_selection_seed": 3,
                "reference_norm_source": "matched_twin",
                "reference_fallback_used": False,
                "context_vector_fallback_used": False,
                "reference_norm": 2.0,
                "matched_raw_norm": 2.0,
                "context_raw_norm": 2.1,
                "context_projected_raw_norm": 2.0,
                "context_applied_norm": 2.0,
                "context_dot_truth_direction": 0.0,
                "context_norm_relative_error": 0.0,
            }
        ]
    )


def test_source_plan_seed_lambda_norm_identity_and_shape_contract():
    plan = _plan_frame()
    audit = validate_source_plan_identity(plan, plan.copy())
    assert audit["source_plan_mismatch_count"] == 0
    changed = copy.deepcopy(plan)
    changed.loc[0, "context_applied_norm"] = 2.2
    with pytest.raises(RuntimeError, match="source plan mismatch"):
        validate_source_plan_identity(plan, changed)

    prior, y01 = _factorial_inputs()
    factorial, _ = build_factorial_rows(prior, y01, y11_run_id="prior", y01_run_id="current")
    context_trace = pd.DataFrame(index=range(16))
    factorial_trace = pd.DataFrame(index=range(16))
    validate_e01b3_artifact_shape(
        y01,
        context_trace,
        factorial,
        factorial_trace,
        n_base_examples=2,
        n_context_specs=2,
        n_trace_layers=4,
    )


class _Classifier:
    coef_ = np.array([[1.0, 2.0]])
    intercept_ = np.array([0.25])


def test_probe_scaler_digest_is_deterministic_and_sensitive():
    fit = {
        "classifier": _Classifier(),
        "scaler_mean": np.array([0.1, 0.2]),
        "scaler_scale": np.array([1.0, 2.0]),
        "chosen_C": 0.1,
    }
    first = probe_scaler_digest({17: fit})
    assert first == probe_scaler_digest({17: fit})
    changed = dict(fit)
    changed["chosen_C"] = 1.0
    assert first != probe_scaler_digest({17: changed})
