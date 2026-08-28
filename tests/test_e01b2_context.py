from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from representation_reliability.data.base import samples_to_dataframe
from representation_reliability.data.synthetic import generate_synthetic_relations
from representation_reliability.interventions.orthogonal_context import (
    fixed_setpoint_context_edit,
    orthogonal_component,
    resolve_context_reference_norm,
    standardize_orthogonal_context,
)
from representation_reliability.interventions.setpoint import source_free_setpoint_delta
from representation_reliability.interventions.truth_coordinate import (
    coordinate_value,
    normalized_direction,
    random_unit_direction,
)
from representation_reliability.metrics.orthogonal_context import (
    aggregate_context_rows,
    attach_context_increments,
    attach_trace_context_increments,
    paired_context_contrast,
)
from representation_reliability.runners.e01b2 import _load_shard, _save_shard
from representation_reliability.runners.e01b2_support import (
    build_context_source_plans,
    e01b2_profile_limits,
    parse_context_strengths,
    validate_context_source_plans,
    validate_e01b2_artifact_shape,
    validation_matched_norm_fallback,
)


def test_orthogonal_decomposition_and_fixed_setpoint_identity():
    base = np.array([1.0, -2.0, 3.0, 0.5])
    source = np.array([-1.0, 4.0, 2.0, 3.0])
    u = normalized_direction(np.array([1.0, 2.0, -1.0, 0.5]))
    raw = orthogonal_component(source, base, u)
    assert abs(float(np.dot(raw, u))) < 1e-12
    context, diagnostics = standardize_orthogonal_context(
        raw, u, 2.75, epsilon=1e-12
    )
    assert np.linalg.norm(context) == pytest.approx(2.75)
    assert abs(float(np.dot(context, u))) < 1e-12
    assert diagnostics["context_norm_relative_error"] < 1e-12
    semantic, scaled_context, total = fixed_setpoint_context_edit(
        base, u, 5.0, context, 0.5
    )
    np.testing.assert_allclose(total - semantic, scaled_context)
    assert coordinate_value(base + total, u) == pytest.approx(5.0)


def test_coordinate_only_is_exact_e01b1_setpoint_edit():
    base = np.array([0.25, -0.5, 1.25])
    u = normalized_direction(np.array([2.0, 1.0, -1.0]))
    expected = source_free_setpoint_delta(base, u, -1.75)
    semantic, context, total = fixed_setpoint_context_edit(
        base, u, -1.75, np.zeros_like(base), 0.0
    )
    np.testing.assert_array_equal(semantic, expected)
    np.testing.assert_array_equal(context, np.zeros_like(base))
    np.testing.assert_array_equal(total, expected)


def test_degenerate_reference_and_vector_fallback_are_explicit():
    norm, source, used = resolve_context_reference_norm(
        0.0, 3.5, epsilon=1e-6
    )
    assert (norm, source, used) == (
        3.5,
        "validation_median_nondegenerate_matched",
        True,
    )
    u = normalized_direction(np.array([1.0, 0.0, 0.0]))
    fallback = random_unit_direction(3, 17, orthogonal_to=u)
    context, diagnostics = standardize_orthogonal_context(
        np.zeros(3),
        u,
        norm,
        epsilon=1e-6,
        fallback_direction=fallback,
    )
    assert diagnostics["context_vector_fallback_used"] is True
    assert np.linalg.norm(context) == pytest.approx(3.5)
    assert abs(float(np.dot(context, u))) < 1e-12


def _source_fixture():
    samples = generate_synthetic_relations(100, seed=31)
    frame = samples_to_dataframe(samples)
    frame["split"] = "discovery_test"
    by_id = {sample.sample_id: sample for sample in samples}
    base_ids = frame["sample_id"].astype(str).tolist()[:10]
    return samples, frame, by_id, base_ids


def test_context_source_selection_invariants_and_determinism():
    _samples, frame, by_id, base_ids = _source_fixture()
    plans = build_context_source_plans(
        frame, by_id, base_sample_ids=base_ids, seed=41
    )
    assert plans == build_context_source_plans(
        frame, by_id, base_sample_ids=base_ids, seed=41
    )
    validate_context_source_plans(frame, plans)
    rows = frame.set_index("sample_id")
    for sid, plan in plans.items():
        base = rows.loc[sid]
        same_family = rows.loc[plan.same_family_source_id]
        different = rows.loc[plan.different_family_source_id]
        same_label = rows.loc[plan.same_label_source_id]
        assert same_family["relation"] == base["relation"]
        assert same_family["pair_id"] != base["pair_id"]
        assert int(same_family["target_label"]) != int(base["target_label"])
        assert different["relation"] != base["relation"]
        assert different["pair_id"] != base["pair_id"]
        assert int(same_label["target_label"]) == int(base["target_label"])
        assert same_label["pair_id"] != base["pair_id"]


def test_context_source_plan_rejects_corrupted_matched_twin():
    _samples, frame, by_id, base_ids = _source_fixture()
    base = by_id[base_ids[0]]
    matched_id = str(base.counterfactual_id)
    by_id[matched_id] = replace(
        by_id[matched_id],
        metadata={**by_id[matched_id].metadata, "entity_a": "corrupted"},
    )
    with pytest.raises(RuntimeError, match="counterfactual nuisance mismatch"):
        build_context_source_plans(
            frame, by_id, base_sample_ids=[base.sample_id], seed=2
        )


def test_validation_fallback_uses_validation_only_matched_norms():
    samples = generate_synthetic_relations(20, seed=8)
    frame = samples_to_dataframe(samples)
    frame["split"] = "validation"
    by_id = {sample.sample_id: sample for sample in samples}
    activations = {
        sample.sample_id: np.array([index, index % 3, 1.0], dtype=float)
        for index, sample in enumerate(samples)
    }
    result = validation_matched_norm_fallback(
        frame,
        by_id,
        activations,
        normalized_direction(np.array([0.0, 0.0, 1.0])),
    )
    assert result["n_validation"] == 20
    assert result["n_nondegenerate"] == 20
    assert result["median_nondegenerate_matched_orthogonal_norm"] > 0.0
    assert result["confirmation_accessed"] is False


def _metric_rows() -> pd.DataFrame:
    output = []
    effects = {
        "coordinate_only": 1.0,
        "matched_orthogonal": 1.5,
        "same_family_shuffled_orthogonal": 1.4,
        "different_family_shuffled_orthogonal": 1.1,
        "same_label_orthogonal": 1.2,
        "random_orthogonal": 1.05,
    }
    for pair in range(3):
        for side in ("a", "b"):
            sid = f"p{pair}-{side}"
            output.append(
                {
                    "base_sample_id": sid,
                    "pair_id": f"p{pair}",
                    "condition": "coordinate_only",
                    "lambda_context": 0.0,
                    "direction_seed": np.nan,
                    "delta_margin_toward_target": effects["coordinate_only"],
                }
            )
            for strength in (0.5, 1.0):
                for condition, effect in effects.items():
                    if condition == "coordinate_only":
                        continue
                    seeds = (11, 12) if condition == "random_orthogonal" else (None,)
                    for seed in seeds:
                        output.append(
                            {
                                "base_sample_id": sid,
                                "pair_id": f"p{pair}",
                                "condition": condition,
                                "lambda_context": strength,
                                "direction_seed": seed,
                                "delta_margin_toward_target": 1.0
                                + strength * (effect - 1.0),
                            }
                        )
    return pd.DataFrame(output)


def test_context_increment_aggregation_and_pair_cluster_contrast():
    rows = attach_context_increments(_metric_rows())
    matched = rows[
        (rows["condition"] == "matched_orthogonal")
        & np.isclose(rows["lambda_context"], 1.0)
    ]
    assert np.allclose(matched["context_increment_vs_coordinate_only"], 0.5)
    aggregate = aggregate_context_rows(
        rows,
        context_strengths=(0.5, 1.0),
        n_bootstraps=30,
        confidence_level=0.95,
        seed=4,
    )
    row = aggregate[
        (aggregate["condition"] == "matched_orthogonal")
        & np.isclose(aggregate["comparison_lambda"], 1.0)
    ].iloc[0]
    assert row["mean_context_increment"] == pytest.approx(0.5)
    contrast = paired_context_contrast(
        rows,
        left_condition="matched_orthogonal",
        right_condition="random_orthogonal",
        context_strength=1.0,
        n_bootstraps=30,
        confidence_level=0.95,
        seed=5,
    )
    assert contrast["mean_difference"] == pytest.approx(0.45)


def test_trace_context_increment_schema_and_identity():
    rows = []
    for sid in ("a", "b"):
        for layer in (17, 20):
            rows.extend(
                [
                    {
                        "base_sample_id": sid,
                        "pair_id": "p",
                        "trace_layer": layer,
                        "condition": "coordinate_only",
                        "lambda_context": 0.0,
                        "oriented_delta_q_z": 1.0,
                        "oriented_delta_native_margin_z": 0.2,
                    },
                    {
                        "base_sample_id": sid,
                        "pair_id": "p",
                        "trace_layer": layer,
                        "condition": "matched_orthogonal",
                        "lambda_context": 1.0,
                        "oriented_delta_q_z": 1.0,
                        "oriented_delta_native_margin_z": 0.5,
                    },
                ]
            )
    attached = attach_trace_context_increments(pd.DataFrame(rows))
    matched = attached[attached["condition"] == "matched_orthogonal"]
    assert np.allclose(matched["context_increment_delta_q_z"], 0.0)
    assert np.allclose(matched["context_increment_delta_m_z"], 0.3)


def test_e01b2_profiles_and_frozen_strengths():
    assert e01b2_profile_limits("smoke", None, 10) == (25, 1)
    assert e01b2_profile_limits("pilot", None, 10) == (75, 3)
    assert e01b2_profile_limits("full", None, 10) == (None, 10)
    assert parse_context_strengths("1.0,0.5") == (0.5, 1.0)
    with pytest.raises(ValueError, match="frozen"):
        parse_context_strengths("0.5,1.0,2.0")


def test_artifact_shape_contract():
    raw = attach_context_increments(_metric_rows())
    for column in (
        "q_base",
        "q_target",
        "q_after",
        "context_reference_norm",
        "context_applied_norm",
        "context_dot_truth_direction",
        "semantic_delta_norm",
        "context_delta_norm",
        "total_delta_norm",
    ):
        raw[column] = 0.0
    trace = pd.DataFrame(index=range(len(raw) * 4))
    validate_e01b2_artifact_shape(
        raw,
        trace,
        n_base_examples=6,
        n_specs=13,
        n_trace_layers=4,
    )
    with pytest.raises(RuntimeError, match="row-count"):
        validate_e01b2_artifact_shape(
            raw.iloc[:-1],
            trace,
            n_base_examples=6,
            n_specs=13,
            n_trace_layers=4,
        )


def test_shard_resume_identity_rejects_changed_plan(tmp_path):
    spec = {
        "key": "matched_orthogonal_lambda_1",
        "condition": "matched_orthogonal",
        "lambda_context": 1.0,
        "direction_seed": None,
    }
    raw = pd.DataFrame({"base_sample_id": ["a", "b"]})
    trace = pd.DataFrame(
        {"base_sample_id": ["a", "a", "b", "b"], "trace_layer": [17, 20, 17, 20]}
    )
    _save_shard(
        tmp_path,
        spec,
        base_ids=["a", "b"],
        plan_digest="plan-a",
        raw=raw,
        trace=trace,
    )
    loaded = _load_shard(
        tmp_path,
        spec,
        base_ids=["a", "b"],
        plan_digest="plan-a",
        n_traces=2,
    )
    assert loaded is not None
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _load_shard(
            tmp_path,
            spec,
            base_ids=["a", "b"],
            plan_digest="plan-b",
            n_traces=2,
        )
