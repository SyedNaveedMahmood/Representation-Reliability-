import numpy as np
import pandas as pd
import pytest

from representation_reliability.analysis.e01a_trace_mechanism import (
    _cluster_bootstrap_regression_coefficients,
    cluster_bootstrap_draws,
    conversion_slope,
    deduplicate_clean_traces,
    merge_matched_shuffled,
    orient_trace_changes,
    propagation_ratio,
    source_regression_design,
    standardize_changes,
)


def _identity_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base_sample_id": ["yes", "no"],
            "pair_id": ["p1", "p2"],
            "expected_label": [1, 0],
            "relation_family": ["east_west", "north_south"],
            "gold_label": [0, 1],
            "base_prediction": [0, 1],
        }
    )


def test_clean_trace_deduplication_checks_repeated_values():
    trace = pd.DataFrame(
        {
            "base_sample_id": ["a", "a", "b", "b"],
            "pair_id": ["p1", "p1", "p2", "p2"],
            "trace_layer": [17, 17, 17, 17],
            "clean_truth_coordinate": [1.0, 1.0, 2.0, 2.0],
            "clean_native_yes_no_margin": [0.5, 0.5, -0.5, -0.5],
        }
    )
    clean, audit = deduplicate_clean_traces(trace)
    assert len(clean) == 2
    assert audit["max_clean_truth_coordinate_disagreement"] == 0.0
    inconsistent = trace.copy()
    inconsistent.loc[1, "clean_truth_coordinate"] = 1.1
    with pytest.raises(RuntimeError, match="baselines disagree"):
        deduplicate_clean_traces(inconsistent)


def test_expected_label_orientation_handles_yes_and_no_targets():
    trace = pd.DataFrame(
        {
            "base_sample_id": ["yes", "no"],
            "pair_id": ["p1", "p2"],
            "condition": ["truth_coordinate"] * 2,
            "alpha": [1.0, 1.0],
            "direction_seed": [np.nan, np.nan],
            "trace_layer": [17, 17],
            "delta_truth_coordinate": [2.0, -3.0],
            "delta_native_yes_no_margin": [0.4, -0.7],
        }
    )
    oriented = orient_trace_changes(trace, _identity_rows())
    assert oriented["oriented_delta_truth_coordinate"].tolist() == [2.0, 3.0]
    assert oriented["oriented_delta_native_margin"].tolist() == [0.4, 0.7]


def test_pair_cluster_bootstrap_keeps_twin_rows_together():
    frame = pd.DataFrame(
        {"pair_id": ["p1", "p1", "p2", "p2"], "value": [0.0, 2.0, 10.0, 12.0]}
    )
    draws = cluster_bootstrap_draws(
        frame,
        lambda sample: float(sample["value"].mean()),
        n_bootstraps=100,
        seed=7,
    )
    assert set(np.unique(draws)) <= {1.0, 6.0, 11.0}
    assert np.array_equal(
        draws,
        cluster_bootstrap_draws(
            frame,
            lambda sample: float(sample["value"].mean()),
            n_bootstraps=100,
            seed=7,
        ),
    )


def test_standardization_guards_zero_variance():
    assert np.allclose(standardize_changes([2.0, -4.0], 2.0), [1.0, -2.0])
    with pytest.raises(ValueError, match="standard deviation"):
        standardize_changes([1.0], 0.0)


def test_propagation_ratio_guards_small_denominator():
    assert propagation_ratio(0.5, 1.0) == 0.5
    with pytest.raises(ValueError, match="too small"):
        propagation_ratio(1.0, 0.0)


def test_conversion_slope_recovers_linear_mapping_and_sensitivity():
    result = conversion_slope([-2.0, -1.0, 1.0, 2.0], [-6.0, -3.0, 3.0, 6.0])
    assert result["beta"] == pytest.approx(3.0)
    assert result["r2_uncentered"] == pytest.approx(1.0)
    assert result["intercept"] == pytest.approx(0.0)
    assert result["beta_with_intercept"] == pytest.approx(3.0)


def test_matched_shuffled_merge_requires_exact_identity():
    base = {
        "base_sample_id": "a",
        "pair_id": "p1",
        "alpha": 1.0,
        "base_truth_coordinate": 1.0,
        "delta_margin_toward_expected": 0.2,
        "expected_label": 1,
    }
    rows = pd.DataFrame(
        [
            {
                **base,
                "condition": "truth_coordinate",
                "source_truth_coordinate": 3.0,
                "delta_truth_coordinate": 2.0,
            },
            {
                **base,
                "condition": "shuffled_coordinate",
                "source_truth_coordinate": 2.5,
                "delta_truth_coordinate": 1.5,
            },
        ]
    )
    merged = merge_matched_shuffled(rows)
    assert len(merged) == 1
    assert merged.iloc[0]["matched_oriented_delta_q"] == 2.0
    with pytest.raises(RuntimeError, match="one row"):
        merge_matched_shuffled(pd.concat([rows, rows.iloc[[0]]], ignore_index=True))


def test_source_indicator_regression_setup_orients_and_labels_columns():
    rows = pd.DataFrame(
        {
            "condition": ["truth_coordinate", "shuffled_coordinate"] * 2,
            "alpha": [1.0, 1.0, -1.0, -1.0],
            "expected_label": [1, 0, 1, 0],
            "delta_truth_coordinate": [2.0, -3.0, -2.0, 3.0],
            "delta_margin_toward_expected": [1.0, 1.5, -1.0, -1.5],
        }
    )
    design, outcome, names = source_regression_design(rows, include_alpha=True)
    assert names == ["intercept", "beta_coordinate", "beta_matched_indicator", "beta_alpha"]
    assert design[:, 1].tolist() == [2.0, 3.0, -2.0, -3.0]
    assert design[:, 2].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert outcome.tolist() == [1.0, 1.5, -1.0, -1.5]


def test_source_regression_cluster_bootstrap_returns_all_coefficients():
    rows = pd.DataFrame(
        {
            "pair_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "condition": ["truth_coordinate", "shuffled_coordinate"] * 4,
            "alpha": [1.0, 1.0, -1.0, -1.0] * 2,
            "expected_label": [1] * 4 + [0] * 4,
            "delta_truth_coordinate": [2.0, 3.0, -2.0, -3.0, -2.5, -3.5, 2.5, 3.5],
            "delta_margin_toward_expected": [1.0, 1.5, -1.0, -1.5] * 2,
        }
    )
    draws, names = _cluster_bootstrap_regression_coefficients(
        rows, include_alpha=False, n_bootstraps=25, seed=3
    )
    assert draws.shape == (25, 3)
    assert names == ["intercept", "beta_coordinate", "beta_matched_indicator"]
    assert np.isfinite(draws).all()
