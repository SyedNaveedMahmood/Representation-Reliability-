import pandas as pd
import pytest

from representation_reliability.runners.e01b_support import (
    profile_limits,
    validate_artifact_shape,
    validation_identity,
)


def test_e01b_profile_limits_are_bounded_and_predeclared():
    assert profile_limits("smoke", None, 10, 10) == (25, 1, 1)
    assert profile_limits("pilot", None, 10, 10) == (75, 3, 3)
    assert profile_limits("full", None, 10, 10) == (None, 10, 10)
    assert profile_limits("smoke", 7, 4, 4) == (7, 1, 1)


def test_validation_identity_rejects_confirmation_and_is_deterministic():
    rows = pd.DataFrame(
        {"sample_id": ["b", "a"], "split": ["validation", "validation"]}
    )
    assert validation_identity(rows) == validation_identity(rows.iloc[::-1])
    bad = pd.concat(
        [rows, pd.DataFrame({"sample_id": ["secret"], "split": ["confirmation"]})],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="confirmation"):
        validation_identity(bad)


def test_e01b_artifact_schema_row_counts_and_source_free_contract():
    required = {
        "base_sample_id": ["a", "a"],
        "pair_id": ["p", "p"],
        "condition": ["no_op", "source_free_grid"],
        "target_name": ["q_base", "Q05"],
        "q_base": [1.0, 1.0],
        "q_target": [1.0, 0.0],
        "q_after": [1.0, 0.0],
        "base_yes_no_margin": [0.0, 0.0],
        "intervened_yes_no_margin": [0.0, -1.0],
        "activation_norm": [2.0, 2.0],
        "delta_norm": [0.0, 1.0],
        "delta_norm_ratio": [0.0, 0.5],
    }
    raw = pd.DataFrame(required)
    trace = pd.DataFrame({"trace_layer": [17, 20, 17, 20]})
    validate_artifact_shape(
        raw, trace, n_base_examples=1, n_treatment_specs=2, n_trace_layers=2
    )
    with pytest.raises(RuntimeError, match="donor/source"):
        validate_artifact_shape(
            raw.assign(source_sample_id="forbidden"),
            trace,
            n_base_examples=1,
            n_treatment_specs=2,
            n_trace_layers=2,
        )
