import numpy as np
import pandas as pd
import pytest

from representation_reliability.interventions.setpoint import (
    norm_matched_direction_delta,
    setpoint_fidelity_tolerances,
    setpoint_identity_diagnostics,
    source_free_setpoint_delta,
)
from representation_reliability.interventions.truth_coordinate import (
    normalized_direction,
    random_unit_direction,
)
from representation_reliability.metrics.causal import margin_toward_label
from representation_reliability.metrics.setpoint import (
    GRID_QUANTILES,
    grid_example_metrics,
    summarize_grid_response,
    validation_setpoint_targets,
    validation_standardized_effect,
    within_base_centered_slope,
)


def test_source_free_setpoint_math_projection_and_orthogonal_preservation():
    base = np.array([1.0, -2.0, 3.0, 4.0])
    u = normalized_direction(np.array([1.0, 2.0, -1.0, 0.5]))
    target = 5.25
    delta = source_free_setpoint_delta(base, u, target)
    after = base + delta
    diagnostics = setpoint_identity_diagnostics(base, after, u, target)
    assert diagnostics["q_after"] == pytest.approx(target)
    assert diagnostics["projection_abs_deviation"] < 1e-12
    assert diagnostics["orthogonal_abs_deviation"] < 1e-12


def test_bfloat16_projection_gate_accounts_for_small_edit_rounding():
    bf16 = setpoint_fidelity_tolerances("bfloat16")
    assert 0.024704876291083977 < bf16["projection_relative"]
    assert bf16["orthogonal_relative"] == 0.02
    assert bf16["target_state_relative_l2"] == 0.02
    assert setpoint_fidelity_tolerances("float32")["projection_relative"] < 0.001


def test_validation_targets_use_exact_class_medians_and_fixed_quantiles():
    q = np.arange(20, dtype=float)
    labels = np.array([0] * 10 + [1] * 10)
    margins = np.linspace(-2.0, 2.0, 20)
    targets = validation_setpoint_targets(q, labels, margins)
    assert targets["q0_star"] == 4.5
    assert targets["q1_star"] == 14.5
    expected = np.quantile(q, [value for _name, value in GRID_QUANTILES])
    assert list(targets["grid"]) == [name for name, _value in GRID_QUANTILES]
    np.testing.assert_allclose(list(targets["grid"].values()), expected)


def test_probe_orientation_rejects_reversed_validation_medians():
    with pytest.raises(RuntimeError, match="probe orientation"):
        validation_setpoint_targets(
            [5.0, 6.0, -5.0, -6.0],
            [0, 0, 1, 1],
            [-1.0, -0.5, 0.5, 1.0],
        )


def test_validation_standardization_and_zero_variance_guards():
    result = validation_standardized_effect(
        -4.0,
        3.0,
        sigma_q_validation=2.0,
        sigma_margin_validation=1.5,
    )
    assert result == {"delta_q_z": -2.0, "delta_m_z": 2.0, "kappa_z": 1.0}
    with pytest.raises(ValueError, match="scale"):
        validation_standardized_effect(
            1.0, 1.0, sigma_q_validation=0.0, sigma_margin_validation=1.0
        )
    with pytest.raises(ValueError, match="variance"):
        validation_setpoint_targets([1.0] * 4, [0, 0, 1, 1], [0.0, 1.0, 2.0, 3.0])


def test_yes_and_no_target_orientation_have_the_same_causal_sign():
    assert margin_toward_label(3.0, 1) - margin_toward_label(1.0, 1) == 2.0
    assert margin_toward_label(-3.0, 0) - margin_toward_label(-1.0, 0) == 2.0


def test_random_and_orthogonal_controls_are_norm_matched_and_deterministic():
    semantic = np.array([3.0, 4.0, 0.0, 0.0])
    u = normalized_direction(np.array([1.0, 2.0, 3.0, 4.0]))
    random = random_unit_direction(4, 17)
    orthogonal = random_unit_direction(4, 23, orthogonal_to=u)
    random_delta = norm_matched_direction_delta(semantic, random)
    orthogonal_delta = norm_matched_direction_delta(semantic, orthogonal)
    assert np.linalg.norm(random_delta) == pytest.approx(np.linalg.norm(semantic))
    assert np.linalg.norm(orthogonal_delta) == pytest.approx(np.linalg.norm(semantic))
    assert abs(float(np.dot(orthogonal, u))) < 1e-12
    np.testing.assert_array_equal(random, random_unit_direction(4, 17))


def _grid_rows() -> pd.DataFrame:
    rows = []
    targets = dict(zip([name for name, _ in GRID_QUANTILES], [-2, -1, 0, 1, 2]))
    for pair_index in range(2):
        for side in ("a", "b"):
            base = f"p{pair_index}-{side}"
            for name, target in targets.items():
                rows.append(
                    {
                        "base_sample_id": base,
                        "pair_id": f"p{pair_index}",
                        "target_name": name,
                        "q_target": float(target),
                        "intervened_yes_no_margin": 0.5 + 2.0 * target,
                    }
                )
    return pd.DataFrame(rows)


def test_continuous_grid_ordering_spearman_monotonicity_and_slope():
    rows = _grid_rows()
    examples = grid_example_metrics(rows)
    assert np.allclose(examples["spearman"], 1.0)
    assert examples["spearman_ge_0_8"].all()
    assert examples["monotonic_nondecreasing"].all()
    assert within_base_centered_slope(rows) == pytest.approx(2.0)
    targets, examples_again, summary = summarize_grid_response(
        rows,
        n_bootstraps=50,
        confidence_level=0.95,
        seed=11,
    )
    assert len(targets) == 5 and len(examples_again) == 4
    assert summary["within_base_centered_slope"] == pytest.approx(2.0)
    assert summary["fraction_monotonic_nondecreasing"] == 1.0


def test_grid_rejects_missing_or_out_of_order_targets():
    rows = _grid_rows().iloc[:-1]
    with pytest.raises(RuntimeError, match="complete frozen grid"):
        grid_example_metrics(rows)


def test_flat_grid_response_is_finite_zero_association():
    rows = _grid_rows()
    rows["intervened_yes_no_margin"] = 1.25
    examples = grid_example_metrics(rows)
    assert np.isfinite(examples["spearman"]).all()
    assert (examples["spearman"] == 0.0).all()
    assert (examples["per_base_slope"] == 0.0).all()
