"""Unit contracts for the E15 Gate 1 addendum and the E01 calibration audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from representation_reliability.runners.e01_calibration_audit import (
    RESIDUAL_FRACTIONS,
    _curve_for,
    _diag_mahalanobis,
    _knn_distance,
    _pairwise_distance_scales,
)
from representation_reliability.runners.e15_gate1 import (
    FULL_PATCH_RANDOM_SEED_BASE,
    GATE_HORIZON,
    _twin_carrier_states,
)
from representation_reliability.runners.e15_support import (
    DIRECTION_SEED_BASE,
    K0,
    N_ORTHOGONAL_DIRECTIONS,
    N_RANDOM_DIRECTIONS,
)


# --------------------------------------------------------------------- Gate 1
def test_twin_carrier_map_pairs_each_sample_with_its_counterfactual():
    view = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "pair_id": ["p0", "p0", "p1", "p1"],
        }
    )
    clean = {
        sid: {"sites": {"carrier": np.full(3, float(i))}}
        for i, sid in enumerate(["s0", "s1", "s2", "s3"])
    }
    twins = _twin_carrier_states(clean, view)
    np.testing.assert_allclose(twins["s0"], np.full(3, 1.0))
    np.testing.assert_allclose(twins["s1"], np.full(3, 0.0))
    np.testing.assert_allclose(twins["s2"], np.full(3, 3.0))
    np.testing.assert_allclose(twins["s3"], np.full(3, 2.0))


def test_twin_carrier_map_rejects_an_incomplete_pair():
    view = pd.DataFrame({"sample_id": ["s0", "s1", "s2"], "pair_id": ["p0", "p0", "p1"]})
    clean = {sid: {"sites": {"carrier": np.zeros(3)}} for sid in ["s0", "s1", "s2"]}
    with pytest.raises(RuntimeError, match="not complete"):
        _twin_carrier_states(clean, view)


def test_twin_carrier_map_returns_copies_not_aliases():
    view = pd.DataFrame({"sample_id": ["s0", "s1"], "pair_id": ["p0", "p0"]})
    original = np.ones(3)
    clean = {"s0": {"sites": {"carrier": np.zeros(3)}}, "s1": {"sites": {"carrier": original}}}
    twins = _twin_carrier_states(clean, view)
    twins["s0"][0] = 99.0
    assert original[0] == 1.0


def test_gate1_random_seeds_do_not_collide_with_stage3_control_seeds():
    stage3 = set(range(DIRECTION_SEED_BASE, DIRECTION_SEED_BASE + N_RANDOM_DIRECTIONS))
    stage3 |= set(
        range(DIRECTION_SEED_BASE + 100, DIRECTION_SEED_BASE + 100 + N_ORTHOGONAL_DIRECTIONS)
    )
    gate1 = set(range(FULL_PATCH_RANDOM_SEED_BASE, FULL_PATCH_RANDOM_SEED_BASE + 5))
    assert not (stage3 & gate1)


def test_gate1_horizon_is_the_frozen_k0():
    assert GATE_HORIZON == K0 == 1


# ----------------------------------------------------------- calibration audit
def test_residual_fraction_grid_is_frozen_and_increasing():
    assert list(RESIDUAL_FRACTIONS) == sorted(RESIDUAL_FRACTIONS)
    assert len(set(RESIDUAL_FRACTIONS)) == len(RESIDUAL_FRACTIONS)
    assert all(0.0 < r <= 1.0 for r in RESIDUAL_FRACTIONS)


def test_pairwise_distance_scales_separates_within_and_between_class():
    rng = np.random.default_rng(0)
    states = np.vstack(
        [rng.normal(0, 0.1, size=(60, 8)), rng.normal(5, 0.1, size=(60, 8))]
    )
    labels = np.array([0] * 60 + [1] * 60)
    scales = _pairwise_distance_scales(states, labels, seed=1, n_pairs=2000)
    assert scales["between_class_mean_distance"] > scales["within_class_mean_distance"]
    assert scales["activation_norm_mean"] > 0


def test_diag_mahalanobis_grows_with_standardized_displacement():
    mean = np.zeros(4)
    inv_var = np.ones(4)
    near = _diag_mahalanobis(np.array([1.0, 0, 0, 0]), mean, inv_var)
    far = _diag_mahalanobis(np.array([3.0, 0, 0, 0]), mean, inv_var)
    assert far > near
    assert near == pytest.approx(1.0)


def test_knn_distance_is_zero_inside_the_cloud_and_positive_outside():
    cloud = np.zeros((20, 3))
    assert _knn_distance(np.zeros(3), cloud, 5) == pytest.approx(0.0)
    assert _knn_distance(np.array([0.0, 0.0, 2.0]), cloud, 5) == pytest.approx(2.0)


def _factorial_rows() -> pd.DataFrame:
    """Y00=0 by construction; Q=2, A=1, G=0.5 at every residual fraction."""
    records = []
    for r in (0.05, 0.10):
        for pair in range(12):
            for twin in range(2):
                sid = f"p{pair}-{twin}"
                for condition, value in (
                    ("Y10_scalar", 2.0),
                    ("Y01_matched_context", 1.0),
                    ("Y11_both", 3.5),
                    ("random", 0.0),
                    ("full_patch_direction", 4.0),
                ):
                    records.append(
                        {
                            "residual_fraction": r,
                            "condition": condition,
                            "base_sample_id": sid,
                            "pair_id": f"p{pair}",
                            "delta_margin_toward_expected": value,
                        }
                    )
    return pd.DataFrame(records)


def test_curve_for_recovers_the_factorial_estimands_at_every_residual_fraction():
    curves = _curve_for(
        _factorial_rows(), value_col="delta_margin_toward_expected",
        n_bootstraps=200, confidence=0.95,
    )
    for r_key in ("0.05", "0.1"):
        assert curves["Q"][r_key]["mean"] == pytest.approx(2.0)
        assert curves["A"][r_key]["mean"] == pytest.approx(1.0)
        assert curves["G"][r_key]["mean"] == pytest.approx(0.5)
        assert curves["Q"][r_key]["n_pairs"] == 12
        assert curves["random"][r_key]["mean"] == pytest.approx(0.0)
        assert curves["full_patch_direction"][r_key]["mean"] == pytest.approx(4.0)


def test_curve_for_rejects_a_missing_factorial_arm():
    rows = _factorial_rows()
    rows = rows[rows["condition"] != "Y11_both"]
    with pytest.raises(RuntimeError, match="missing factorial arms"):
        _curve_for(
            rows, value_col="delta_margin_toward_expected",
            n_bootstraps=50, confidence=0.95,
        )
