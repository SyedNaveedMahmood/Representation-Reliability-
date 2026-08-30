"""Contracts for the E19 curve bootstrap, Holm family and component algebra."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from representation_reliability.metrics.temporal_half_life import (
    bootstrap_two_sided_p,
    curve_cluster_bootstrap,
    holm_adjust,
)
from representation_reliability.runners.e19 import (
    COMPONENTS,
    ESTIMANDS,
    HORIZONS,
    K0,
    LOCI,
    SESOI_RELATIVE,
    SUFFICIENCY_FLIP,
    _component_frame,
    _twin_map,
)


# ------------------------------------------------------------ frozen constants
def test_loci_are_the_two_e18_validated_sites_at_opposite_ends():
    names = [locus["name"] for locus in LOCI]
    assert names == ["S_source", "D_decision"]
    source, decision = LOCI
    assert (source["site"], source["layer"]) == ("state_word_last", 8)
    assert (decision["site"], decision["layer"]) == ("decision", 24)
    # The design's whole decomposition rests on these two labels.
    assert source["grows"] == "age_and_distance"
    assert decision["grows"] == "age_only"
    # Propagation layers must lie strictly after their edit layer.
    for locus in LOCI:
        assert all(p > locus["layer"] for p in locus["propagation"])


def test_horizon_grid_and_thresholds_are_frozen():
    assert list(HORIZONS) == [1, 2, 4, 8]
    assert K0 == HORIZONS[0]
    assert SESOI_RELATIVE == 0.25
    assert SUFFICIENCY_FLIP == 0.10
    assert tuple(COMPONENTS) == ("Q", "A", "G")
    assert tuple(ESTIMANDS) == ("native", "ref")


# --------------------------------------------------------------- curve boostrap
def _curve_frame(per_horizon: dict[int, float], n_clusters: int = 40) -> pd.DataFrame:
    rows = []
    for cluster in range(n_clusters):
        for horizon, value in per_horizon.items():
            rows.append(
                {"episode_index": cluster, "horizon": horizon, "value": value}
            )
    return pd.DataFrame(rows)


def test_curve_bootstrap_recovers_the_per_horizon_means():
    frame = _curve_frame({1: 2.0, 2: 1.5, 4: 1.0, 8: 0.5})
    boot = curve_cluster_bootstrap(
        frame, cluster_col="episode_index", horizon_col="horizon",
        value_col="value", n_bootstraps=200, confidence_level=0.95, seed=0,
    )
    assert boot["horizons"] == [1, 2, 4, 8]
    np.testing.assert_allclose(boot["point"], [2.0, 1.5, 1.0, 0.5])
    assert boot["n_clusters"] == 40
    assert boot["draws"].shape == (200, 4)


def test_curve_bootstrap_resamples_whole_curves_as_units():
    """A cluster's entire trajectory must move together across horizons."""
    rows = []
    for cluster in range(30):
        # Half the episodes are uniformly high, half uniformly low, so a
        # per-horizon-independent resample would decorrelate the columns and a
        # curve-level one must not.
        level = 10.0 if cluster % 2 == 0 else 0.0
        for horizon in (1, 2, 4, 8):
            rows.append({"episode_index": cluster, "horizon": horizon, "value": level})
    boot = curve_cluster_bootstrap(
        pd.DataFrame(rows), cluster_col="episode_index", horizon_col="horizon",
        value_col="value", n_bootstraps=400, confidence_level=0.95, seed=1,
    )
    draws = boot["draws"]
    # Every column is the same resampled cluster mean, so correlation is exact.
    corr = np.corrcoef(draws[:, 0], draws[:, 3])[0, 1]
    assert corr == pytest.approx(1.0, abs=1e-9)


def test_curve_bootstrap_rejects_degenerate_input():
    with pytest.raises(ValueError, match="at least two clusters"):
        curve_cluster_bootstrap(
            _curve_frame({1: 1.0}, n_clusters=1), cluster_col="episode_index",
            horizon_col="horizon", value_col="value",
            n_bootstraps=10, confidence_level=0.95, seed=0,
        )
    with pytest.raises(ValueError, match="missing columns"):
        curve_cluster_bootstrap(
            pd.DataFrame({"a": [1]}), cluster_col="episode_index",
            horizon_col="horizon", value_col="value",
            n_bootstraps=10, confidence_level=0.95, seed=0,
        )


# --------------------------------------------------------------- p-values, Holm
def test_two_sided_bootstrap_p_is_small_for_a_clear_effect():
    assert bootstrap_two_sided_p(np.full(500, 3.0)) == pytest.approx(0.0)
    assert bootstrap_two_sided_p(np.linspace(-1.0, 1.0, 501)) > 0.5


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm_adjust({"Q": 0.001, "A": 0.02, "G": 0.60})
    assert adjusted["Q"] == pytest.approx(0.003)
    assert adjusted["A"] == pytest.approx(0.04)
    assert adjusted["G"] == pytest.approx(0.60)
    assert adjusted["Q"] <= adjusted["A"] <= adjusted["G"]
    assert all(v <= 1.0 for v in adjusted.values())


def test_holm_handles_a_non_finite_member():
    adjusted = holm_adjust({"Q": 0.01, "A": float("nan")})
    assert adjusted["Q"] == pytest.approx(0.01)
    assert np.isnan(adjusted["A"])


# ------------------------------------------------------------ component algebra
def test_component_frame_reproduces_the_frozen_factorial_algebra():
    rows = []
    for episode in range(5):
        for horizon in (1, 2):
            for condition, value in (
                ("Y10_scalar", 2.0), ("Y01_context", 1.0), ("Y11_both", 3.5),
            ):
                rows.append({
                    "locus": "S_source", "estimand": "native", "condition": condition,
                    "episode_index": episode, "horizon": horizon,
                    "base_sample_id": f"e{episode}h{horizon}",
                    "delta_margin_toward_expected": value,
                })
    wide = _component_frame(pd.DataFrame(rows), "S_source", "native")
    assert set(wide["Q"]) == {2.0}
    assert set(wide["A"]) == {1.0}
    # G = (Y11 - Y10) - Y01 = (3.5 - 2.0) - 1.0
    assert set(wide["G"]) == {0.5}


def test_component_frame_isolates_locus_and_estimand():
    rows = []
    for locus, estimand, scale in (
        ("S_source", "native", 1.0), ("D_decision", "native", 5.0),
        ("S_source", "ref", 9.0),
    ):
        for episode in range(4):
            for condition in ("Y10_scalar", "Y01_context", "Y11_both"):
                rows.append({
                    "locus": locus, "estimand": estimand, "condition": condition,
                    "episode_index": episode, "horizon": 1,
                    "base_sample_id": f"{locus}{estimand}{episode}",
                    "delta_margin_toward_expected": scale,
                })
    frame = pd.DataFrame(rows)
    assert set(_component_frame(frame, "S_source", "native")["Q"]) == {1.0}
    assert set(_component_frame(frame, "D_decision", "native")["Q"]) == {5.0}
    assert set(_component_frame(frame, "S_source", "ref")["Q"]) == {9.0}


def test_twin_map_is_reciprocal_and_complete():
    view = pd.DataFrame(
        {"sample_id": ["a0", "a1", "b0", "b1"], "pair_id": ["a", "a", "b", "b"]}
    )
    twins = _twin_map(view)
    assert twins == {"a0": "a1", "a1": "a0", "b0": "b1", "b1": "b0"}
    with pytest.raises(RuntimeError, match="incomplete"):
        _twin_map(pd.DataFrame({"sample_id": ["a0"], "pair_id": ["a"]}))


def test_holm_p_of_exactly_zero_is_the_strongest_result_not_a_falsy_miss():
    """Regression: `(p or 1.0) < 0.05` silently discards p == 0.0."""
    holm_p = 0.0
    assert not ((holm_p or 1.0) < 0.05)  # the trap
    significant = holm_p is not None and np.isfinite(holm_p) and holm_p < 0.05
    assert significant

    from representation_reliability.metrics.temporal_half_life import (
        bootstrap_two_sided_p as _p,
    )

    # A contrast entirely on one side of zero yields exactly 0.0, so this is
    # the ordinary case for a strong effect, not an edge case.
    assert _p(np.full(1000, -0.35)) == 0.0
    assert holm_adjust({"A": 0.0, "Q": 0.0, "G": 0.4})["A"] == 0.0
