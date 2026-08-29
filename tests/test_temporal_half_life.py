"""Contract tests for the E15 decay-curve and half-life metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from representation_reliability.metrics.temporal_half_life import (
    curve_smoothness,
    half_life,
    horizon_condition_summary,
    paired_horizon_contrast,
    relative_curve,
    shuffled_decision_null,
)

HORIZONS = [1, 2, 4, 8, 16, 32]


def test_relative_curve_normalises_against_the_earliest_horizon():
    curve = relative_curve(HORIZONS, [2.0, 1.6, 1.0, 0.6, 0.3, 0.1])
    assert curve["relative"][0] == pytest.approx(1.0)
    assert curve["relative"][2] == pytest.approx(0.5)
    assert curve["baseline_positive"] is True


def test_relative_curve_respects_the_chance_floor_for_decodability():
    curve = relative_curve(HORIZONS, [1.0, 1.0, 1.0, 0.9, 0.8, 0.75], floor=0.5)
    assert curve["relative"][0] == pytest.approx(1.0)
    assert curve["relative"][-1] == pytest.approx(0.5)


def test_half_life_interpolates_between_bracketing_grid_points():
    curve = relative_curve(HORIZONS, [1.0, 0.9, 0.7, 0.4, 0.2, 0.05])
    result = half_life(
        HORIZONS, curve["relative"],
        baseline_positive=True, baseline_ci_excludes_zero=True,
    )
    assert result["status"] == "estimated"
    assert 4.0 < result["value"] < 8.0
    assert result["bracket"] == [4, 8]


def test_half_life_is_right_censored_when_the_curve_never_halves():
    curve = relative_curve(HORIZONS, [1.0, 0.99, 0.98, 0.97, 0.96, 0.95])
    result = half_life(
        HORIZONS, curve["relative"],
        baseline_positive=True, baseline_ci_excludes_zero=True,
    )
    assert result["status"] == "right_censored"
    assert result["value"] is None
    assert result["censored_at"] == 32


def test_half_life_is_refused_for_a_non_monotone_curve():
    curve = relative_curve(HORIZONS, [1.0, 0.2, 0.9, 0.1, 0.8, 0.05])
    result = half_life(
        HORIZONS, curve["relative"],
        baseline_positive=True, baseline_ci_excludes_zero=True,
    )
    assert result["status"] == "not_estimable"
    assert "curve_not_sufficiently_monotone" in result["reasons"]


def test_half_life_is_refused_when_the_baseline_effect_is_not_real():
    curve = relative_curve(HORIZONS, [1.0, 0.9, 0.7, 0.4, 0.2, 0.05])
    result = half_life(
        HORIZONS, curve["relative"],
        baseline_positive=True, baseline_ci_excludes_zero=False,
    )
    assert result["status"] == "not_estimable"
    assert "baseline_ci_includes_zero" in result["reasons"]


def test_curve_smoothness_flags_a_large_upward_step():
    smooth = curve_smoothness([1, 2, 4], [1.0, 0.6, 0.3])
    assert smooth["monotone_enough"] is True
    jumpy = curve_smoothness([1, 2, 4], [1.0, 0.2, 0.6])
    assert jumpy["monotone_enough"] is False
    assert jumpy["max_upward_step"] > 0.15


def _rows(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for horizon in (1, 4):
        for pair in range(20):
            for condition, scale in (("setpoint", 1.0), ("random_normmatched", 0.0)):
                for direction in range(1 if condition == "setpoint" else 3):
                    records.append(
                        {
                            "horizon": horizon,
                            "condition": condition,
                            "direction_index": direction,
                            "pair_id": f"p{pair}",
                            "base_sample_id": f"p{pair}-{condition}-{direction}-{horizon}",
                            "delta_margin_toward_expected": scale * (2.0 / horizon)
                            + rng.normal(0, 0.05),
                            "expected_label": pair % 2,
                            "delta_margin_raw": rng.normal(0, 0.05),
                        }
                    )
    return pd.DataFrame(records)


def test_horizon_condition_summary_clusters_on_pair_id():
    summary = horizon_condition_summary(
        _rows(), n_bootstraps=200, confidence_level=0.95, seed=0
    )
    assert set(summary["condition"]) == {"setpoint", "random_normmatched"}
    treat = summary[
        (summary["condition"] == "setpoint") & (summary["horizon"] == 1)
    ].iloc[0]
    assert treat["mean_effect"] == pytest.approx(2.0, abs=0.1)
    assert treat["n_pairs"] == 20


def test_paired_horizon_contrast_averages_control_directions_per_episode():
    rows = _rows()
    # give the treatment and control a shared per-episode key so pairing works
    rows["base_sample_id"] = rows["pair_id"]
    contrast = paired_horizon_contrast(
        rows, horizon=1, treatment="setpoint", control="random_normmatched",
        n_bootstraps=200, confidence_level=0.95, seed=0,
    )
    assert contrast["mean_difference"] == pytest.approx(2.0, abs=0.1)
    assert contrast["ci_excludes_zero"] is True
    assert contrast["n_pairs"] == 20


def test_shuffled_decision_null_rejects_a_real_effect_and_accepts_noise():
    rng = np.random.default_rng(3)
    labels = np.array([1, 0] * 60)
    orientation = np.where(labels == 1, 1.0, -1.0)
    real = orientation * (1.0 + rng.normal(0, 0.1, size=len(labels)))
    detected = shuffled_decision_null(real, labels, n_permutations=500, seed=1)
    assert detected["exceeds_null"] is True
    assert detected["observed"] == pytest.approx(1.0, abs=0.1)

    noise = rng.normal(0, 1.0, size=len(labels))
    undetected = shuffled_decision_null(noise, labels, n_permutations=500, seed=1)
    assert undetected["exceeds_null"] is False
