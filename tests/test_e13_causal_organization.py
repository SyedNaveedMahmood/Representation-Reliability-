import numpy as np
import pandas as pd
import pytest

from representation_reliability.metrics.causal_organization import (
    add_factorial_effect_views,
    causal_organization_distance,
    controlled_profiles,
    factorial_effect_views,
    immutable_run_identity,
    linear_cka,
    matched_profiles,
    representation_similarity,
    select_b_matched_checkpoint,
    validation_margin_statistics,
)


def test_validation_only_scaling_and_factorial_z_algebra():
    stats = validation_margin_statistics([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1])
    assert stats["source_split"] == "validation"
    assert stats["sigma_margin_validation"] == pytest.approx(np.sqrt(2.5))
    views = factorial_effect_views(
        np.array([-1.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3.0]),
        sigma_margin_validation=2.0,
    )
    assert views["Q_raw"][0] == pytest.approx(2.0)
    assert views["A_raw"][0] == pytest.approx(1.0)
    assert views["G_raw"][0] == pytest.approx(1.0)
    assert views["Q_z"][0] == pytest.approx(1.0)
    assert views["A_z"][0] == pytest.approx(0.5)
    assert views["G_z"][0] == pytest.approx(0.5)


def test_probability_factorial_and_strict_flip_metrics():
    views = factorial_effect_views(
        [-2.0, 2.0], [2.0, 3.0], [-1.0, -3.0], [3.0, 4.0],
        sigma_margin_validation=1.0,
    )
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    expected_g = (sigmoid(3.0) - sigmoid(2.0)) - (
        sigmoid(-1.0) - sigmoid(-2.0)
    )
    assert views["G_prob"][0] == pytest.approx(expected_g)
    assert views["q_target_flip"].tolist() == [1, 0]
    assert views["context_target_flip"].tolist() == [0, 0]
    assert views["joint_target_flip"].tolist() == [1, 0]


def _factorial_rows(offset=0.0):
    rows = []
    for sample, pair, q, a, g in (
        ("a", "p1", 1.0, 2.0, 3.0),
        ("b", "p1", -1.0, -2.0, -3.0),
        ("c", "p2", 0.5, 1.0, -0.5),
    ):
        for context, random_offset in (("matched", 0.0), ("random", 0.5), ("random", -0.5)):
            rows.append(
                {
                    "base_sample_id": sample,
                    "pair_id": pair,
                    "context": context,
                    "Q_z": q + offset,
                    "A_z": a + offset + (random_offset if context == "random" else 0.0),
                    "G_z": g + offset + (random_offset if context == "random" else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_cod_alignment_components_correlations_and_controls():
    teacher = matched_profiles(_factorial_rows())
    student = matched_profiles(_factorial_rows(offset=1.0))
    result = causal_organization_distance(teacher, student)
    assert result["COD"] == pytest.approx(np.sqrt(3.0))
    assert result["mean_abs_Q_z_gap"] == pytest.approx(1.0)
    assert result["pearson_profile_correlation"] is not None
    controlled = controlled_profiles(_factorial_rows())
    assert len(controlled) == 3
    assert controlled.loc[controlled.base_sample_id.eq("a"), "A_z"].item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="sample IDs"):
        causal_organization_distance(teacher, student.iloc[:-1])


def test_cod_degenerate_correlations_report_na():
    profile = pd.DataFrame(
        {
            "base_sample_id": ["a", "b"],
            "pair_id": ["p", "p"],
            "Q_z": [1.0, 1.0],
            "A_z": [1.0, 1.0],
            "G_z": [1.0, 1.0],
        }
    )
    result = causal_organization_distance(profile, profile)
    assert result["pearson_profile_correlation"] is None
    assert result["pearson_na_reason"] == "degenerate variance"


def test_b_matched_selection_tie_break_and_discovery_leak_guard():
    rows = pd.DataFrame(
        {
            "step": [0, 10, 25],
            "validation_B": [0.7, 0.9, 0.9],
            "selection_split": ["validation"] * 3,
            "discovery_B": [1.0, 0.0, 0.95],
        }
    )
    selected = select_b_matched_checkpoint(rows, 0.8)
    assert selected["selected_step"] == 0
    assert selected["selection_split"] == "validation"
    rows.loc[0, "selection_split"] = "discovery_test"
    with pytest.raises(ValueError, match="validation"):
        select_b_matched_checkpoint(rows, 0.8)


def test_representation_similarity_and_linear_cka_contract():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    projected = np.pad(x, ((0, 0), (0, 1)))
    target = projected.copy()
    assert linear_cka(x, target) == pytest.approx(1.0)
    diagnostics = representation_similarity(x, target, projected)
    assert diagnostics["mean_cosine_after_projector"] == pytest.approx(1.0)
    assert diagnostics["projected_hidden_MSE"] == pytest.approx(0.0)


def test_multiseed_manifest_identity_is_seed_and_regime_isolated():
    base = {"protocol": "digest", "corpus": "fixed", "seed": 20261305, "regime": "R1"}
    first = immutable_run_identity(base)
    assert first == immutable_run_identity(dict(base))
    assert first != immutable_run_identity({**base, "seed": 20261315})
    assert first != immutable_run_identity({**base, "regime": "R2"})


def test_add_factorial_views_marks_validation_scale_source():
    rows = pd.DataFrame({"Y00": [-1.0], "Y10": [1.0], "Y01": [0.0], "Y11": [2.0]})
    result = add_factorial_effect_views(rows, sigma_margin_validation=2.0)
    assert result.loc[0, "Q_z"] == pytest.approx(1.0)
    assert result.loc[0, "scale_source_split"] == "validation"
