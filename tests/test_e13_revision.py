import numpy as np
import pandas as pd

from representation_reliability.runners.e13_revision import (
    _cosine_from_norms,
    _distribution,
    _fixed_point_free_permutation,
    _revision_validation_geometry,
    _teacher_signal_tables,
)


def test_distribution_reports_frozen_summary_and_near_zero_fraction():
    result = _distribution(pd.Series([-2.0, 0.0, 0.0005, 2.0]))
    assert result["n"] == 4
    assert np.isclose(result["mean"], 0.000125)
    assert result["sd"] > 1.0
    assert result["fraction_near_zero"] == 0.5
    assert result["near_zero_threshold"] == 1e-3


def test_teacher_signal_tables_keep_semantic_and_random_families_separate():
    rows = []
    for split in ("train", "validation"):
        for value in range(4):
            rows.append(
                {
                    "split": split,
                    "R5_Q": float(value),
                    "R5_A": float(2 * value),
                    "R5_G": float(-value),
                    "R6_Q": float(3 * value),
                    "R6_A": float(4 * value),
                    "R6_G": float(-2 * value),
                }
            )
    distributions, correlations = _teacher_signal_tables(pd.DataFrame(rows))
    assert len(distributions) == 12
    assert set(distributions["family"]) == {"semantic", "random"}
    observed = correlations.loc[
        correlations["split"].eq("train")
        & correlations["left"].eq("R5_A")
        & correlations["right"].eq("R6_A"),
        "pearson",
    ].iloc[0]
    assert observed == 1.0


def test_cosine_is_recovered_from_three_gradient_norms():
    left = np.asarray([1.0, 0.0])
    right = np.asarray([-0.5, np.sqrt(3.0) / 2.0])
    observed = _cosine_from_norms(
        np.linalg.norm(left), np.linalg.norm(right), np.linalg.norm(left + right)
    )
    assert np.isclose(observed, -0.5)


def test_pair_permutation_is_deterministic_and_fixed_point_free():
    items = [f"pair-{index}" for index in range(20)]
    first = _fixed_point_free_permutation(items, 20261331)
    second = _fixed_point_free_permutation(items, 20261331)
    assert first == second
    assert set(first.values()) == set(items)
    assert all(source != target for source, target in first.items())


def test_revision_validation_geometry_uses_validation_only():
    frame = pd.DataFrame(
        {
            "split": ["train", "validation", "validation", "validation", "validation"],
            "R5_Q": [999.0, -2.0, -1.0, 1.0, 2.0],
            "R5_A": [999.0, -1.0, 2.0, -2.0, 1.0],
            "R5_G": [999.0, 0.5, -0.5, 1.5, -1.5],
        }
    )
    result = _revision_validation_geometry(frame, {"response_tensor_sha256": "cache-digest"})
    assert result["source_split"] == "validation"
    assert result["source_cache_digest"] == "cache-digest"
    assert result["R12_epsilon"] > 0
    assert min(result["R12_eigenvalues"]) > 0
    assert np.isfinite(result["R12_condition_number"])
