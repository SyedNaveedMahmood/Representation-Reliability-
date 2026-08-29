import numpy as np
import pandas as pd

from representation_reliability.runners.e13_revision import (
    _cosine_from_norms,
    _distribution,
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
