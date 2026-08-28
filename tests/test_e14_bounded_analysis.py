from __future__ import annotations

import pandas as pd

from representation_reliability.analysis.e14_bounded import paired_precision_changes


def _rows(q: float, a_matched: float, a_random: float, g_matched: float, g_random: float):
    output = []
    for pair, base in (("p1", "a"), ("p1", "b"), ("p2", "c"), ("p2", "d")):
        output.extend(
            [
                {"base_sample_id": base, "pair_id": pair, "context": "matched", "Q0": q, "A": a_matched, "Q_context": q + g_matched, "G": g_matched},
                {"base_sample_id": base, "pair_id": pair, "context": "random", "direction_seed": 1, "Q0": q, "A": a_random, "Q_context": q + g_random, "G": g_random},
            ]
        )
    return pd.DataFrame(output)


def test_paired_precision_changes_use_identical_examples_and_pair_clusters():
    reference = _rows(2.0, 4.0, 1.0, 1.5, 0.5)
    comparison = _rows(3.0, 3.0, 1.0, 1.0, 0.5)
    result = paired_precision_changes(reference, comparison, n_bootstraps=20, seed=4)
    assert result["Q"]["mean"] == 1.0
    assert result["A"]["mean"] == -1.0
    assert result["G"]["mean"] == -0.5
    assert result["G"]["n_clusters"] == 2
    assert result["G"]["percent_change"] == -50.0
