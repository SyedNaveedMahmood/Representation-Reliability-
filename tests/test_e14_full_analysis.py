import numpy as np
import pandas as pd
import pytest

from representation_reliability.analysis.e14_full import paired_behavior_auroc_change


def _behavior_rows(margins: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "base_sample_id": ["a0", "a1", "b0", "b1"],
            "pair_id": ["a", "a", "b", "b"],
            "gold_label": [0, 1, 0, 1],
            "yes_no_margin": margins,
        }
    )


def test_paired_behavior_auroc_change_uses_pair_clusters():
    reference = _behavior_rows([0.0, 0.1, 0.2, 0.3])
    comparison = _behavior_rows([-1.0, 1.0, -0.5, 0.5])
    result = paired_behavior_auroc_change(
        reference, comparison, n_bootstraps=40, seed=7
    )
    assert result["mean"] == pytest.approx(0.25)
    assert result["n_rows"] == 4
    assert result["n_clusters"] == 2
    assert np.isfinite(result["ci_low"])


def test_paired_behavior_rejects_label_mismatch():
    reference = _behavior_rows([0.0, 0.1, 0.2, 0.3])
    comparison = _behavior_rows([-1.0, 1.0, -0.5, 0.5])
    comparison.loc[0, "gold_label"] = 1
    with pytest.raises(RuntimeError, match="labels disagree"):
        paired_behavior_auroc_change(reference, comparison, n_bootstraps=10, seed=2)
