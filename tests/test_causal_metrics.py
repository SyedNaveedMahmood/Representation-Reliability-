import numpy as np
import pandas as pd

from representation_reliability.metrics.causal import (
    aggregate_intervention_rows,
    counterfactual_outcome,
    margin_toward_label,
    paired_control_contrast,
)


def test_margin_orientation():
    assert margin_toward_label(2.0, 1) == 2.0
    assert margin_toward_label(2.0, 0) == -2.0


def test_counterfactual_flip_is_a_transition_for_both_target_labels():
    assert counterfactual_outcome(0, 1, 1) == {
        "expected_label_after": 1,
        "counterfactual_flip": 1,
    }
    assert counterfactual_outcome(1, 0, 0) == {
        "expected_label_after": 1,
        "counterfactual_flip": 1,
    }
    assert counterfactual_outcome(1, 1, 1) == {
        "expected_label_after": 1,
        "counterfactual_flip": 0,
    }


def test_clustered_aggregate_and_seed_averaged_contrast():
    rows = []
    for pair in ("p0", "p1"):
        for base in (f"{pair}-a", f"{pair}-b"):
            rows.append(
                {
                    "condition": "truth_coordinate",
                    "alpha": 1.0,
                    "direction_seed": None,
                    "pair_id": pair,
                    "base_sample_id": base,
                    "delta_margin_toward_expected": 2.0,
                    "expected_label": 1,
                    "prediction_after": 1,
                    "counterfactual_flip": 1,
                    "delta_norm": 1.0,
                    "activation_norm": 10.0,
                }
            )
            for seed, effect in ((0, 0.2), (1, 0.4)):
                rows.append(
                    {
                        "condition": "random_direction",
                        "alpha": 1.0,
                        "direction_seed": seed,
                        "pair_id": pair,
                        "base_sample_id": base,
                        "delta_margin_toward_expected": effect,
                        "expected_label": 1,
                        "prediction_after": 0,
                        "counterfactual_flip": 0,
                        "delta_norm": 1.0,
                        "activation_norm": 10.0,
                    }
                )
    df = pd.DataFrame(rows)
    agg = aggregate_intervention_rows(df, n_bootstraps=50, confidence_level=0.95, seed=3)
    truth = agg[agg["condition"] == "truth_coordinate"].iloc[0]
    assert np.isclose(truth["mean_delta_margin_toward_expected"], 2.0)
    assert int(truth["n_pairs"]) == 2

    contrast = paired_control_contrast(
        df,
        treatment="truth_coordinate",
        control="random_direction",
        alpha=1.0,
        n_bootstraps=50,
        confidence_level=0.95,
        seed=4,
    )
    assert np.isclose(contrast["mean_difference"], 1.7)
