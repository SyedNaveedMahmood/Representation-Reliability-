import pandas as pd

from representation_reliability.runners.e13_multiseed import (
    dataframe_to_markdown,
    evaluate_method_gate,
)


def _reference():
    return {
        "R0": {
            "D_native": {"auroc": 1.0},
            "D_frozen_initial_axis": {"auroc": 0.99},
        },
        "R0_quality": {
            "wikitext": {"perplexity": 20.0},
            "hellaswag": {"accuracy": 0.5},
        },
    }


def _job(seed, regime="R2", gap=0.02, cod=1.0, component=0.3, ppl=25.0, hs=0.48):
    return {
        "regime": regime,
        "seed": seed,
        "b_matched": {"selected_step": 10, "absolute_B_gap": gap},
        "checkpoints": [
            {
                "step": 10,
                "COD": {
                    "COD": cod,
                    "mean_abs_Q_z_gap": component,
                    "mean_abs_A_z_gap": 0.1,
                    "mean_abs_G_z_gap": 0.1,
                },
            }
        ],
        "general_quality": {
            "step_010": {
                "wikitext": {"perplexity": ppl},
                "hellaswag": {"accuracy": hs},
            }
        },
    }


def test_method_gate_requires_two_joint_seed_passes():
    jobs = [_job(1), _job(2), _job(3, gap=0.04)]
    gate = evaluate_method_gate(_reference(), jobs)
    assert gate["Gate_A"] and gate["Gate_D"] and gate["Gate_E"]
    assert gate["conversion_response_authorized"]


def test_method_gate_stops_on_quality_or_component_floor():
    jobs = [_job(1, component=0.1), _job(2, ppl=250.0), _job(3, gap=0.04)]
    gate = evaluate_method_gate(_reference(), jobs)
    assert not gate["Gate_D"]
    assert not gate["conversion_response_authorized"]


def test_report_table_does_not_require_optional_tabulate():
    rendered = dataframe_to_markdown(
        pd.DataFrame({"regime": ["R2"], "COD": [0.123456789], "note": ["a|b"]})
    )
    assert "0.123457" in rendered
    assert "a\\|b" in rendered
