from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from representation_reliability.adapters.quantization import (
    normalize_precision,
    quantize_weight_only,
)
from representation_reliability.metrics.quantization import (
    average_random_contexts,
    evidence_is_finite,
    factorial_components,
    percent_change,
    summarize_factorial,
)
from representation_reliability.runners.e14 import _load_native_probe_reference, e14_profile


@pytest.mark.parametrize(
    ("arms", "expected"),
    [
        ((0.0, 2.0, 3.0, 5.0), (2.0, 3.0, 0.0)),
        ((0.0, 2.0, 0.0, 4.0), (2.0, 0.0, 2.0)),
        ((1.0, 3.0, 4.0, 8.0), (2.0, 3.0, 2.0)),
        ((0.0, 2.0, 1.0, 2.0), (2.0, 1.0, -1.0)),
    ],
)
def test_factorial_components_cover_additive_gating_mixed_and_suppression(arms, expected):
    result = factorial_components(*arms)
    assert float(result["Q0"]) == expected[0]
    assert float(result["A"]) == expected[1]
    assert float(result["G"]) == expected[2]
    assert np.allclose(result["G"], result["Q_context"] - result["Q0"])


def test_frozen_native_probe_is_loaded_without_refitting(tmp_path):
    np.savez(
        tmp_path / "native_probe_reference.npz",
        coef_17=np.array([1.0, 2.0]),
        intercept_17=np.array([0.5]),
        mean_17=np.array([3.0, 4.0]),
        scale_17=np.array([5.0, 6.0]),
        direction_17=np.array([0.6, 0.8]),
    )
    fits, directions = _load_native_probe_reference(tmp_path, [17])
    assert np.array_equal(fits[17]["coef"], [1.0, 2.0])
    assert np.array_equal(fits[17]["scale"], [5.0, 6.0])
    assert np.array_equal(directions[17], [0.6, 0.8])


def test_random_seeds_are_averaged_per_base_before_contrast():
    rows = pd.DataFrame(
        [
            {"base_sample_id": "a", "pair_id": "p", "context": "matched", "Q0": 1, "A": 4, "Q_context": 2, "G": 1},
            {"base_sample_id": "a", "pair_id": "p", "context": "random", "direction_seed": 1, "Q0": 1, "A": 1, "Q_context": 1, "G": 0},
            {"base_sample_id": "a", "pair_id": "p", "context": "random", "direction_seed": 2, "Q0": 1, "A": 3, "Q_context": 3, "G": 2},
        ]
    )
    averaged = average_random_contexts(rows)
    random = averaged[averaged["context"] == "random"].iloc[0]
    assert random["A"] == 2.0
    assert random["G"] == 1.0


def test_factorial_summary_uses_pair_cluster_bootstrap_and_paired_contrast():
    rows = []
    for pair, base in (("p1", "a"), ("p1", "b"), ("p2", "c"), ("p2", "d")):
        rows.extend(
            [
                {"base_sample_id": base, "pair_id": pair, "context": "matched", "Q0": 2.0, "A": 3.0, "Q_context": 3.0, "G": 1.0},
                {"base_sample_id": base, "pair_id": pair, "context": "random", "direction_seed": 1729, "Q0": 2.0, "A": 1.0, "Q_context": 2.0, "G": 0.0},
            ]
        )
    table, contrasts = summarize_factorial(
        pd.DataFrame(rows), n_bootstraps=40, confidence_level=0.95, seed=7
    )
    assert set(table["context"]) == {"matched", "random"}
    assert contrasts["A_matched_minus_random"]["mean"] == 2.0
    assert contrasts["A_matched_minus_random"]["n_clusters"] == 2
    assert contrasts["G_matched_minus_random"]["mean"] == 1.0


def test_profiles_and_precision_scope_are_fail_closed():
    assert e14_profile("smoke", None) == (25, 1, 200)
    assert e14_profile("pilot", None) == (75, 3, 500)
    assert e14_profile("full", None) == (150, 10, 2000)
    with pytest.raises(ValueError):
        e14_profile("smoke", 26)
    assert normalize_precision("INT4") == "int4"
    with pytest.raises(ValueError):
        normalize_precision("int3")


def test_bf16_backend_is_nonmutating_and_percent_change_is_guarded():
    model = torch.nn.Linear(3, 2)
    before = model.weight.detach().clone()
    manifest = quantize_weight_only(model, "bf16")
    assert manifest["backend"] == "none"
    assert torch.equal(model.weight, before)
    assert percent_change(2.0, 1.0) == 100.0
    assert percent_change(1.0, 0.0) is None


def test_finite_guard_allows_null_provenance_seed_but_rejects_nan_evidence():
    factorial = pd.DataFrame(
        [{
            "direction_seed": np.nan,
            "q_base": 0.0, "q_target": 1.0, "q_after_y01": 0.0, "q_after_y11": 1.0,
            "context_norm": 2.0, "context_dot_u": 0.0,
            "Y00": 0.0, "Y10": 1.0, "Y01": 2.0, "Y11": 3.0,
            "Q0": 1.0, "A": 2.0, "Q_context": 1.0, "G": 0.0,
        }]
    )
    trace = pd.DataFrame(
        [{
            "direction_seed": np.nan,
            "q00": 0.0, "q10": 1.0, "q01": 0.0, "q11": 1.0,
            "m00": 0.0, "m10": 1.0, "m01": 2.0, "m11": 3.0,
            "A_q_z": 0.0, "G_q_z": 0.0, "A_margin_z": 2.0, "G_margin_z": 0.0,
        }]
    )
    assert evidence_is_finite(factorial, trace)
    factorial.loc[0, "Q0"] = np.nan
    assert not evidence_is_finite(factorial, trace)


def test_quanto_int8_contract_on_tiny_linear():
    pytest.importorskip("optimum.quanto")
    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False))
    manifest = quantize_weight_only(model, "int8")
    assert manifest["backend"] == "optimum-quanto"
    assert manifest["activation_qtype"] is None
    assert len(manifest["quantized_modules"]) == 1
    assert manifest["quantized_modules"][0]["weight_qtype"] == "quanto.qint8"
