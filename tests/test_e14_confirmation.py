import pandas as pd
import pytest

from representation_reliability.metrics.e14_confirmation import (
    classify_e14_confirmation,
    evaluate_e14_hypotheses,
    holm_adjust_family,
)
from representation_reliability.runners.e14_confirmation_support import (
    HOLDOUT_SPEC,
    HOLDOUT_SPEC_SHA256,
    canonical_digest,
    validate_e14_confirmation_lock,
)


def test_frozen_e14_holdout_spec_digest_without_materializing_rows():
    assert canonical_digest(HOLDOUT_SPEC) == HOLDOUT_SPEC_SHA256


def test_e14_protocol_rejects_wrong_commit_before_any_access(tmp_path):
    with pytest.raises(RuntimeError, match="wrong E14 confirmation protocol commit"):
        validate_e14_confirmation_lock(tmp_path, "not-the-frozen-commit")


def test_holm_family_is_exact_and_monotone():
    adjusted = holm_adjust_family({"H14.1": 0.02, "H14.2": 0.001, "H14.3": 0.01})
    assert adjusted["H14.2"] == pytest.approx(0.003)
    assert adjusted["H14.3"] == pytest.approx(0.02)
    assert adjusted["H14.1"] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        holm_adjust_family({"H14.1": 0.1})


def _factorial(precision: str, g: float, a: float) -> pd.DataFrame:
    rows = []
    for pair in range(10):
        for side in range(2):
            sid = f"p{pair}s{side}"
            rows.extend(
                [
                    {
                        "precision": precision,
                        "base_sample_id": sid,
                        "pair_id": f"p{pair}",
                        "context": "matched",
                        "direction_seed": None,
                        "Q0": 1.0,
                        "Q_context": 1.0 + g,
                        "A": a,
                        "G": g,
                    },
                    {
                        "precision": precision,
                        "base_sample_id": sid,
                        "pair_id": f"p{pair}",
                        "context": "random",
                        "direction_seed": 1729,
                        "Q0": 1.0,
                        "Q_context": 1.0,
                        "A": 0.0,
                        "G": 0.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _behavior(precision: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "precision": precision,
                "base_sample_id": f"p{pair}s{side}",
                "pair_id": f"p{pair}",
                "gold_label": side,
                "native_probe_score": -5.0 if side == 0 else 5.0,
            }
            for pair in range(10)
            for side in range(2)
        ]
    )


def test_e14_primary_algebra_and_classification():
    behavior = {key: _behavior(key) for key in ("bf16", "int8", "int4")}
    factorial = {
        "bf16": _factorial("bf16", 1.0, 2.0),
        "int8": _factorial("int8", 0.9, 1.9),
        "int4": _factorial("int4", 0.1, 0.5),
    }
    primary, details = evaluate_e14_hypotheses(
        behavior, factorial, bootstrap_draws=100, randomization_draws=2000, seed=3
    )
    estimates = dict(zip(primary["hypothesis"], primary["estimate"]))
    assert estimates["H14.1"] == pytest.approx(0.01)
    assert estimates["H14.2"] == pytest.approx(0.9)
    assert estimates["H14.3"] == pytest.approx(1.5)
    assert len(details["H14.2"]) == 20
    assert classify_e14_confirmation(primary) == "strong"
