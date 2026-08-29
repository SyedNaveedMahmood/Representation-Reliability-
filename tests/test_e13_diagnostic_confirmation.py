"""Contract tests for the one-shot E13 diagnostic confirmation.

These exercise the locked inference, the identity locks and the split routing
without loading any model and without touching the confirmation namespace.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from representation_reliability.metrics.e13_diagnostic_confirmation import (
    behavior_noninferiority,
    classify_confirmation,
    component_mismatch,
    evaluate_regime_components,
    holm_adjust_components,
)
from representation_reliability.runners.e13 import validate_evaluation_routing
from representation_reliability.runners.e13_diagnostic_confirmation_support import (
    CHECKPOINT_REGISTRY,
    DELTA_B,
    DELTA_C,
    HOLDOUT_SPEC,
    HOLDOUT_SPEC_SHA256,
    TRAINING_SEEDS,
    canonical_digest,
    materialize_e13_holdout,
    open_e13_access_record,
    resolve_checkpoint,
)

PAIRS = 40


def _rows(values: np.ndarray) -> pd.DataFrame:
    """Two rows per pair, deterministic identities."""
    n = values.size
    return pd.DataFrame(
        {
            "base_sample_id": [f"s{index:04d}" for index in range(n)],
            "pair_id": [f"p{index // 2:04d}" for index in range(n)],
            "gap": values.astype(float),
        }
    )


def _margins(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    n = scores.size
    return pd.DataFrame(
        {
            "base_sample_id": [f"s{index:04d}" for index in range(n)],
            "pair_id": [f"p{index // 2:04d}" for index in range(n)],
            "gold_label": labels.astype(int),
            "clean_margin": scores.astype(float),
        }
    )


# --------------------------------------------------------------------------
# Holm correction
# --------------------------------------------------------------------------


def test_holm_adjust_components_is_step_down_and_monotone():
    adjusted = holm_adjust_components({"Q": 0.01, "A": 0.02, "G": 0.60})
    assert adjusted["Q"] == pytest.approx(0.03)
    assert adjusted["A"] == pytest.approx(0.04)
    assert adjusted["G"] == pytest.approx(0.60)
    assert adjusted["Q"] <= adjusted["A"] <= adjusted["G"]


def test_holm_adjust_components_clamps_at_one():
    adjusted = holm_adjust_components({"Q": 0.5, "A": 0.6, "G": 0.9})
    assert all(value <= 1.0 for value in adjusted.values())


def test_holm_rejects_a_family_that_is_not_exactly_qag():
    with pytest.raises(ValueError):
        holm_adjust_components({"Q": 0.1, "A": 0.2})
    with pytest.raises(ValueError):
        holm_adjust_components({"Q": 0.1, "A": 0.2, "G": 0.3, "B": 0.4})


# --------------------------------------------------------------------------
# Stage A: behavioural non-inferiority
# --------------------------------------------------------------------------


def _behaviour_fixture(shift: float) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(7)
    labels = np.tile([0, 1], PAIRS)
    teacher_scores = rng.normal(size=labels.size) + labels * 3.0
    teacher = _margins(teacher_scores, labels)
    per_seed = {
        int(seed): _margins(teacher_scores + labels * shift, labels)
        for seed in TRAINING_SEEDS
    }
    return per_seed, teacher


def test_behavior_noninferiority_passes_when_student_matches_teacher():
    per_seed, teacher = _behaviour_fixture(0.0)
    result = behavior_noninferiority(
        per_seed, teacher=teacher, delta_b=DELTA_B, n_draws=400, seed=11
    )
    assert result["verdict"] == "PASS"
    assert result["aggregate_delta_B"] == pytest.approx(0.0, abs=1e-12)
    assert result["seeds_noninferior"] == 3
    assert result["aggregate_ci_low"] > -DELTA_B


def test_behavior_noninferiority_fails_when_student_is_clearly_worse():
    rng = np.random.default_rng(3)
    labels = np.tile([0, 1], PAIRS)
    teacher_scores = rng.normal(size=labels.size) + labels * 4.0
    teacher = _margins(teacher_scores, labels)
    # Students carry almost no label signal, so B collapses toward chance.
    per_seed = {
        int(seed): _margins(rng.normal(size=labels.size) * 0.01, labels)
        for seed in TRAINING_SEEDS
    }
    result = behavior_noninferiority(
        per_seed, teacher=teacher, delta_b=DELTA_B, n_draws=400, seed=12
    )
    assert result["verdict"] == "FAIL"
    assert result["aggregate_delta_B"] < -DELTA_B
    assert result["seeds_noninferior"] == 0


def test_behavior_noninferiority_needs_two_of_three_seeds():
    rng = np.random.default_rng(5)
    labels = np.tile([0, 1], PAIRS)
    teacher_scores = rng.normal(size=labels.size) + labels * 4.0
    teacher = _margins(teacher_scores, labels)
    seeds = list(TRAINING_SEEDS)
    per_seed = {
        int(seeds[0]): _margins(teacher_scores, labels),
        int(seeds[1]): _margins(rng.normal(size=labels.size) * 0.01, labels),
        int(seeds[2]): _margins(rng.normal(size=labels.size) * 0.01, labels),
    }
    result = behavior_noninferiority(
        per_seed, teacher=teacher, delta_b=DELTA_B, n_draws=400, seed=13
    )
    assert result["seeds_noninferior"] == 1
    assert result["verdict"] == "FAIL"


def test_behavior_noninferiority_rejects_misaligned_rows():
    per_seed, teacher = _behaviour_fixture(0.0)
    broken = {key: value.copy() for key, value in per_seed.items()}
    first = next(iter(broken))
    broken[first] = broken[first].assign(
        base_sample_id=lambda f: f["base_sample_id"].str.replace("s0000", "sXXXX", regex=False)
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        behavior_noninferiority(broken, teacher=teacher, delta_b=DELTA_B, n_draws=50, seed=1)


def test_behavior_noninferiority_resamples_pairs_not_rows():
    """Every bootstrap draw must keep both members of a sampled pair together."""
    per_seed, teacher = _behaviour_fixture(0.0)
    result = behavior_noninferiority(
        per_seed, teacher=teacher, delta_b=DELTA_B, n_draws=200, seed=17
    )
    assert result["n_pairs"] == PAIRS
    assert result["n_seeds"] == 3


# --------------------------------------------------------------------------
# Stage B: component SESOI testing
# --------------------------------------------------------------------------


def test_component_mismatch_flags_a_gap_far_beyond_the_sesoi():
    rng = np.random.default_rng(21)
    per_seed = {
        int(seed): _rows(rng.normal(loc=0.45, scale=0.05, size=2 * PAIRS))
        for seed in TRAINING_SEEDS
    }
    result = component_mismatch(
        per_seed, component="A", delta_c=DELTA_C, n_draws=600, seed=23
    )
    assert result["aggregate_gap"] > DELTA_C
    assert result["ci_outside_sesoi"] is True
    assert result["ci_low"] > DELTA_C
    assert result["raw_p"] < 0.05
    assert result["seeds_beyond_sesoi_same_direction"] == 3
    assert result["direction"] == 1


def test_component_mismatch_spares_a_gap_inside_the_sesoi():
    rng = np.random.default_rng(22)
    per_seed = {
        int(seed): _rows(rng.normal(loc=0.0, scale=0.05, size=2 * PAIRS))
        for seed in TRAINING_SEEDS
    }
    result = component_mismatch(
        per_seed, component="G", delta_c=DELTA_C, n_draws=600, seed=24
    )
    assert abs(result["aggregate_gap"]) < DELTA_C
    assert result["ci_outside_sesoi"] is False
    assert result["raw_p"] == pytest.approx(1.0)


def test_component_mismatch_detects_negative_direction():
    rng = np.random.default_rng(25)
    per_seed = {
        int(seed): _rows(rng.normal(loc=-0.5, scale=0.05, size=2 * PAIRS))
        for seed in TRAINING_SEEDS
    }
    result = component_mismatch(
        per_seed, component="Q", delta_c=DELTA_C, n_draws=600, seed=26
    )
    assert result["direction"] == -1
    assert result["ci_high"] < -DELTA_C
    assert result["ci_outside_sesoi"] is True


def test_component_mismatch_rejects_nonfinite_gaps():
    per_seed = {int(seed): _rows(np.full(2 * PAIRS, np.nan)) for seed in TRAINING_SEEDS}
    with pytest.raises(RuntimeError, match="nonfinite"):
        component_mismatch(per_seed, component="A", delta_c=DELTA_C, n_draws=10, seed=1)


def test_evaluate_regime_components_applies_holm_within_the_regime():
    rng = np.random.default_rng(31)
    gaps = {
        "Q": {
            int(seed): _rows(rng.normal(0.0, 0.05, size=2 * PAIRS)) for seed in TRAINING_SEEDS
        },
        "A": {
            int(seed): _rows(rng.normal(0.5, 0.05, size=2 * PAIRS)) for seed in TRAINING_SEEDS
        },
        "G": {
            int(seed): _rows(rng.normal(0.0, 0.05, size=2 * PAIRS)) for seed in TRAINING_SEEDS
        },
    }
    frame = evaluate_regime_components(gaps, delta_c=DELTA_C, n_draws=600, seed=41)
    assert set(frame["component"]) == {"Q", "A", "G"}
    assert (frame["holm_p"] >= frame["raw_p"] - 1e-12).all()
    row_a = frame.loc[frame["component"] == "A"].iloc[0]
    assert bool(row_a["mismatch"]) is True
    assert not frame.loc[frame["component"] != "A", "mismatch"].any()


def test_evaluate_regime_components_requires_the_full_family():
    with pytest.raises(ValueError):
        evaluate_regime_components({"Q": {}}, delta_c=DELTA_C, n_draws=10, seed=1)


# --------------------------------------------------------------------------
# Hierarchical gatekeeping
# --------------------------------------------------------------------------


def _component_frame(mismatch_flags: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": name,
                "aggregate_gap": 0.4 if flag else 0.01,
                "ci_low": 0.3 if flag else -0.05,
                "ci_high": 0.5 if flag else 0.05,
                "mismatch": flag,
                "seeds_beyond_sesoi_same_direction": 3 if flag else 0,
            }
            for name, flag in mismatch_flags.items()
        ]
    )


def test_stage_b_cannot_pass_when_stage_a_fails():
    behavior = {
        "R2": {"verdict": "FAIL"},
        "R3": {"verdict": "PASS"},
    }
    components = {
        "R2": _component_frame({"Q": False, "A": True, "G": False}),
        "R3": _component_frame({"Q": False, "A": True, "G": False}),
    }
    result = classify_confirmation(behavior, components)
    assert result["regimes"]["R2"]["causal_mismatch"] == "FAIL"
    assert result["regimes"]["R3"]["causal_mismatch"] == "PASS"
    assert result["classification"] == "partial"
    assert result["entry_gate_for_cross_family"] is True


def test_classification_is_strong_only_when_both_regimes_pass_both_stages():
    behavior = {"R2": {"verdict": "PASS"}, "R3": {"verdict": "PASS"}}
    components = {
        "R2": _component_frame({"Q": False, "A": True, "G": False}),
        "R3": _component_frame({"Q": False, "A": True, "G": False}),
    }
    assert classify_confirmation(behavior, components)["classification"] == "strong"


def test_classification_fails_and_closes_the_entry_gate():
    behavior = {"R2": {"verdict": "PASS"}, "R3": {"verdict": "PASS"}}
    components = {
        "R2": _component_frame({"Q": False, "A": False, "G": False}),
        "R3": _component_frame({"Q": False, "A": False, "G": False}),
    }
    result = classify_confirmation(behavior, components)
    assert result["classification"] == "failed"
    assert result["entry_gate_for_cross_family"] is False


# --------------------------------------------------------------------------
# Split routing and identity locks
# --------------------------------------------------------------------------


def _routing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "split": ["train", "validation", "discovery_test", "confirmation"],
        }
    )


def test_routing_requires_the_confirmation_flag_for_confirmation_rows():
    frame = _routing_frame()
    with pytest.raises(RuntimeError, match="confirmation flag"):
        validate_evaluation_routing(frame, "confirmation", False)


def test_routing_forbids_the_confirmation_flag_on_open_rows():
    frame = _routing_frame()
    with pytest.raises(RuntimeError, match="confirmation flag"):
        validate_evaluation_routing(frame, "discovery_test", True)


def test_routing_accepts_the_two_legitimate_combinations():
    frame = _routing_frame()
    assert validate_evaluation_routing(frame, "discovery_test", False) == ["c"]
    assert validate_evaluation_routing(frame, "confirmation", True) == ["d"]


def test_routing_rejects_an_empty_evaluation_split():
    frame = _routing_frame()
    frame = frame[frame["split"] != "confirmation"]
    with pytest.raises(RuntimeError, match="selected no rows"):
        validate_evaluation_routing(frame, "confirmation", True)


def test_default_routing_is_the_open_discovery_split():
    assert validate_evaluation_routing(_routing_frame(), "discovery_test", False) == ["c"]


def test_holdout_specification_is_frozen():
    assert HOLDOUT_SPEC["namespace"] == "e13_confirmation_v1"
    assert HOLDOUT_SPEC["generator_seed"] == 20261304
    assert HOLDOUT_SPEC["n_directed"] == 200
    assert HOLDOUT_SPEC["n_pairs"] == 100
    assert len(HOLDOUT_SPEC["families"]) == 5


def test_checkpoint_registry_covers_exactly_the_frozen_grid():
    assert len(CHECKPOINT_REGISTRY) == 6
    for regime in ("R2", "R3"):
        for seed in TRAINING_SEEDS:
            entry = CHECKPOINT_REGISTRY[f"{regime}_seed_{seed}"]
            assert entry["regime"] == regime
            assert entry["seed"] == seed
            assert entry["step"] in (0, 10, 25, 50, 100)
            assert len(entry["run_identity_sha256"]) == 64
            assert len(entry["weight_sha256"]) == 64
    for key in CHECKPOINT_REGISTRY:
        if key.startswith("R3"):
            assert len(CHECKPOINT_REGISTRY[key]["projector_sha256"]) == 64


def test_registry_digest_is_stable_across_calls():
    assert canonical_digest(CHECKPOINT_REGISTRY) == canonical_digest(CHECKPOINT_REGISTRY)
    assert canonical_digest(HOLDOUT_SPEC) == canonical_digest(HOLDOUT_SPEC)


def test_resolve_checkpoint_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        resolve_checkpoint(tmp_path, "R2_seed_20261305")


def test_resolve_checkpoint_rejects_identity_drift(tmp_path):
    entry = CHECKPOINT_REGISTRY["R2_seed_20261305"]
    path = (
        tmp_path
        / "runs"
        / "E13_MULTI_SEED"
        / "E13MS_04daa7fcc66c"
        / "jobs"
        / str(entry["run_id"])
        / "checkpoints"
        / f"step_{int(entry['step']):03d}"
    )
    path.mkdir(parents=True)
    (path / "checkpoint.complete.json").write_text(
        json.dumps({"complete": True, "identity": "0" * 64, "step": entry["step"]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="identity drifted"):
        resolve_checkpoint(tmp_path, "R2_seed_20261305")


def test_resolve_checkpoint_rejects_an_incomplete_marker(tmp_path):
    entry = CHECKPOINT_REGISTRY["R3_seed_20261305"]
    path = (
        tmp_path
        / "runs"
        / "E13_MULTI_SEED"
        / "E13MS_04daa7fcc66c"
        / "jobs"
        / str(entry["run_id"])
        / "checkpoints"
        / f"step_{int(entry['step']):03d}"
    )
    path.mkdir(parents=True)
    (path / "checkpoint.complete.json").write_text(
        json.dumps({"complete": False, "identity": entry["run_identity_sha256"]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        resolve_checkpoint(tmp_path, "R3_seed_20261305")


# --------------------------------------------------------------------------
# Holdout materialization and the access ledger
# --------------------------------------------------------------------------


DECOY_SPEC = {
    **HOLDOUT_SPEC,
    "namespace": "e13_decoy_not_confirmation",
    "generator_seed": 999_001,
    "split_name": "decoy",
    "sample_id_prefix": "e13-decoy-v1-sample-",
    "pair_id_prefix": "e13-decoy-v1-pair-",
}


def test_frozen_holdout_spec_digest_without_materializing_rows():
    """Mirrors the E14 precedent: freeze the spec, never generate the rows."""
    assert canonical_digest(HOLDOUT_SPEC) == HOLDOUT_SPEC_SHA256


def test_materialization_produces_the_frozen_shape_on_a_decoy_namespace():
    samples, frame = materialize_e13_holdout(set(), spec=DECOY_SPEC)
    assert len(samples) == 200
    assert len(frame) == 200
    assert frame["pair_id"].nunique() == 100
    assert (frame["split"] == "confirmation").all()
    assert frame["sample_id"].astype(str).str.startswith("e13-decoy-v1-sample-").all()
    assert frame["pair_id"].astype(str).str.startswith("e13-decoy-v1-pair-").all()
    assert not frame["prompt"].duplicated().any()
    assert bool(frame.groupby("pair_id")["sample_id"].size().eq(2).all())


def test_materialization_is_deterministic_on_a_decoy_namespace():
    _first_samples, first = materialize_e13_holdout(set(), spec=DECOY_SPEC)
    _second_samples, second = materialize_e13_holdout(set(), spec=DECOY_SPEC)
    assert first["prompt"].tolist() == second["prompt"].tolist()
    assert first["sample_id"].tolist() == second["sample_id"].tolist()


def test_materialization_honours_the_deduplication_chain():
    """Signatures already used upstream must never be reissued."""
    _samples, baseline = materialize_e13_holdout(set(), spec=DECOY_SPEC)
    first_pair = baseline.iloc[:2]
    blocked = {(str(first_pair.iloc[0]["prompt"]), str(first_pair.iloc[1]["prompt"]))}
    _samples2, shifted = materialize_e13_holdout(blocked, spec=DECOY_SPEC)
    assert shifted["prompt"].tolist() != baseline["prompt"].tolist()
    assert len(shifted) == 200
    assert first_pair.iloc[0]["prompt"] not in set(shifted["prompt"])


def test_materialization_refuses_a_namespace_mismatch():
    broken = {**DECOY_SPEC, "sample_id_prefix": "wrong-prefix-"}
    with pytest.raises(RuntimeError, match="namespace violated"):
        materialize_e13_holdout(set(), spec=broken)


def test_access_record_opens_once_and_does_not_increment(tmp_path):
    identity = {"protocol_sha256": "abc", "protocol_commit": "def"}
    first = open_e13_access_record(tmp_path, campaign_id="camp", protocol_identity=identity)
    assert first["access_count"] == 1
    assert first["first_access_timestamp"]
    second = open_e13_access_record(tmp_path, campaign_id="camp", protocol_identity=identity)
    assert second["access_count"] == 1
    assert second["first_access_timestamp"] == first["first_access_timestamp"]


def test_access_record_refuses_a_second_campaign(tmp_path):
    identity = {"protocol_sha256": "abc", "protocol_commit": "def"}
    open_e13_access_record(tmp_path, campaign_id="camp", protocol_identity=identity)
    with pytest.raises(RuntimeError, match="already been accessed"):
        open_e13_access_record(tmp_path, campaign_id="other", protocol_identity=identity)


def test_frozen_margins_are_the_discovery_values():
    assert DELTA_B == 0.03
    assert DELTA_C == 0.10
