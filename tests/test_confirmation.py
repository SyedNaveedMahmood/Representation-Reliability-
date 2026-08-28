from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from representation_reliability.data.base import samples_to_dataframe
from representation_reliability.data.synthetic import generate_synthetic_relations
from representation_reliability.metrics.confirmation import (
    cluster_sign_flip_pvalue,
    confirmation_classification,
    evaluate_primary_hypotheses,
    holm_adjust,
    pair_cluster_bootstrap_ci,
)
from representation_reliability.metrics.factorial import factorial_estimands
from representation_reliability.runners import confirmation_support as support


def test_protocol_digest_and_commit_identity():
    repo_root = support.Path(__file__).resolve().parents[1]
    identity = support.validate_protocol_lock(repo_root, support.PREREGISTRATION_COMMIT)
    assert identity["protocol_sha256"] == support.PROTOCOL_SHA256
    with pytest.raises(RuntimeError, match="wrong confirmation protocol"):
        support.validate_protocol_lock(repo_root, "0" * 40)


def test_confirmation_routing_is_explicit_and_exact():
    frame = pd.DataFrame(
        {
            "sample_id": ["train", "holdout"],
            "split": ["train", "confirmation"],
        }
    )
    routed = support.route_confirmation_rows(frame)
    assert routed["sample_id"].tolist() == ["holdout"]


def test_probe_fit_rejects_confirmation_and_cannot_select_on_it():
    frame = pd.DataFrame(
        {
            "sample_id": ["t0", "t1", "v0", "v1", "c0"],
            "split": ["train", "train", "validation", "validation", "confirmation"],
            "target_label": [0, 1, 0, 1, 1],
        }
    )
    activations = {
        17: {
            sid: np.array([index, index % 2], float) for index, sid in enumerate(frame["sample_id"])
        }
    }
    with pytest.raises(RuntimeError, match="train/validation"):
        support.fit_locked_probe_layers(activations, frame, layers=[17], c_grid=[1.0], seed=2)
    directions, fits, digest = support.fit_locked_probe_layers(
        activations,
        frame[frame["split"] != "confirmation"],
        layers=[17],
        c_grid=[0.1, 1.0],
        seed=2,
    )
    assert directions[17].shape == (2,)
    assert fits[17]["chosen_C"] in {0.1, 1.0}
    assert len(digest) == 64


def test_model_target_and_analysis_identity_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="revision mismatch"):
        support.validate_model_identity(
            "Qwen/Qwen3-0.6B",
            resolved_revision="wrong",
            tokenizer_revision="wrong",
            candidate_token_ids=[7414, 2308],
        )
    with pytest.raises(RuntimeError, match="token IDs"):
        support.validate_model_identity(
            "Qwen/Qwen3-0.6B",
            resolved_revision=support.FROZEN_MODELS["Qwen/Qwen3-0.6B"]["revision"],
            tokenizer_revision=support.FROZEN_MODELS["Qwen/Qwen3-0.6B"]["revision"],
            candidate_token_ids=[1, 2],
        )
    bad = tmp_path / "target.json"
    bad.write_text("{}", encoding="utf-8")
    identity = dict(support.FROZEN_MODELS["Qwen/Qwen3-0.6B"])
    identity["target_path"] = "target.json"
    monkeypatch.setitem(support.FROZEN_MODELS, "Qwen/Qwen3-0.6B", identity)
    with pytest.raises(RuntimeError, match="target digest"):
        support.load_frozen_targets(tmp_path, "Qwen/Qwen3-0.6B")
    with pytest.raises(RuntimeError, match="source-plan digest"):
        support.require_identity_digest("source-plan", "bad", "frozen")
    monkeypatch.setitem(support.PRIMARY_HYPOTHESES, "H4", "modified")
    assert support.analysis_definition_digest() != support.ANALYSIS_DEFINITION_SHA256


def test_source_plan_is_deterministic_and_respects_confirmation_invariants():
    samples = generate_synthetic_relations(n_samples=400, seed=17)
    frame = samples_to_dataframe(samples)
    frame["split"] = "confirmation"
    by_id = {str(sample.sample_id): sample for sample in samples}
    first = support.build_confirmation_source_plans(frame, by_id, seed=23)
    second = support.build_confirmation_source_plans(frame, by_id, seed=23)
    first_frame = support.source_plan_frame(first)
    second_frame = support.source_plan_frame(second)
    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert support.source_plan_digest(first_frame) == support.source_plan_digest(second_frame)
    rows = frame.set_index("sample_id")
    for sid, plan in first.items():
        base = rows.loc[sid]
        assert rows.loc[plan.matched_source_id, "pair_id"] == base["pair_id"]
        assert rows.loc[plan.same_family_source_id, "pair_id"] != base["pair_id"]
        assert rows.loc[plan.same_family_source_id, "relation"] == base["relation"]
        assert rows.loc[plan.different_family_source_id, "relation"] != base["relation"]
        assert rows.loc[plan.same_label_source_id, "target_label"] == base["target_label"]


def test_holm_and_cluster_inference_are_deterministic():
    adjusted = holm_adjust({"H1": 0.01, "H2": 0.02, "H3": 0.03, "H4": 0.04})
    assert adjusted == {"H1": 0.04, "H2": 0.06, "H3": 0.06, "H4": 0.06}
    values = np.arange(1.0, 21.0)
    pairs = [f"p{i}" for i in range(20)]
    assert pair_cluster_bootstrap_ci(
        values, pairs, n_draws=200, seed=9
    ) == pair_cluster_bootstrap_ci(values, pairs, n_draws=200, seed=9)
    assert cluster_sign_flip_pvalue(values, pairs, n_draws=1000, seed=9) < 0.01


def _scalar_frame(model: str) -> pd.DataFrame:
    rows = []
    for index in range(24):
        base = f"s{index}"
        pair = f"p{index // 2}"
        for condition, value, seeds in (
            ("source_free_opposite_class_median", 1.0 if "1.7" in model else 0.5, [None]),
            ("random_direction", 0.1, [1, 2]),
            ("orthogonal_random", 0.05, [3, 4]),
        ):
            for seed in seeds:
                rows.append(
                    {
                        "model_id": model,
                        "base_sample_id": base,
                        "pair_id": pair,
                        "condition": condition,
                        "direction_seed": seed,
                        "delta_margin_toward_target": value,
                    }
                )
    return pd.DataFrame(rows)


def _factorial_frame(model: str) -> pd.DataFrame:
    rows = []
    large = "1.7" in model
    for index in range(24):
        base = f"s{index}"
        pair = f"p{index // 2}"
        for condition, additive, interaction, seeds in (
            ("matched_orthogonal", 2.0 if large else 1.0, 0.4 if large else 0.02, [None]),
            ("random_orthogonal", 0.1, 0.02, [1, 2]),
        ):
            for seed in seeds:
                rows.append(
                    {
                        "model_id": model,
                        "base_sample_id": base,
                        "pair_id": pair,
                        "condition": condition,
                        "direction_seed": seed,
                        "lambda_context": 1.0,
                        "A_context": additive,
                        "G_interaction": interaction,
                    }
                )
                rows.append(
                    {
                        "model_id": model,
                        "base_sample_id": base,
                        "pair_id": pair,
                        "condition": condition,
                        "direction_seed": seed,
                        "lambda_context": 0.5,
                        "A_context": -100.0,
                        "G_interaction": -100.0,
                    }
                )
    return pd.DataFrame(rows)


def test_primary_hypothesis_algebra_and_cross_model_h4():
    models = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
    primary, details = evaluate_primary_hypotheses(
        {model: _scalar_frame(model) for model in models},
        {model: _factorial_frame(model) for model in models},
        n_bootstraps=200,
        n_randomizations=2000,
        seed=101,
    )
    estimates = primary.set_index("hypothesis")["estimate"]
    assert estimates["H1"] == pytest.approx(0.4)
    assert estimates["H2"] == pytest.approx(0.9)
    assert estimates["H3"] == pytest.approx(0.38)
    assert estimates["H4"] == pytest.approx(0.38)
    assert set(details) == {"H1", "H2", "H3", "H4"}
    assert confirmation_classification(primary) == "strong"


def test_factorial_trace_algebra_covers_additive_gating_mixed_and_suppression():
    cases = {
        "additive": (0.0, 1.0, 2.0, 3.0, 2.0, 0.0),
        "gating": (0.0, 1.0, 0.0, 2.0, 0.0, 1.0),
        "mixed": (0.0, 1.0, 2.0, 4.0, 2.0, 1.0),
        "suppression": (0.0, 2.0, 1.0, 2.0, 1.0, -1.0),
    }
    for y00, y10, y01, y11, expected_a, expected_g in cases.values():
        result = factorial_estimands(y00, y10, y01, y11)
        assert float(result["A_context"]) == pytest.approx(expected_a)
        assert float(result["G_interaction"]) == pytest.approx(expected_g)
        assert float(result["G_interaction"]) == pytest.approx(
            float(result["Q_context"] - result["Q0"])
        )


def test_single_use_access_record_allows_same_campaign_resume_only(tmp_path):
    identity = {
        "protocol_commit": support.PREREGISTRATION_COMMIT,
        "protocol_sha256": support.PROTOCOL_SHA256,
        "analysis_definition_sha256": support.ANALYSIS_DEFINITION_SHA256,
    }
    first = support.open_access_record(
        tmp_path,
        campaign_id="one",
        git_commit="abc",
        protocol_identity=identity,
        environment={"python": "test"},
    )
    resumed = support.open_access_record(
        tmp_path,
        campaign_id="one",
        git_commit="def",
        protocol_identity=identity,
        environment={"python": "test"},
    )
    assert resumed == first
    assert first["confirmation_access_count"] == 1
    with pytest.raises(RuntimeError, match="second campaign"):
        support.open_access_record(
            tmp_path,
            campaign_id="two",
            git_commit="abc",
            protocol_identity=identity,
            environment={},
        )
