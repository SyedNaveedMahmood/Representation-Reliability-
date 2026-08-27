from __future__ import annotations

import inspect

import numpy as np
import pytest

from representation_reliability.adapters.hf import (
    instantiate_random_model_from_config,
)
from representation_reliability.data.base import samples_to_dataframe
from representation_reliability.data.splits import (
    ConfirmationSplitAccessError,
    apply_splits,
    assign_group_splits,
    build_discovery_label_map,
    discovery_view,
)
from representation_reliability.data.synthetic import (
    RELATION_FAMILIES,
    generate_synthetic_relations,
)
from representation_reliability.extraction.activations import build_cache_identity
from representation_reliability.probes.linear import fit_probe, raw_probe_direction
from representation_reliability.runners import e00c


@pytest.fixture(scope="module")
def split_samples_and_frame():
    samples = generate_synthetic_relations(n_samples=200, seed=29)
    frame = samples_to_dataframe(samples)
    assignment = assign_group_splits(frame["pair_id"].tolist(), seed=31)
    frame = apply_splits(frame, assignment)
    return samples, frame


def test_e00c_runner_defaults_to_its_own_experiment_config(monkeypatch):
    captured = {}

    def stop_after_config_resolution(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after inspecting paths")

    monkeypatch.setattr(e00c, "resolve_config", stop_after_config_resolution)
    with pytest.raises(RuntimeError, match="stop after inspecting paths"):
        e00c.run_e00c()
    assert captured["experiment_path"].name == "E00C_readout_diagnosis.yaml"


@pytest.mark.parametrize("family", sorted(RELATION_FAMILIES))
def test_lofo_train_and_validation_exclude_held_family(
    split_samples_and_frame, family
):
    _samples, full = split_samples_and_frame
    discovery = discovery_view(full)
    plan = e00c.lofo_family_split(discovery, family)
    relation_of = dict(zip(discovery["sample_id"], discovery["relation"]))
    assert all(relation_of[sid] != family for sid in plan["train_ids"])
    assert all(relation_of[sid] != family for sid in plan["validation_ids"])


@pytest.mark.parametrize("family", sorted(RELATION_FAMILIES))
def test_lofo_evaluation_contains_only_held_family(
    split_samples_and_frame, family
):
    _samples, full = split_samples_and_frame
    discovery = discovery_view(full)
    plan = e00c.lofo_family_split(discovery, family)
    rows = discovery.set_index("sample_id").loc[plan["evaluation_ids"]]
    assert set(rows["relation"]) == {family}
    assert set(rows["split"]) == {"discovery_test"}


def test_threshold_api_can_only_read_validation_arrays():
    assert list(inspect.signature(e00c.select_threshold).parameters) == [
        "y_val", "margins_val"
    ]
    y_val = np.array([0, 0, 1, 1])
    margins_val = np.array([-2.0, -0.5, 0.25, 3.0])
    tau = e00c.select_threshold(y_val, margins_val)
    impossible_discovery_labels = np.array([7, -5, 7, -5])
    impossible_discovery_labels[:] = impossible_discovery_labels[::-1]
    assert e00c.select_threshold(y_val, margins_val) == tau


def test_paired_origin_bootstrap_resamples_labels_with_scores():
    rng = np.random.default_rng(44)
    y = np.tile([0, 1], 200)
    pretrained = y.astype(float) + rng.normal(0, 0.02, size=len(y))
    random_scores = [rng.normal(size=len(y)) for _ in range(3)]
    result = e00c.paired_bootstrap_delta(
        pretrained, random_scores, y, n_bootstraps=400, seed=9)
    assert result["delta_D"] > 0.4
    assert result["ci_low"] > 0.35
    assert result["ci_low"] <= result["delta_D"] <= result["ci_high"]


class _ChatTokenizer:
    chat_template = "test-template"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking=False
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return f"<chat>{messages[0]['content']}<assistant>"


class _ChatAdapter:
    tokenizer = _ChatTokenizer()


def test_chat_arm_preserves_semantic_ids_order_and_pairing():
    raw = generate_synthetic_relations(n_samples=20, seed=5)
    chat = e00c.build_chat_samples(_ChatAdapter(), raw)
    assert [s.sample_id for s in chat] == [s.sample_id for s in raw]
    assert [s.pair_id for s in chat] == [s.pair_id for s in raw]
    assert [s.target_label for s in chat] == [s.target_label for s in raw]
    assert all(s.metadata["chat_template_used"] for s in chat)
    assert all(a.prompt != b.prompt for a, b in zip(raw, chat))


def test_raw_and_chat_cache_identities_are_distinct():
    raw = generate_synthetic_relations(n_samples=20, seed=7)
    chat = e00c.build_chat_samples(_ChatAdapter(), raw)
    common = {
        "model_id": "Qwen/Qwen3-0.6B",
        "model_resolved_revision": "a" * 40,
        "tokenizer_id": "Qwen/Qwen3-0.6B",
        "tokenizer_resolved_revision": "a" * 40,
        "sites": ["resid_post"],
        "layers": [0],
        "token_selectors": ["last_prompt"],
        "model_dtype": "bfloat16",
    }
    raw_id = build_cache_identity(
        experiment_id="E00C", samples=raw,
        tokenization={"chat_template": False}, **common)
    chat_id = build_cache_identity(
        experiment_id="E00C_chat", samples=chat,
        tokenization={"chat_template": True, "thinking_enabled": False},
        **common)
    assert raw_id["dataset_content_hash"] != chat_id["dataset_content_hash"]
    assert raw_id != chat_id


def test_raw_probe_direction_recovers_unstandardized_scores():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5)) * np.array([1.0, 3.0, 0.2, 8.0, 2.0])
    y = (X[:, 0] - 0.3 * X[:, 1] > 0).astype(int)
    fit = fit_probe(X[:80], y[:80], X[80:100], y[80:100], c_grid=[0.1, 1.0])
    raw_w = raw_probe_direction(fit)
    expected = fit["classifier"].coef_[0] / fit["scaler_scale"]
    np.testing.assert_allclose(raw_w, expected)
    raw_intercept = float(fit["classifier"].intercept_[0]
                          - np.dot(raw_w, fit["scaler_mean"]))
    direct = X[100:] @ raw_w + raw_intercept
    standardized = fit["classifier"].decision_function(
        (X[100:] - fit["scaler_mean"]) / fit["scaler_scale"])
    np.testing.assert_allclose(direct, standardized, atol=1e-10)


def test_error_subset_analysis_consumes_frozen_scores_only(monkeypatch):
    monkeypatch.setattr(
        e00c, "fit_probe",
        lambda *args, **kwargs: pytest.fail("subset analysis retrained a probe"),
    )
    block = {
        ("resid_post", "last_prompt", 4): {
            "sample_ids": ["a", "b", "c", "d"],
            "y": np.array([0, 1, 0, 1]),
            "scores": np.array([-2.0, 2.0, -1.0, 1.0]),
        }
    }
    behavior = {
        "a": {"correct": True}, "b": {"correct": False},
        "c": {"correct": False}, "d": {"correct": True},
    }
    out = e00c.frozen_probe_subset_metrics(block, behavior)
    assert set(out["subset"]) == {"errors", "correct"}
    assert set(out["n"]) == {2}


def test_confirmation_guard_remains_active(split_samples_and_frame):
    _samples, full = split_samples_and_frame
    with pytest.raises(ConfirmationSplitAccessError):
        build_discovery_label_map(full)


def _small_qwen_config():
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8,
    )


def _parameter_vector(model):
    import torch

    return torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])


def test_random_init_is_reproducible_by_seed():
    from transformers import AutoModelForCausalLM

    first = instantiate_random_model_from_config(
        AutoModelForCausalLM, _small_qwen_config(), seed=17)
    second = instantiate_random_model_from_config(
        AutoModelForCausalLM, _small_qwen_config(), seed=17)
    np.testing.assert_array_equal(_parameter_vector(first), _parameter_vector(second))


def test_random_init_parameters_change_with_seed():
    from transformers import AutoModelForCausalLM

    random_arm = instantiate_random_model_from_config(
        AutoModelForCausalLM, _small_qwen_config(), seed=17)
    checkpoint_reference = instantiate_random_model_from_config(
        AutoModelForCausalLM, _small_qwen_config(), seed=999)
    assert not np.array_equal(
        _parameter_vector(random_arm), _parameter_vector(checkpoint_reference))
