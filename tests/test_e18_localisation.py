"""Contracts for E18 site resolution, span editing and the frozen grade scale."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from representation_reliability.adapters.intervention import (
    forward_multi_capture,
    forward_resid_post_span_edit,
    resid_post_hook_count,
)
from representation_reliability.data.stateful_console import (
    DENIED,
    GRANTED,
    generate_console_samples,
)
from representation_reliability.runners.e18 import (
    LAYERS,
    SITES,
    SPAN_SITES,
    grade_cell,
    prompt_char_spans,
    resolve_site_tokens,
)

from test_e15_trajectory_intervention import DummyAdapter


def _samples(n: int = 6):
    return generate_console_samples(n, 20261801, (1, 8), namespace="e18-test-v1")


# ------------------------------------------------------------------ char spans
def test_char_spans_land_on_the_intended_text():
    for sample in _samples():
        prompt = sample.prompt
        spans = prompt_char_spans(sample)
        assert set(spans) == set(SITES)
        state = sample.metadata["target_state"]
        s, e = spans["state_word_last"]
        assert prompt[s:e] == state
        s, e = spans["carrier"]
        assert prompt[s:e] == sample.metadata["target_clearance_line"]
        s, e = spans["request_step_last"]
        assert prompt[s:e].startswith("Step") and "requests to run a transfer" in prompt[s:e]
        s, e = spans["decision"]
        assert prompt[s:e] == ":"
        s, e = spans["prefix_span"]
        assert prompt[s:e].startswith("Console log.")
        assert prompt[s:e].endswith(".")
        assert "requests to run a transfer" not in prompt[s:e]


def test_state_word_span_is_not_the_header_occurrence():
    """The header contains the literal GRANTED; the span must skip it."""
    for sample in _samples():
        prompt = sample.prompt
        header_end = prompt.index("\n")
        start, _end = prompt_char_spans(sample)["state_word_last"]
        assert start > header_end
        if sample.metadata["target_state"] == GRANTED:
            assert prompt.index(GRANTED) < header_end < start


def test_prefix_span_contains_both_clearance_writes_and_no_distractor_gap():
    for sample in _samples():
        start, end = prompt_char_spans(sample)["prefix_span"]
        text = sample.prompt[start:end]
        assert text.count("sets the clearance for terminal") == 2
        assert sample.metadata["target_clearance_line"] in text
        assert sample.metadata["irrelevant_clearance_line"] in text
        assert sample.metadata["late_gap_line"] not in text


def test_char_spans_reject_a_malformed_prompt():
    sample = _samples(2)[0]
    broken = SimpleNamespace(
        sample_id="x", prompt="one\ntwo\nthree", metadata=dict(sample.metadata)
    )
    with pytest.raises(RuntimeError, match="unexpected prompt shape"):
        prompt_char_spans(broken)


# ------------------------------------------------------------ token resolution
class _Tok:
    """Whitespace/character tokenizer good enough for offset-based resolution."""

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        offsets, ids, cursor = [], [], 0
        for piece in text.split(" "):
            if piece:
                offsets.append((cursor, cursor + len(piece)))
                ids.append(len(ids))
            cursor += len(piece) + 1
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_site_token_resolution_is_single_or_span_as_declared():
    tok = _Tok()
    for sample in _samples(4):
        resolved = resolve_site_tokens(tok, sample)
        for name in SITES:
            assert len(resolved[name]) >= 1
            if name in SPAN_SITES:
                assert len(resolved[name]) > 1
            else:
                assert len(resolved[name]) == 1
        assert resolved["decision"] == resolved["_readout"]
        # A span must strictly contain its single-token counterpart.
        assert resolved["carrier"][0] in resolved["clearance_line_span"]
        assert resolved["state_word_last"][0] in resolved["clearance_line_span"]
        assert resolved["clearance_line_span"][-1] == resolved["carrier"][0]


def test_prefix_span_tokens_precede_the_decision_token():
    tok = _Tok()
    for sample in _samples(4):
        resolved = resolve_site_tokens(tok, sample)
        assert max(resolved["prefix_span"]) < resolved["decision"][0]
        assert len(resolved["prefix_span"]) > len(resolved["clearance_line_span"])


def test_declared_layer_sweep_is_unique_and_includes_the_frozen_qwen3_site():
    assert list(LAYERS) == sorted(LAYERS)
    assert len(set(LAYERS)) == len(LAYERS)
    assert 17 in LAYERS


# ---------------------------------------------------------------- span editing
def _adapter():
    return DummyAdapter(length=6)


def test_span_edit_applies_a_per_position_delta_block():
    adapter = _adapter()
    clean = forward_multi_capture(
        adapter, ["a"], readout_token_indices=[5], output_token_ids=[0, 3],
        capture_specs=[("p1", 0, [1]), ("p2", 0, [2])],
    )
    delta = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    out = forward_resid_post_span_edit(
        adapter, ["a"], edit_layer=0, edit_token_indices=[[1, 2]],
        deltas=[delta[0]], readout_token_indices=[5], output_token_ids=[0, 3],
    )
    np.testing.assert_allclose(
        out["edited_states"][0][0], clean["captured"]["p1"][0] + delta[0][0], atol=1e-6
    )
    np.testing.assert_allclose(
        out["edited_states"][0][1], clean["captured"]["p2"][0] + delta[0][1], atol=1e-6
    )


def test_span_edit_handles_ragged_spans_and_leaves_no_hooks():
    adapter = _adapter()
    out = forward_resid_post_span_edit(
        adapter, ["a", "b"], edit_layer=0,
        edit_token_indices=[[1], [1, 2, 3]],
        deltas=[np.ones((1, 3)), np.ones((3, 3)) * 2.0],
        readout_token_indices=[5, 5], output_token_ids=[0, 3],
    )
    assert out["edited_states"][0].shape == (1, 3)
    assert out["edited_states"][1].shape == (3, 3)
    for layer in range(adapter.num_layers):
        assert resid_post_hook_count(adapter, layer=layer) == 0


def test_span_edit_zero_delta_matches_the_clean_forward():
    adapter = _adapter()
    clean = forward_multi_capture(
        adapter, ["a", "b"], readout_token_indices=[5, 5], output_token_ids=[0, 3]
    )
    zero = forward_resid_post_span_edit(
        adapter, ["a", "b"], edit_layer=0, edit_token_indices=[[1, 2], [3, 4]],
        deltas=[np.zeros((2, 3)), np.zeros((2, 3))],
        readout_token_indices=[5, 5], output_token_ids=[0, 3],
    )
    np.testing.assert_allclose(
        zero["selected_logits"], clean["selected_logits"], atol=1e-6
    )


def test_span_edit_rejects_bad_shapes_and_repeated_positions():
    adapter = _adapter()
    with pytest.raises(ValueError, match="n_positions, hidden"):
        forward_resid_post_span_edit(
            adapter, ["a"], edit_layer=0, edit_token_indices=[[1, 2]],
            deltas=[np.ones((3, 3))], readout_token_indices=[5], output_token_ids=[0],
        )
    with pytest.raises(ValueError, match="repeats an edit position"):
        forward_resid_post_span_edit(
            adapter, ["a"], edit_layer=0, edit_token_indices=[[1, 1]],
            deltas=[np.ones((2, 3))], readout_token_indices=[5], output_token_ids=[0],
        )
    with pytest.raises(ValueError, match="outside the prompt length"):
        forward_resid_post_span_edit(
            adapter, ["a"], edit_layer=0, edit_token_indices=[[99]],
            deltas=[np.ones((1, 3))], readout_token_indices=[5], output_token_ids=[0],
        )


# ----------------------------------------------------------------- grade scale
def test_grade_scale_matches_the_frozen_thresholds():
    assert grade_cell(0.80, True, True) == "STRONG"
    assert grade_cell(0.50, True, True) == "STRONG"
    assert grade_cell(0.30, True, True) == "PARTIAL"
    assert grade_cell(0.10, True, True) == "PARTIAL"
    assert grade_cell(0.05, True, True) == "WEAK"
    # E15's failing carrier: flip rate 0.010 grades WEAK however it is scored.
    assert grade_cell(0.010, True, True) == "WEAK"


def test_grade_requires_both_ci_conditions():
    assert grade_cell(0.90, False, True) == "WEAK"
    assert grade_cell(0.90, True, False) == "WEAK"
    assert grade_cell(0.90, False, False) == "WEAK"


def test_state_words_are_the_frozen_pair():
    assert (GRANTED, DENIED) == ("GRANTED", "DENIED")
