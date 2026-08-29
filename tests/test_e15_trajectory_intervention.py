"""Contract tests for the E15 trajectory intervention primitives.

The dummy stack mixes positions causally (a running prefix sum), so an edit at an
earlier position genuinely reaches a later readout position at later layers --
exactly the propagation path E15 measures.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from representation_reliability.adapters.intervention import (
    forward_multi_capture,
    forward_resid_post_edit,
    resid_post_hook_count,
)


class CausalMixLayer(torch.nn.Module):
    """h[t] <- h[t] + bias + mean of all h[<=t] (causal, position mixing)."""

    def __init__(self, bias: float) -> None:
        super().__init__()
        self.bias = float(bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        prefix = torch.cumsum(hidden, dim=1)
        counts = torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        return hidden + self.bias + prefix / counts


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.layers = torch.nn.ModuleList(
            [CausalMixLayer(1.0), CausalMixLayer(0.5), CausalMixLayer(0.25)]
        )
        self.head = torch.nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                )
            )

    def forward(self, input_ids, attention_mask=None):
        h = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.head(h))


class DummyAdapter:
    def __init__(self, length: int = 5) -> None:
        self.model = DummyModel()
        self.tokenizer = object()
        self.hidden_size = 3
        self.num_layers = 3
        self.device = torch.device("cpu")
        self.length = int(length)

    def tokenize(self, prompts):
        n = len(prompts)
        ids = torch.tensor([list(range(self.length)) for _ in range(n)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def _raw_layer(self, layer):
        return self.model.layers[layer]

    def resolve_site(self, site, layer):  # pragma: no cover - unused here
        raise NotImplementedError


def _clean(adapter, **kwargs):
    return forward_multi_capture(
        adapter,
        ["a", "b"],
        readout_token_indices=[4, 4],
        output_token_ids=[0, 3],
        **kwargs,
    )


def test_edit_at_an_earlier_token_changes_a_later_decision_readout():
    adapter = DummyAdapter()
    clean = _clean(adapter)
    edited = forward_resid_post_edit(
        adapter,
        ["a", "b"],
        edit_layer=0,
        edit_token_indices=[1, 1],
        deltas=np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        readout_token_indices=[4, 4],
        output_token_ids=[0, 3],
        capture_layers=[1, 2],
    )
    assert not np.allclose(edited["selected_logits"], clean["selected_logits"])
    # The perturbation must be visible downstream at the readout position.
    assert edited["captured"][1].shape == (2, 3)
    assert edited["captured"][2].shape == (2, 3)


def test_zero_edit_reproduces_the_clean_forward_and_leaves_no_hooks():
    adapter = DummyAdapter()
    clean = _clean(adapter)
    zero = forward_resid_post_edit(
        adapter,
        ["a", "b"],
        edit_layer=0,
        edit_token_indices=[1, 3],
        deltas=np.zeros((2, 3)),
        readout_token_indices=[4, 4],
        output_token_ids=[0, 3],
        capture_layers=[1],
    )
    np.testing.assert_allclose(zero["selected_logits"], clean["selected_logits"], atol=1e-6)
    for layer in range(adapter.num_layers):
        assert resid_post_hook_count(adapter, layer=layer) == 0


def test_edited_carrier_state_equals_base_plus_delta():
    adapter = DummyAdapter()
    clean = _clean(adapter, capture_specs=[("carrier", 0, [2, 2])])
    delta = np.array([[1.5, -2.0, 0.25], [0.0, 0.0, 4.0]])
    edited = forward_resid_post_edit(
        adapter,
        ["a", "b"],
        edit_layer=0,
        edit_token_indices=[2, 2],
        deltas=delta,
        readout_token_indices=[4, 4],
        output_token_ids=[0, 3],
    )
    np.testing.assert_allclose(
        edited["edited_carrier_state"], clean["captured"]["carrier"] + delta, atol=1e-6
    )


def test_capture_at_or_before_the_edit_layer_is_rejected():
    adapter = DummyAdapter()
    for layer in (0, 3):
        with pytest.raises(ValueError):
            forward_resid_post_edit(
                adapter,
                ["a"],
                edit_layer=0,
                edit_token_indices=[1],
                deltas=np.zeros((1, 3)),
                readout_token_indices=[4],
                output_token_ids=[0],
                capture_layers=[layer],
            )


def test_multi_capture_reads_several_named_sites_in_one_forward():
    adapter = DummyAdapter()
    out = forward_multi_capture(
        adapter,
        ["a", "b"],
        readout_token_indices=[4, 4],
        output_token_ids=[0, 3],
        capture_specs=[
            ("carrier", 0, [1, 1]),
            ("irrelevant_carrier", 0, [2, 2]),
            ("decision_l0", 0, [4, 4]),
            ("decision_l2", 2, [4, 4]),
        ],
    )
    assert set(out["captured"]) == {
        "carrier", "irrelevant_carrier", "decision_l0", "decision_l2",
    }
    assert not np.allclose(out["captured"]["carrier"], out["captured"]["decision_l0"])
    with pytest.raises(ValueError, match="unique"):
        forward_multi_capture(
            adapter,
            ["a"],
            readout_token_indices=[4],
            output_token_ids=[0],
            capture_specs=[("x", 0, [1]), ("x", 1, [1])],
        )


def test_positions_outside_the_prompt_are_rejected():
    adapter = DummyAdapter()

    def padded_tokenize(prompts):
        ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    adapter.tokenize = padded_tokenize
    with pytest.raises(ValueError, match="outside the prompt length"):
        forward_resid_post_edit(
            adapter,
            ["short", "long"],
            edit_layer=0,
            edit_token_indices=[1, 1],
            deltas=np.zeros((2, 3)),
            readout_token_indices=[4, 4],
            output_token_ids=[0],
        )
