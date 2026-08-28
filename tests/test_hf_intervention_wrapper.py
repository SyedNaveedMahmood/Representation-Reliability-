from types import SimpleNamespace

import numpy as np
import pytest
import torch

from representation_reliability.adapters.intervention import (
    forward_resid_post_intervention,
    forward_selected_token_logits,
)


class DummyLayer(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = float(amount)

    def forward(self, hidden):
        return hidden + self.amount


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.layers = torch.nn.ModuleList([DummyLayer(1), DummyLayer(2), DummyLayer(3)])
        self.head = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ]
                )
            )

    def forward(self, input_ids, attention_mask=None):
        h = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.head(h))


class DummyAdapter:
    def __init__(self):
        self.model = DummyModel()
        self.tokenizer = object()
        self.hidden_size = 3
        self.num_layers = 3
        self.device = torch.device("cpu")

    def tokenize(self, prompts):
        n = len(prompts)
        ids = torch.tensor([[0, 1, 2] for _ in range(n)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def _raw_layer(self, layer):
        return self.model.layers[layer]


def test_resid_post_intervention_applies_exact_delta_and_propagates():
    adapter = DummyAdapter()
    delta = np.array([[2.0, -1.0, 0.5]], dtype=np.float32)
    result = forward_resid_post_intervention(
        adapter,
        ["x"],
        layer=1,
        token_indices=[2],
        deltas=delta,
        output_token_ids=[3, 4],
        capture_layers=[1, 2],
    )

    expected_l1 = np.array([5.0, 2.0, 4.5])
    np.testing.assert_allclose(result["captured"][1][0], expected_l1)
    np.testing.assert_allclose(result["captured"][2][0], expected_l1 + 3.0)

    final = expected_l1 + 3.0
    expected_logits = np.array([final[0] + final[1], final[1] + final[2]])
    np.testing.assert_allclose(result["selected_logits"][0], expected_logits)


def test_zero_edit_matches_unhooked_forward_and_hooks_do_not_leak():
    adapter = DummyAdapter()
    baseline = forward_selected_token_logits(
        adapter,
        ["a", "b"],
        token_indices=[0, 2],
        output_token_ids=[3, 4],
    )
    zero = forward_resid_post_intervention(
        adapter,
        ["a", "b"],
        layer=1,
        token_indices=[0, 2],
        deltas=np.zeros((2, 3), dtype=np.float32),
        output_token_ids=[3, 4],
        capture_layers=[1],
    )
    np.testing.assert_allclose(zero["selected_logits"], baseline)

    forward_resid_post_intervention(
        adapter,
        ["a", "b"],
        layer=1,
        token_indices=[0, 2],
        deltas=np.ones((2, 3), dtype=np.float32),
        output_token_ids=[3, 4],
    )
    after = forward_selected_token_logits(
        adapter,
        ["a", "b"],
        token_indices=[0, 2],
        output_token_ids=[3, 4],
    )
    np.testing.assert_allclose(after, baseline)


def test_sample_specific_indices_reject_a_right_padding_position():
    adapter = DummyAdapter()

    def padded_tokenize(_prompts):
        ids = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    adapter.tokenize = padded_tokenize
    with pytest.raises(ValueError, match="outside prompt length"):
        forward_resid_post_intervention(
            adapter,
            ["short", "long"],
            layer=1,
            token_indices=[2, 2],
            deltas=np.ones((2, 3), dtype=np.float32),
            output_token_ids=[3, 4],
        )

    result = forward_resid_post_intervention(
        adapter,
        ["short", "long"],
        layer=1,
        token_indices=[1, 2],
        deltas=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        output_token_ids=[3, 4],
        capture_layers=[1],
    )
    np.testing.assert_allclose(result["captured"][1][0], [4.0, 4.0, 3.0])
    np.testing.assert_allclose(result["captured"][1][1], [3.0, 5.0, 4.0])
