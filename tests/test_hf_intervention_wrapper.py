from types import SimpleNamespace

import numpy as np
import torch

from representation_reliability.adapters.intervention import (
    forward_resid_post_intervention,
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
