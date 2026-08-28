from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from representation_reliability.adapters.intervention import (
    differentiable_resid_post_logits,
    forward_with_resid_post_capture,
)
from representation_reliability.runners.e13_methods import (
    METHOD_PROTOCOL_SHA256,
    _orientation,
    _random_pair,
    _sha256_file,
    _validate_cache_artifacts,
)


def test_target_orientation_matches_yes_minus_no_margin():
    labels = {"yes": 1, "no": 0}
    assert _orientation(labels, ["yes", "no"]).tolist() == [1.0, -1.0]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 5)
        self.layer = nn.Linear(5, 5)
        self.head = nn.Linear(5, 7)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.layer(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


class _TinyAdapter:
    def __init__(self) -> None:
        self.model = _TinyModel()
        self.hidden_size = 5

    def _raw_layer(self, layer):
        assert layer == 0
        return self.model.layer


def test_random_response_pair_is_deterministic_orthogonal_and_norm_matched():
    direction = np.zeros(12)
    direction[0] = 1.0
    first = _random_pair(12, "sample-7", direction, 2.5, 4.0, "model-a")
    repeat = _random_pair(12, "sample-7", direction, 2.5, 4.0, "model-a")
    changed = _random_pair(12, "sample-7", direction, 2.5, 4.0, "model-b")
    assert np.allclose(first[0], repeat[0])
    assert np.allclose(first[1], repeat[1])
    assert not np.allclose(first[0], changed[0])
    assert np.linalg.norm(first[0]) == pytest.approx(2.5)
    assert np.linalg.norm(first[1]) == pytest.approx(4.0)
    assert np.dot(first[0], direction) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(first[1], direction) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(first[0], first[1]) == pytest.approx(0.0, abs=1e-10)


def test_differentiable_intervention_and_capture_remove_hooks():
    adapter = _TinyAdapter()
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention = torch.ones_like(input_ids)
    positions = torch.tensor([1, 1])
    delta = torch.ones(2, 5)
    before = len(adapter.model.layer._forward_hooks)
    logits = differentiable_resid_post_logits(
        adapter,
        input_ids=input_ids,
        attention_mask=attention,
        layer=0,
        token_indices=positions,
        deltas=delta,
        output_token_ids=[1, 2],
    )
    logits.sum().backward()
    assert adapter.model.layer.weight.grad is not None
    assert len(adapter.model.layer._forward_hooks) == before
    adapter.model.zero_grad(set_to_none=True)
    outputs, captured = forward_with_resid_post_capture(
        adapter,
        input_ids=input_ids,
        attention_mask=attention,
        layer=0,
    )
    (outputs.logits.sum() + captured.sum()).backward()
    assert adapter.model.layer.weight.grad is not None
    assert len(adapter.model.layer._forward_hooks) == before


def test_cache_validation_checks_provenance_and_tensor_digest(tmp_path):
    columns = [
        "Y00",
        "r4_neg",
        "r4_pos",
        "R5_Q",
        "R5_A",
        "R5_G",
        "R6_Q",
        "R6_A",
        "R6_G",
    ]
    evidence = pd.DataFrame(
        [{"sample_id": "a", **{name: float(i) for i, name in enumerate(columns)}}]
    )
    table = tmp_path / "teacher_response_rows.parquet"
    evidence.to_parquet(table, index=False)
    numeric = evidence[columns].to_numpy(np.float64)
    metadata = {
        "protocol_sha256": METHOD_PROTOCOL_SHA256,
        "confirmation_accessed": False,
        "component_scale_split": "validation",
        "response_table_sha256": _sha256_file(table),
        "response_tensor_sha256": hashlib.sha256(numeric.tobytes()).hexdigest(),
        "ordered_id_sha256": hashlib.sha256(b"a").hexdigest(),
        "response_columns": columns,
        "n_rows": 1,
        "live_check": {"passed": True},
    }
    (tmp_path / "cache_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert _validate_cache_artifacts(tmp_path)["n_rows"] == 1
    metadata["component_scale_split"] = "discovery_test"
    (tmp_path / "cache_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scale provenance"):
        _validate_cache_artifacts(tmp_path)
