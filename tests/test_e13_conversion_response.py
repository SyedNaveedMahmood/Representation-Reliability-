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
from representation_reliability.metrics.quantization import factorial_components
from representation_reliability.runners.e13_methods import (
    METHOD_PROTOCOL_SHA256,
    ModelLocalReference,
    _deterministic_live_batch_ids,
    _model_local_geometry,
    _orientation,
    _random_pair,
    _sha256_file,
    _validate_cache_artifacts,
)


def test_target_orientation_matches_yes_minus_no_margin():
    labels = {"yes": 1, "no": 0}
    assert _orientation(labels, ["yes", "no"]).tolist() == [1.0, -1.0]


def _local_reference(role: str, hidden_size: int) -> ModelLocalReference:
    direction = np.zeros(hidden_size)
    direction[0] = 1.0
    return ModelLocalReference.from_reference(
        model_role=role,
        hidden_size=hidden_size,
        reference={
            "direction": direction,
            "probe": {
                "coef": direction,
                "intercept": np.zeros(1),
                "mean": np.zeros(hidden_size),
                "scale": np.ones(hidden_size),
            },
            "targets": {
                "q0_star": -1.0,
                "q1_star": 1.0,
                "sigma_q_validation": 0.5,
                "sigma_margin_validation": 2.0,
            },
        },
        model_revisions={"model_sha": f"{role}-sha"},
    )


def test_teacher_and_student_semantic_edits_enforce_model_local_space():
    teacher = _local_reference("teacher", 2048)
    student = _local_reference("student", 1024)
    assert teacher.semantic_delta(np.zeros(2048), 1.0).shape == (2048,)
    assert student.semantic_delta(np.zeros(1024), 1.0).shape == (1024,)
    with pytest.raises(ValueError, match="activation_size=2048"):
        student.semantic_delta(np.zeros(2048), 1.0)
    with pytest.raises(ValueError, match="activation_size=1024"):
        teacher.semantic_delta(np.zeros(1024), 1.0)


@pytest.mark.parametrize(("role", "hidden_size"), [("teacher", 2048), ("student", 1024)])
def test_factorial_and_random_geometry_stays_in_model_space(role, hidden_size):
    reference = _local_reference(role, hidden_size)
    base = np.linspace(-0.2, 0.2, hidden_size)
    source = base.copy()
    source[1:] += np.linspace(0.1, 0.3, hidden_size - 1)
    kwargs = {
        "reference": reference,
        "expected_role": role,
        "runtime_hidden_size": hidden_size,
        "runtime_revisions": {"model_sha": f"{role}-sha"},
        "ids": ["sample"],
        "labels": {"sample": 0},
        "base_by_id": {"sample": base},
        "source_by_id": {"sample": source},
    }
    geometry = _model_local_geometry(**kwargs)
    repeat = _model_local_geometry(**kwargs)
    semantic = geometry["semantic"]["sample"]
    context = geometry["context"]["sample"]
    random_q = geometry["random_q"]["sample"]
    random_c = geometry["random_c"]["sample"]
    assert all(
        vector.shape == (hidden_size,)
        for vectors in geometry.values()
        for vector in vectors.values()
    )
    assert np.dot(context, reference.direction) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(random_q, reference.direction) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(random_c, reference.direction) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(random_c, random_q) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(random_q) == pytest.approx(np.linalg.norm(semantic))
    assert np.linalg.norm(random_c) == pytest.approx(np.linalg.norm(context))
    assert np.allclose(random_q, repeat["random_q"]["sample"])
    assert np.allclose(random_c, repeat["random_c"]["sample"])
    readout = np.linspace(-0.5, 0.5, hidden_size)
    y00 = base @ readout
    y10 = (base + semantic) @ readout
    y01 = (base + context) @ readout
    y11 = (base + semantic + context) @ readout
    components = factorial_components(y00, y10, y01, y11)
    assert set(components) == {"Q0", "A", "Q_context", "G"}
    assert np.isfinite(list(components.values())).all()


def test_teacher_cache_geometry_rejects_student_reference_before_edit():
    student = _local_reference("student", 1024)
    with pytest.raises(ValueError, match="expected_role='teacher'"):
        _model_local_geometry(
            reference=student,
            expected_role="teacher",
            runtime_hidden_size=2048,
            runtime_revisions={"model_sha": "teacher-sha"},
            ids=["sample"],
            labels={"sample": 0},
            base_by_id={"sample": np.zeros(2048)},
            source_by_id={"sample": np.ones(2048)},
        )


def test_live_validation_preserves_original_bf16_batch_identity():
    ordered = [f"sample-{index:03d}" for index in range(32)]
    chosen = _deterministic_live_batch_ids(ordered, batch_size=8, n_rows=16)
    assert len(chosen) == 16
    for start in range(0, len(chosen), 8):
        block = chosen[start : start + 8]
        original_start = ordered.index(block[0])
        assert original_start % 8 == 0
        assert block == ordered[original_start : original_start + 8]
    assert chosen == _deterministic_live_batch_ids(ordered, batch_size=8, n_rows=16)


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
        "model_role": "teacher",
        "teacher_hidden_size": 2048,
        "semantic_direction_size": 2048,
        "component_scale_split": "validation",
        "response_table_sha256": _sha256_file(table),
        "response_tensor_sha256": hashlib.sha256(numeric.tobytes()).hexdigest(),
        "ordered_id_sha256": hashlib.sha256(b"a").hexdigest(),
        "response_columns": columns,
        "response_family_columns": {"R4": columns},
        "response_family_sha256": {"R4": hashlib.sha256(numeric.tobytes()).hexdigest()},
        "n_rows": 1,
        "live_check": {
            "passed": True,
            "by_response": {name: {"passed": True} for name in columns},
        },
    }
    (tmp_path / "cache_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert _validate_cache_artifacts(tmp_path)["n_rows"] == 1
    metadata["component_scale_split"] = "discovery_test"
    (tmp_path / "cache_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="scale provenance"):
        _validate_cache_artifacts(tmp_path)
