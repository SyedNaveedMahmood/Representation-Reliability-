import pandas as pd
import pytest
import torch

from representation_reliability.config import resolve_config
from representation_reliability.runners.e13 import (
    HiddenStateProjector,
    build_e13_open_corpus,
    distillation_loss,
    factorial_evidence_is_finite,
    learning_rate_for_step,
    representation_kd_loss,
    rms_normalize_hidden,
    teacher_gap_closure,
)


def test_e13_open_corpus_is_pair_complete_and_disjoint():
    samples, frame, stats = build_e13_open_corpus(
        (("train", 40, 11), ("validation", 20, 12), ("discovery_test", 20, 13))
    )
    assert len(samples) == 80
    assert frame["prompt"].is_unique
    assert frame.groupby("pair_id")["sample_id"].size().eq(2).all()
    assert frame.groupby("pair_id")["split"].nunique().eq(1).all()
    assert stats["confirmation_accessed"] is False


def test_distillation_loss_r1_and_r2_are_finite_and_differentiable():
    student = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    teacher = torch.tensor([[0.5, 1.5, -0.5]])
    target = torch.tensor([1])
    r1, _ = distillation_loss(student, target, teacher_logits=None, regime="R1")
    r2, parts = distillation_loss(student, target, teacher_logits=teacher, regime="R2")
    assert torch.isfinite(r1) and torch.isfinite(r2)
    assert parts["kd"] > 0
    r2.backward()
    assert torch.isfinite(student.grad).all()


def test_r3_hidden_matching_is_normalized_and_differentiable():
    student_hidden = torch.tensor([[3.0, 4.0]], requires_grad=True)
    teacher_hidden = torch.tensor([[1.0, -1.0, 0.5]])
    projector = HiddenStateProjector(2, 3)
    hidden = representation_kd_loss(student_hidden, teacher_hidden, projector)
    student_logits = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    teacher_logits = torch.tensor([[0.5, 1.5, -0.5]])
    loss, parts = distillation_loss(
        student_logits,
        torch.tensor([1]),
        teacher_logits=teacher_logits,
        regime="R3",
        hidden_loss=hidden,
    )
    loss.backward()
    assert parts["hidden"] > 0
    assert torch.isfinite(student_hidden.grad).all()
    normalized = rms_normalize_hidden(torch.tensor([[3.0, 4.0]]))
    assert normalized.float().pow(2).mean().item() == pytest.approx(1.0)


def test_frozen_training_schedule_and_gap_guard():
    assert learning_rate_for_step(1) == pytest.approx(2e-6)
    assert learning_rate_for_step(10) == pytest.approx(2e-5)
    assert learning_rate_for_step(100) == pytest.approx(0.0)
    assert teacher_gap_closure(2.0, 1.0, 3.0) == pytest.approx(0.5)
    assert teacher_gap_closure(2.0, 1.0, 1.0 + 1e-10) is None
    with pytest.raises(ValueError):
        learning_rate_for_step(0)


def test_e13_frozen_config_resolves():
    cfg, _ = resolve_config(
        base_path="configs/base.yaml",
        model_path="configs/models/qwen3_0.6b.yaml",
        experiment_path="configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    assert cfg.experiment.id == "E13"
    assert cfg.experiment.mode == "discovery"


def test_nullable_direction_seed_is_not_treated_as_nonfinite_evidence():
    rows = pd.DataFrame(
        {
            "direction_seed": [None, 2130],
            **{
                name: [0.0, 1.0]
                for name in ("Y00", "Y10", "Y01", "Y11", "Q0", "A", "Q_context", "G")
            },
        }
    )
    assert factorial_evidence_is_finite(rows)
    rows.loc[0, "G"] = float("nan")
    assert not factorial_evidence_is_finite(rows)
