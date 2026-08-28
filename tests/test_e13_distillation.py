import pytest
import torch

from representation_reliability.config import resolve_config
from representation_reliability.runners.e13 import (
    build_e13_open_corpus,
    distillation_loss,
    learning_rate_for_step,
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
