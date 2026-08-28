import pytest

from representation_reliability.runtime.checkpoint import (
    begin_atomic_checkpoint,
    commit_atomic_checkpoint,
    latest_complete_checkpoint,
)


def test_partial_checkpoint_is_never_resumed_and_complete_is_atomic(tmp_path):
    partial = begin_atomic_checkpoint(tmp_path, 10)
    (partial / "partial.bin").write_bytes(b"partial")
    assert latest_complete_checkpoint(tmp_path, identity="run-a") is None
    complete_temp = begin_atomic_checkpoint(tmp_path, 25)
    (complete_temp / "model.safetensors").write_bytes(b"model")
    final = commit_atomic_checkpoint(
        complete_temp, tmp_path, step=25, identity="run-a", metadata={"seed": 1}
    )
    latest = latest_complete_checkpoint(tmp_path, identity="run-a")
    assert latest is not None
    assert latest[0] == final
    assert latest[1]["metadata"]["seed"] == 1


def test_resume_rejects_seed_or_regime_identity_mismatch(tmp_path):
    temporary = begin_atomic_checkpoint(tmp_path, 10)
    commit_atomic_checkpoint(temporary, tmp_path, step=10, identity="seed-a")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        latest_complete_checkpoint(tmp_path, identity="seed-b")


def test_checkpoint_cannot_overwrite_completed_step(tmp_path):
    temporary = begin_atomic_checkpoint(tmp_path, 10)
    commit_atomic_checkpoint(temporary, tmp_path, step=10, identity="same")
    with pytest.raises(FileExistsError):
        begin_atomic_checkpoint(tmp_path, 10)
