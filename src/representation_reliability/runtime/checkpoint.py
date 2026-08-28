"""Identity-locked atomic checkpoint helpers for resumable long jobs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .status import atomic_write_json

COMPLETE_MARKER = "checkpoint.complete.json"


def begin_atomic_checkpoint(root: str | Path, step: int) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    final = root_path / f"step_{int(step):03d}"
    if final.exists():
        raise FileExistsError(f"checkpoint target already exists: {final}")
    return Path(tempfile.mkdtemp(prefix=f".step_{int(step):03d}.", dir=str(root_path)))


def commit_atomic_checkpoint(
    temporary: str | Path,
    root: str | Path,
    *,
    step: int,
    identity: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    temporary_path = Path(temporary)
    root_path = Path(root)
    if temporary_path.parent.resolve() != root_path.resolve():
        raise ValueError("temporary checkpoint is outside its declared root")
    final = root_path / f"step_{int(step):03d}"
    marker = {
        "step": int(step),
        "identity": str(identity),
        "complete": True,
        "metadata": metadata or {},
    }
    atomic_write_json(temporary_path / COMPLETE_MARKER, marker)
    os.replace(temporary_path, final)
    return final


def checkpoint_marker(path: str | Path) -> dict[str, Any] | None:
    marker_path = Path(path) / COMPLETE_MARKER
    if not marker_path.exists():
        return None
    with marker_path.open("r", encoding="utf-8") as handle:
        marker = json.load(handle)
    if marker.get("complete") is not True:
        return None
    return marker


def latest_complete_checkpoint(
    root: str | Path, *, identity: str
) -> tuple[Path, dict[str, Any]] | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    for path in root_path.glob("step_[0-9][0-9][0-9]"):
        marker = checkpoint_marker(path)
        if marker is None:
            continue
        if str(marker.get("identity")) != str(identity):
            raise RuntimeError(f"checkpoint identity mismatch: {path}")
        if int(marker.get("step", -1)) != int(path.name.removeprefix("step_")):
            raise RuntimeError(f"checkpoint step marker mismatch: {path}")
        complete.append((int(marker["step"]), path, marker))
    if not complete:
        return None
    _step, path, marker = max(complete, key=lambda item: item[0])
    return path, marker
