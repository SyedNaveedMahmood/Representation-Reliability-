"""Run status tracking with atomic writes.

States: running | complete | failed | partial | partial_budget_stop.
A partially written status file must never be readable as complete.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

STATUS_STATES = (
    "running",
    "complete",
    "failed",
    "partial",
    "partial_budget_stop",
)


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Write JSON atomically: tmp file in same dir, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class StatusFile:
    """status.json tracking the lifecycle of a single run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "status.json"
        self.state: dict[str, Any] = {}

    @classmethod
    def create(cls, run_dir: str | Path, run_id: str, experiment_id: str) -> StatusFile:
        sf = cls(run_dir)
        sf.state = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "state": "running",
            "started_at": _utcnow(),
            "updated_at": _utcnow(),
            "finished_at": None,
            "message": None,
            "progress": {},
            "history": [],
        }
        sf.write()
        return sf

    @classmethod
    def load(cls, run_dir: str | Path) -> StatusFile | None:
        p = Path(run_dir) / "status.json"
        if not p.exists():
            return None
        sf = cls(run_dir)
        with p.open("r", encoding="utf-8") as fh:
            sf.state = json.load(fh)
        if sf.state.get("state") not in STATUS_STATES:
            raise ValueError(f"unknown state {sf.state.get('state')!r} in {p}")
        return sf

    # -- predicates -------------------------------------------------------
    @property
    def state_name(self) -> str:
        return self.state.get("state", "running")

    def is_complete(self) -> bool:
        return self.state_name == "complete"

    # -- mutation ---------------------------------------------------------
    def update(self, *, state: str | None = None, message: str | None = None,
               progress: dict | None = None) -> None:
        if state is not None:
            if state not in STATUS_STATES:
                raise ValueError(f"invalid status state {state!r}")
            self.state["state"] = state
        if message is not None:
            self.state["message"] = message
        if progress is not None:
            merged = dict(self.state.get("progress") or {})
            merged.update(progress)
            self.state["progress"] = merged
        self._record()

    def fail(self, message: str) -> None:
        self.update(state="failed", message=message)

    def mark_partial(self, message: str, budget_stop: bool = False) -> None:
        self.update(
            state="partial_budget_stop" if budget_stop else "partial",
            message=message,
        )

    def complete(self, message: str = "run complete") -> None:
        self.state["state"] = "complete"
        self.state["message"] = message
        self.state["finished_at"] = _utcnow()
        self._record()

    def _record(self) -> None:
        self.state["updated_at"] = _utcnow()
        history = self.state.setdefault("history", [])
        entry = {"at": self.state["updated_at"], "state": self.state["state"]}
        if self.state.get("message"):
            entry["message"] = self.state["message"]
        history.append(entry)
        self.write()

    def write(self) -> None:
        atomic_write_json(self.path, self.state)
