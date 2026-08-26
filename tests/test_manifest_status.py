import json

import numpy as np
import pytest

from representation_reliability.runtime.manifest import (
    RunManifest,
    dataset_split_hash,
    environment_manifest,
)
from representation_reliability.runtime.status import (
    StatusFile,
    atomic_write_json,
)


def test_status_lifecycle(tmp_path):
    sf = StatusFile.create(tmp_path, "run-1", "E00")
    assert sf.state_name == "running"
    sf.update(progress={"shards_done": 2})
    sf.complete()
    reloaded = StatusFile.load(tmp_path)
    assert reloaded.is_complete()
    assert reloaded.state["progress"]["shards_done"] == 2
    assert any(h["state"] == "running" for h in reloaded.state["history"])


def test_invalid_status_state_rejected(tmp_path):
    sf = StatusFile.create(tmp_path, "run-2", "E00")
    with pytest.raises(ValueError):
        sf.update(state="sort_of_done")


def test_atomic_write_is_valid_json(tmp_path):
    p = tmp_path / "nested" / "x.json"
    atomic_write_json(p, {"a": 1, "b": [1, 2]})
    with p.open() as fh:
        assert json.load(fh)["a"] == 1
    leftovers = list(p.parent.glob("*.tmp"))
    assert not leftovers


def test_manifest_environment_records_versions_or_nulls(tmp_path):
    env = environment_manifest()
    assert "torch" in env["package_versions"]
    for key in ("project_git", "external_repos", "gpu"):
        assert key in env

    m = RunManifest(tmp_path)
    m.set_start("cfg-hash", {"sources": {}}, {"seed": 1})
    m.finish()
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["start_time"] and payload["finish_time"]
    assert payload["wall_time_s"] is not None


def test_dataset_split_hash_deterministic():
    a = dataset_split_hash({"s1": "train", "s2": "confirmation"})
    b = dataset_split_hash({"s2": "confirmation", "s1": "train"})
    c = dataset_split_hash({"s1": "validation", "s2": "confirmation"})
    assert a == b and a != c
