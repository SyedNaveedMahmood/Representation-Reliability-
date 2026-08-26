"""Deterministic run identity.

Run identity inputs (per docs/IMPLEMENTATION_SPEC.md):
experiment ID; resolved config hash; seed; model revision; dataset split hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def compute_run_input_hash(
    experiment_id: str,
    config_hash: str,
    seed: int,
    model_revision: str | None,
    dataset_split_hash: str | None,
) -> str:
    payload = json.dumps(
        {
            "experiment_id": experiment_id,
            "config_hash": config_hash,
            "seed": seed,
            "model_revision": model_revision,
            "dataset_split_hash": dataset_split_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_run_id(
    experiment_id: str,
    config_hash: str,
    seed: int,
    model_revision: str | None = None,
    dataset_split_hash: str | None = None,
) -> str:
    """Deterministic run ID: identical inputs -> identical run ID."""
    h = compute_run_input_hash(
        experiment_id, config_hash, seed, model_revision, dataset_split_hash
    )
    return f"{experiment_id}_{h[:12]}"


def allocate_run_dir(output_root: str | Path, experiment_id: str, run_id: str) -> Path:
    """Allocate ``<output_root>/<experiment_id>/<run_id>`` without clobbering.

    If the deterministic run directory already exists, a new ``-r<n>`` suffix is
    appended (r2, r3, ...) so completed runs are never overwritten silently.
    """
    base = Path(output_root) / experiment_id / run_id
    candidate = base
    n = 1
    while candidate.exists():
        n += 1
        candidate = base.parent / f"{run_id}-r{n}"
    return candidate
