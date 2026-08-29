"""Identity locks and holdout materialization for the one-shot E13 diagnostic confirmation.

Nothing in this module may be imported for its side effects: materializing the
``e13_confirmation_v1`` namespace is gated behind :func:`open_e13_access_record`,
which the runner opens exactly once after every frozen identity check passes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.base import samples_to_dataframe
from ..data.synthetic import generate_synthetic_relations
from ..runtime.status import atomic_write_json
from .confirmation_support import sha256_file
from .e13 import CORPUS_SPECS, FAMILIES, _pair_signature, _rename_pair

PROTOCOL_PATH = Path("docs/E13_DIAGNOSTIC_CONFIRMATION_PROTOCOL.md")
CONFIRMATION_VERSION = "e13-diagnostic-confirmation-v1"
BASELINE_CAMPAIGN = "E13MS_04daa7fcc66c"
BASELINE_PROTOCOL_SHA256 = "04daa7fcc66cc1c93f8077de23962dfec9861c9412c44367d83603ed0ccb7cac"

# Frozen non-inferiority and mismatch constants. Both are inherited from the
# already-frozen discovery design; neither is a post-discovery invention.
DELTA_B = 0.03
DELTA_C = 0.10
FAMILY_ALPHA = 0.05
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261304

TRAINING_SEEDS = (20261305, 20261315, 20261325)
CONFIRMED_REGIMES = ("R2", "R3")

HOLDOUT_SPEC: dict[str, Any] = {
    "namespace": "e13_confirmation_v1",
    "generator": "generate_synthetic_relations",
    "generator_seed": 20261304,
    "n_directed": 200,
    "n_pairs": 100,
    "n_entities": 42,
    "families": list(FAMILIES),
    "split_name": "confirmation",
    "sample_id_prefix": "e13-confirmation-v1-sample-",
    "pair_id_prefix": "e13-confirmation-v1-pair-",
    "deduplicated_against": [name for name, _count, _seed in CORPUS_SPECS],
}

HOLDOUT_SPEC_SHA256 = "9b93c5b0d42eb42e451fd265863b7e604f3b8da928f5c8f09b62a1c9cb0d4f57"

# Frozen B-matched checkpoint registry. Every entry was selected by the existing
# validation-only rule during discovery and is reproduced here byte-for-byte.
# No confirmation outcome may alter any field.
CHECKPOINT_REGISTRY: dict[str, dict[str, Any]] = {
    "R2_seed_20261305": {
        "regime": "R2",
        "seed": 20261305,
        "step": 10,
        "run_id": "R2_seed_20261305_6c3082fa1756",
        "run_identity_sha256": (
            "6c3082fa17564aada35894f6bd0cefec00d4ef0c2cbb48baafde9b774dffd588"
        ),
        "weight_file": "model.safetensors",
        "weight_sha256": (
            "c6276176b2dadbb0145abbaaefd8e1ed852be2045c99a55a42fa52eeb4bae188"
        ),
        "validation_B": 0.9474079999999999,
        "absolute_validation_B_gap": 0.026200000000000112,
    },
    "R2_seed_20261315": {
        "regime": "R2",
        "seed": 20261315,
        "step": 10,
        "run_id": "R2_seed_20261315_d6c714d2f2a5",
        "run_identity_sha256": (
            "d6c714d2f2a5e15c60f6609380f352c9df861d02bc8e8141a3ca20c667f6d86d"
        ),
        "weight_file": "model.safetensors",
        "weight_sha256": (
            "37820917b3fbdca3b1cae26d2fefd91a012700cc770e8f275ec8f3cae7479892"
        ),
        "validation_B": 0.95908,
        "absolute_validation_B_gap": 0.014527999999999985,
    },
    "R2_seed_20261325": {
        "regime": "R2",
        "seed": 20261325,
        "step": 10,
        "run_id": "R2_seed_20261325_0c80e8504997",
        "run_identity_sha256": (
            "0c80e8504997dfa4daae31b998bbe8ea0949e7a216f334253159b0a0b35985ec"
        ),
        "weight_file": "model.safetensors",
        "weight_sha256": (
            "5d86f7badce0f264414ba43f6e1bb20611d6c66dae96619ac9930e7a99fa1254"
        ),
        "validation_B": 0.9903040000000001,
        "absolute_validation_B_gap": 0.016696000000000044,
    },
    "R3_seed_20261305": {
        "regime": "R3",
        "seed": 20261305,
        "step": 25,
        "run_id": "R3_seed_20261305_61cec33ea4d5",
        "run_identity_sha256": (
            "61cec33ea4d5a3e141ad388f1195951c596ccd47d728e14748fc1e0c763667bf"
        ),
        "weight_file": "model.safetensors",
        "weight_sha256": (
            "418ce8d4adfe9b2a5cdcd4725c4d6aff8557a198b7fddc338d0b40e0b72926e5"
        ),
        "projector_sha256": (
            "d6666cc717bae65393290b941d2c56c6aeda490322d866062d49970013b41607"
        ),
        "validation_B": 1.0,
        "absolute_validation_B_gap": 0.02639199999999997,
    },
    "R3_seed_20261315": {
        "regime": "R3",
        "seed": 20261315,
        "step": 10,
        "run_id": "R3_seed_20261315_10a626202961",
        "run_identity_sha256": (
            "10a626202961670fb2f8e929a7f7e3b5accca153b6897834b4357d7eb25c5067"
        ),
        "weight_file": "model.safetensors",
        "weight_sha256": (
            "28969080420d7cfaac6cf545560d1d740d022eb2aa2108e52e9c50c6fa101f53"
        ),
        "projector_sha256": (
            "1e583ae6e0ac6bdf5ce347e250ac305e27d7a49a1e532f50bf4c8b3748a4795c"
        ),
        "validation_B": 0.959552,
        "absolute_validation_B_gap": 0.014056000000000068,
    },
    "R3_seed_20261325": {
        "regime": "R3",
        "seed": 20261325,
        "step": 10,
        "run_id": "R3_seed_20261325_03ebb10bf535",
        "run_identity_sha256": (
            "03ebb10bf53579c185f02b764d2be8ea0d725138ebcab626bb56bbe219c75f09"
        ),
        "weight_file": "model.safetensors",
        "weight_sha256": (
            "ef7adf7137ef7567c5a82b40dd7875ece2eae5fd588222c083002f41c664a1e6"
        ),
        "projector_sha256": (
            "2b04f50fcd4be43e99dbe49bbd461fa31469348c074c9de0bc8c3065dc3a7534"
        ),
        "validation_B": 0.991216,
        "absolute_validation_B_gap": 0.017607999999999957,
    },
}


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_e13_confirmation_lock(repo_root: Path, protocol_commit: str) -> dict[str, str]:
    """Validate protocol bytes, ancestry, and remote push before holdout access."""
    protocol_sha = sha256_file(repo_root / PROTOCOL_PATH)
    committed = subprocess.run(
        ["git", "show", f"{protocol_commit}:{PROTOCOL_PATH.as_posix()}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != protocol_sha:
        raise RuntimeError("E13 protocol working tree differs from its preregistration commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        cwd=repo_root,
        check=True,
    )
    head = _git(repo_root, "rev-parse", "HEAD")
    remote = _git(repo_root, "rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError("E13 confirmation implementation must be pushed before access")
    if canonical_digest(HOLDOUT_SPEC) != HOLDOUT_SPEC_SHA256:
        raise RuntimeError("E13 holdout specification registry drifted")
    return {
        "protocol_commit": str(protocol_commit),
        "protocol_sha256": protocol_sha,
        "holdout_spec_sha256": canonical_digest(HOLDOUT_SPEC),
        "checkpoint_registry_sha256": canonical_digest(CHECKPOINT_REGISTRY),
        "implementation_commit": head,
    }


def resolve_checkpoint(repo_root: Path, key: str) -> Path:
    """Resolve one frozen B-matched checkpoint and refuse any identity drift."""
    entry = CHECKPOINT_REGISTRY[key]
    path = (
        repo_root
        / "runs"
        / "E13_MULTI_SEED"
        / BASELINE_CAMPAIGN
        / "jobs"
        / str(entry["run_id"])
        / "checkpoints"
        / f"step_{int(entry['step']):03d}"
    )
    marker = path / "checkpoint.complete.json"
    if not marker.exists():
        raise RuntimeError(f"frozen E13 checkpoint is missing: {key}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise RuntimeError(f"frozen E13 checkpoint is incomplete: {key}")
    if str(payload.get("identity")) != str(entry["run_identity_sha256"]):
        raise RuntimeError(f"frozen E13 checkpoint identity drifted: {key}")
    if int(payload.get("step", -1)) != int(entry["step"]):
        raise RuntimeError(f"frozen E13 checkpoint step drifted: {key}")
    weights = path / "model" / str(entry["weight_file"])
    if sha256_file(weights) != str(entry["weight_sha256"]):
        raise RuntimeError(f"frozen E13 checkpoint weights drifted: {key}")
    if "projector_sha256" in entry:
        projector = path / "projector.safetensors"
        if sha256_file(projector) != str(entry["projector_sha256"]):
            raise RuntimeError(f"frozen E13 projector drifted: {key}")
    return path


def verify_selection_evidence(repo_root: Path) -> pd.DataFrame:
    """Re-read discovery selection evidence and confirm the registry matches it."""
    records = []
    for key, entry in sorted(CHECKPOINT_REGISTRY.items()):
        summary = (
            repo_root
            / "runs"
            / "E13_MULTI_SEED"
            / BASELINE_CAMPAIGN
            / "jobs"
            / str(entry["run_id"])
            / "job_summary.json"
        )
        job = json.loads(summary.read_text(encoding="utf-8"))
        matched = job["b_matched"]
        if str(job["regime"]) != str(entry["regime"]) or int(job["seed"]) != int(entry["seed"]):
            raise RuntimeError(f"selection evidence identity mismatch for {key}")
        if int(matched["selected_step"]) != int(entry["step"]):
            raise RuntimeError(f"selection evidence step mismatch for {key}")
        if str(matched["selection_split"]) != "validation":
            raise RuntimeError(f"{key} was not selected on validation only")
        if job.get("confirmation_accessed") is not False:
            raise RuntimeError(f"{key} discovery artifact claims confirmation access")
        records.append(
            {
                "key": key,
                "regime": str(entry["regime"]),
                "seed": int(entry["seed"]),
                "selected_step": int(matched["selected_step"]),
                "selection_split": str(matched["selection_split"]),
                "tie_break": str(matched["tie_break"]),
                "validation_B": float(matched["validation_B"]),
                "teacher_validation_B": float(matched["teacher_validation_B"]),
                "absolute_validation_B_gap": float(matched["absolute_B_gap"]),
                "run_identity_sha256": str(job["run_identity_sha256"]),
            }
        )
    return pd.DataFrame(records)


def materialize_e13_holdout(
    open_signatures: set[tuple[str, str]],
    *,
    spec: dict[str, Any] | None = None,
) -> tuple[list[Any], pd.DataFrame]:
    """Materialize the frozen namespace; call only after the access ledger is opened.

    ``open_signatures`` must be the prompt-pair signature set of the frozen open
    corpus so the holdout continues the same global deduplication chain.

    ``spec`` exists so unit tests can exercise this contract against a decoy
    namespace.  Passing anything other than ``HOLDOUT_SPEC`` never touches
    ``e13_confirmation_v1`` and is refused by the runner.
    """
    spec = HOLDOUT_SPEC if spec is None else spec
    needed = int(spec["n_pairs"])
    candidates = generate_synthetic_relations(
        max(int(spec["n_directed"]) * 4, int(spec["n_directed"]) + 200),
        int(spec["generator_seed"]),
        n_entities=int(spec["n_entities"]),
        families=list(spec["families"]),
    )
    seen = set(open_signatures)
    samples: list[Any] = []
    collisions = 0
    kept = 0
    for offset in range(0, len(candidates), 2):
        first, second = candidates[offset : offset + 2]
        signature = _pair_signature(first, second)
        if signature in seen:
            collisions += 1
            continue
        seen.add(signature)
        samples.extend(_rename_pair(first, second, str(spec["split_name"]), kept))
        kept += 1
        if kept == needed:
            break
    if kept != needed:
        raise RuntimeError("E13 confirmation quota unavailable without duplication")
    for sample in samples:
        if not str(sample.sample_id).startswith(str(spec["sample_id_prefix"])):
            raise RuntimeError("E13 confirmation sample namespace violated")
        if not str(sample.pair_id).startswith(str(spec["pair_id_prefix"])):
            raise RuntimeError("E13 confirmation pair namespace violated")
    frame = samples_to_dataframe(samples)
    frame["split"] = "confirmation"
    if len(frame) != int(spec["n_directed"]) or frame["pair_id"].nunique() != needed:
        raise RuntimeError("E13 holdout materialization violates frozen size")
    if frame["sample_id"].duplicated().any() or frame["prompt"].duplicated().any():
        raise RuntimeError("E13 holdout materialization has duplicate identities or prompts")
    if not bool(frame.groupby("pair_id")["sample_id"].size().eq(2).all()):
        raise RuntimeError("E13 holdout materialization split a counterfactual pair")
    frame.attrs["collisions_before_quota"] = collisions
    return samples, frame


def open_e13_access_record(
    root: Path, *, campaign_id: str, protocol_identity: dict[str, str]
) -> dict[str, Any]:
    """Create one immutable access record; resume does not increment access count."""
    path = root / "E13_CONFIRMATION_ACCESS.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("campaign_id") != campaign_id or record.get("access_count") != 1:
            raise RuntimeError("E13 confirmation has already been accessed by another campaign")
        return record
    record = {
        "campaign_id": campaign_id,
        "confirmation_namespace": str(HOLDOUT_SPEC["namespace"]),
        "first_access_timestamp": datetime.now(UTC).isoformat(),
        "access_count": 1,
        "protocol": protocol_identity,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, record)
    return record
