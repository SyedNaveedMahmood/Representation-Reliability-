"""Identity locks and holdout materialization for the one-shot E14 confirmation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.base import samples_to_dataframe
from ..data.synthetic import generate_synthetic_relations
from ..runtime.status import atomic_write_json
from .confirmation_support import sha256_file

PROTOCOL_PATH = Path("docs/E14_FULL_DISCOVERY_AND_CONFIRMATION_PROTOCOL.md")
PROTOCOL_SHA256 = "54e5d6865ea91f41936948271b3bfcf357240017e82ba853733bef67bd5dbbef"
PREREGISTRATION_COMMIT = "3893814c11ba3f7bc4bc39ccae83191431c35443"
CONFIRMATION_VERSION = "e14-quantization-confirmation-v1"
TRACE_LAYERS = (17, 20, 23, 27)
RANDOM_SEEDS = tuple(range(1729, 1739))
CONTEXT_STRENGTHS = (0.5, 1.0)
FULL_RUNS = {
    "bf16": "E14_98b8d199abbc",
    "int8": "E14_31ff5aa1a9a2",
    "int4": "E14_5cb70b93e672",
}
REFERENCE_DIGESTS = {
    "bf16_probe_reference.npz": "4b88e0277d5792ac1368676227c8b80f65618bcfe68888693070f2e254d242aa",
    "bf16_reference.json": "1e43d9ed1b765c06a86477f6c4df64ee347cba13eca7d557e3bdd5d0401071b",
    "bf16_native_probe_reference.npz": "4b88e0277d5792ac1368676227c8b80f65618bcfe68888693070f2e254d242aa",
    "int8_native_probe_reference.npz": "648ea88934904974ca4fa295d6883f5c91200b47e1e2a06bd95038545fca5227",
    "int4_native_probe_reference.npz": "536096411836b7a3944028c9269e1f418ebb57088341005ac40d0d4d9006275b",
}
HOLDOUT_SPEC: dict[str, Any] = {
    "namespace": "e14_confirmation_v1",
    "generator": "generate_synthetic_relations",
    "generator_seed": 20261401,
    "n_samples": 200,
    "n_entities": 42,
    "families": [
        "north_south",
        "east_west",
        "above_below",
        "before_after",
        "larger_smaller",
    ],
    "sample_id_prefix": "e14-confirmation-v1-",
    "pair_id_prefix": "e14-confirmation-v1-",
}
HOLDOUT_SPEC_SHA256 = "2c166f298b9800a6061b854963fb17791c8265c7b9c801aab9ffb1cd276eba50"


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_e14_confirmation_lock(repo_root: Path, protocol_commit: str) -> dict[str, str]:
    """Validate protocol bytes, ancestry, remote push, and full-discovery gate."""
    if str(protocol_commit).strip() != PREREGISTRATION_COMMIT:
        raise RuntimeError("wrong E14 confirmation protocol commit")
    if sha256_file(repo_root / PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("E14 protocol bytes differ from preregistration")
    committed = subprocess.run(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:{PROTOCOL_PATH.as_posix()}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("E14 protocol commit does not contain the locked bytes")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"],
        cwd=repo_root,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != remote:
        raise RuntimeError("E14 confirmation implementation must be pushed before access")
    if canonical_digest(HOLDOUT_SPEC) != HOLDOUT_SPEC_SHA256:
        raise RuntimeError("E14 holdout specification registry drifted")
    discovery = json.loads(
        (repo_root / "E14_FULL_DISCOVERY_SUMMARY.json").read_text(encoding="utf-8")
    )
    if discovery.get("full_discovery_gate", {}).get("pass") is not True:
        raise RuntimeError("E14 full-discovery gate did not pass")
    return {
        "protocol_commit": PREREGISTRATION_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "holdout_spec_sha256": HOLDOUT_SPEC_SHA256,
        "implementation_commit": head,
    }


def validate_frozen_references(repo_root: Path) -> dict[str, Path]:
    paths = {
        precision: repo_root / "runs" / "E14" / run_id
        for precision, run_id in FULL_RUNS.items()
    }
    checks = {
        "bf16_probe_reference.npz": paths["bf16"] / "bf16_probe_reference.npz",
        "bf16_reference.json": paths["bf16"] / "bf16_reference.json",
        "bf16_native_probe_reference.npz": paths["bf16"] / "native_probe_reference.npz",
        "int8_native_probe_reference.npz": paths["int8"] / "native_probe_reference.npz",
        "int4_native_probe_reference.npz": paths["int4"] / "native_probe_reference.npz",
    }
    for key, path in checks.items():
        if sha256_file(path) != REFERENCE_DIGESTS[key]:
            raise RuntimeError(f"frozen E14 reference digest mismatch: {key}")
    return paths


def materialize_e14_holdout() -> tuple[list[Any], pd.DataFrame]:
    """Materialize the frozen namespace; call only after the access ledger is opened."""
    spec = HOLDOUT_SPEC
    raw = generate_synthetic_relations(
        int(spec["n_samples"]),
        int(spec["generator_seed"]),
        n_entities=int(spec["n_entities"]),
        families=list(spec["families"]),
    )
    sample_prefix = str(spec["sample_id_prefix"])
    pair_prefix = str(spec["pair_id_prefix"])
    id_map = {str(sample.sample_id): sample_prefix + str(sample.sample_id) for sample in raw}
    samples = []
    for sample in raw:
        pair_id = pair_prefix + str(sample.pair_id)
        metadata = dict(sample.metadata)
        metadata["pair_id"] = pair_id
        samples.append(
            replace(
                sample,
                sample_id=id_map[str(sample.sample_id)],
                pair_id=pair_id,
                counterfactual_id=id_map[str(sample.counterfactual_id)],
                metadata=metadata,
            )
        )
    frame = samples_to_dataframe(samples)
    frame["split"] = "confirmation"
    if len(frame) != 200 or frame["pair_id"].nunique() != 100:
        raise RuntimeError("E14 holdout materialization violates frozen size")
    if frame["sample_id"].duplicated().any() or frame["pair_id"].isna().any():
        raise RuntimeError("E14 holdout materialization has invalid identities")
    return samples, frame


def open_e14_access_record(
    root: Path, *, campaign_id: str, protocol_identity: dict[str, str]
) -> dict[str, Any]:
    """Create one immutable access record; resume does not increment access count."""
    path = root / "E14_CONFIRMATION_ACCESS.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("campaign_id") != campaign_id or record.get("access_count") != 1:
            raise RuntimeError("E14 confirmation has already been accessed by another campaign")
        return record
    record = {
        "campaign_id": campaign_id,
        "first_access_timestamp": datetime.now(UTC).isoformat(),
        "access_count": 1,
        "protocol": protocol_identity,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, record)
    return record
