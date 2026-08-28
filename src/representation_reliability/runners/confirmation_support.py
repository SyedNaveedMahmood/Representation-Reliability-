"""Identity locks and non-GPU helpers for the one-shot E01 confirmation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.splits import confirmation_view
from ..interventions.truth_coordinate import normalized_direction
from ..probes.linear import fit_probe, raw_probe_direction
from ..runtime.status import atomic_write_json
from .e01b2_support import ContextSourcePlan, build_context_source_plans
from .e01b3_support import probe_scaler_digest

PROTOCOL_RELATIVE_PATH = Path("docs/E01_CONFIRMATION_PROTOCOL.md")
PROTOCOL_SHA256 = "46312baf59923a2a0e5d1b755d313cd83b42883760baf7b5ba1209410fba81a3"
PREREGISTRATION_COMMIT = "e0ddfae54b350c0545c71a8237645375bdf84929"
CONFIRMATION_VERSION = "e01-actionability-confirmation-v1"
TRACE_LAYERS = (17, 20, 23, 27)
CONTEXT_STRENGTHS = (0.5, 1.0)
PRIMARY_HYPOTHESES = {
    "H1": "Q0 positive and above random/orthogonal controls in both checkpoints",
    "H2": "A_matched above A_random in both checkpoints",
    "H3": "G_matched above G_random in Qwen3-1.7B",
    "H4": "structured G contrast larger in 1.7B than 0.6B",
}
ANALYSIS_DEFINITION_SHA256 = "9245fe544aea0468f8550f65de0fdd224a17c8255ec738c61bf3859296297c0f"
FROZEN_MODELS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-0.6B": {
        "config": "configs/models/qwen3_0.6b.yaml",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "probe_digest": "6177a52089623422091c3f725aaffb18db584063485c91a9c16a7492694d5a2e",
        "target_path": "runs/E01B2/E01B2_f2d75dab1eba/setpoint_targets.json",
        "target_sha256": "709da161845d33e11275b7e66ca0686e652d45cbb86468f101ba7220eae7c7bb",
    },
    "Qwen/Qwen3-1.7B": {
        "config": "configs/models/qwen3_1.7b.yaml",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "probe_digest": "f368240514b0ae5fc9fabd14401656456b16b3a0c9bbc417f8a2dd4982b606b7",
        "target_path": "runs/E01B2/E01B2_e2b4b02cb3a4/setpoint_targets.json",
        "target_sha256": "2c17d27e50869280f64b73624deb31c14302ef729edf263bdd4d0a524a72ccdc",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_definition_digest() -> str:
    payload = {
        "version": CONFIRMATION_VERSION,
        "hypotheses": PRIMARY_HYPOTHESES,
        "trace_layers": TRACE_LAYERS,
        "lambdas": CONTEXT_STRENGTHS,
        "bootstrap_draws": 10_000,
        "randomization_draws": 100_000,
        "bootstrap_seed": 20260831,
        "multiplicity": "Holm H1-H4",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_protocol_lock(repo_root: Path, protocol_commit: str) -> dict[str, str]:
    """Prove the current protocol is byte-identical to its remote preregistration."""
    commit = str(protocol_commit).strip()
    if commit != PREREGISTRATION_COMMIT:
        raise RuntimeError("wrong confirmation protocol commit")
    protocol_path = repo_root / PROTOCOL_RELATIVE_PATH
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("current confirmation protocol digest differs from preregistration")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{PROTOCOL_RELATIVE_PATH.as_posix()}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("protocol commit does not contain the frozen protocol bytes")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("protocol commit is not an ancestor of HEAD")
    analysis_digest = analysis_definition_digest()
    if analysis_digest != ANALYSIS_DEFINITION_SHA256:
        raise RuntimeError("confirmation hypothesis/analysis registry was modified")
    return {
        "protocol_commit": commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "analysis_definition_sha256": analysis_digest,
    }


def load_frozen_targets(repo_root: Path, model_id: str) -> dict[str, Any]:
    identity = FROZEN_MODELS[str(model_id)]
    path = repo_root / str(identity["target_path"])
    if sha256_file(path) != identity["target_sha256"]:
        raise RuntimeError(f"frozen target digest mismatch for {model_id}")
    targets = json.loads(path.read_text(encoding="utf-8"))
    if targets.get("confirmation_accessed") is not False:
        raise RuntimeError("frozen target provenance is not validation-only")
    return targets


def validate_model_identity(
    model_id: str,
    *,
    resolved_revision: str | None,
    tokenizer_revision: str | None,
    candidate_token_ids: Sequence[int],
) -> None:
    identity = FROZEN_MODELS.get(str(model_id))
    if identity is None:
        raise RuntimeError(f"unregistered confirmation model: {model_id}")
    expected = str(identity["revision"])
    if str(resolved_revision) != expected or str(tokenizer_revision) != expected:
        raise RuntimeError(f"frozen model/tokenizer revision mismatch for {model_id}")
    if list(map(int, candidate_token_ids)) != [7414, 2308]:
        raise RuntimeError("frozen Yes/No token IDs differ")


def fit_locked_probe_layers(
    activations: Mapping[int, Mapping[str, np.ndarray]],
    development_df: pd.DataFrame,
    *,
    layers: Sequence[int],
    c_grid: Sequence[float],
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]], str]:
    """Fit only on train/validation and return the frozen scientific digest."""
    observed = set(development_df["split"].astype(str))
    if not observed or not observed.issubset({"train", "validation"}):
        raise RuntimeError("probe development frame may contain only train/validation")
    directions: dict[int, np.ndarray] = {}
    fits: dict[int, dict[str, Any]] = {}
    for layer in map(int, layers):
        blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split in ("train", "validation"):
            rows = development_df[development_df["split"].astype(str) == split]
            ids = rows["sample_id"].astype(str).tolist()
            blocks[split] = (
                np.stack([np.asarray(activations[layer][sid]) for sid in ids]),
                rows["target_label"].to_numpy(int),
            )
        fit = fit_probe(
            blocks["train"][0],
            blocks["train"][1],
            blocks["validation"][0],
            blocks["validation"][1],
            c_grid=c_grid,
            seed=int(seed),
            standardize=True,
            class_weight="balanced",
        )
        fits[layer] = fit
        directions[layer] = normalized_direction(raw_probe_direction(fit))
    return directions, fits, probe_scaler_digest(fits)


def route_confirmation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The runner's sole split-opening function; explicit and unit-testable."""
    rows = confirmation_view(frame, requested_by="confirmatory_evaluation")
    if rows.empty:
        raise RuntimeError("confirmation split is empty")
    return rows.reset_index(drop=True)


def build_confirmation_source_plans(
    confirmation_df: pd.DataFrame,
    samples_by_id: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, ContextSourcePlan]:
    """Reuse the frozen E01B-2 source algorithm on the confirmation pool."""
    if set(confirmation_df["split"].astype(str)) != {"confirmation"}:
        raise RuntimeError("source planning received non-confirmation rows")
    routed = confirmation_df.copy()
    routed["split"] = "discovery_test"
    return build_context_source_plans(
        routed,
        samples_by_id,
        base_sample_ids=routed["sample_id"].astype(str).tolist(),
        seed=int(seed),
    )


def source_plan_frame(plans: Mapping[str, ContextSourcePlan]) -> pd.DataFrame:
    rows = [
        {
            "base_sample_id": sid,
            "matched_source_id": plan.matched_source_id,
            "same_family_source_id": plan.same_family_source_id,
            "different_family_source_id": plan.different_family_source_id,
            "same_label_source_id": plan.same_label_source_id,
            "selection_seed": int(plan.selection_seed),
        }
        for sid, plan in sorted(plans.items())
    ]
    return pd.DataFrame(rows)


def source_plan_digest(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("base_sample_id").to_dict(orient="records")
    return hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()


def require_identity_digest(name: str, actual: str, expected: str) -> None:
    if str(actual) != str(expected):
        raise RuntimeError(f"{name} digest mismatch")


def open_access_record(
    confirmation_root: Path,
    *,
    campaign_id: str,
    git_commit: str,
    protocol_identity: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the global one-shot ledger immediately before split routing."""
    path = confirmation_root / "CONFIRMATION_ACCESS_RECORD.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("campaign_id") == str(campaign_id)
            and existing.get("protocol_commit") == protocol_identity.get("protocol_commit")
            and existing.get("protocol_sha256") == protocol_identity.get("protocol_sha256")
        ):
            return existing
        raise RuntimeError("confirmation was already accessed; a second campaign is forbidden")
    record = {
        "campaign_id": str(campaign_id),
        "first_confirmation_access_timestamp": datetime.now(UTC).isoformat(),
        "confirmation_access_count": 1,
        "git_commit": str(git_commit),
        **dict(protocol_identity),
        "model_revisions": {model: values["revision"] for model, values in FROZEN_MODELS.items()},
        "environment": dict(environment),
    }
    atomic_write_json(path, record)
    return record
