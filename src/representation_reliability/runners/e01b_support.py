"""Deterministic bookkeeping and cache reuse for E01B-1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..runtime.status import StatusFile
from .e01a_support import load_activation_snapshot


def profile_limits(
    profile: str,
    max_pairs: int | None,
    random_directions: int,
    orthogonal_directions: int,
) -> tuple[int | None, int, int]:
    """Resolve the predeclared bounded/full E01B-1 run shape."""
    name = str(profile).strip().lower()
    if name not in {"smoke", "pilot", "full"}:
        raise ValueError("profile must be smoke, pilot, or full")
    default_cap = {"smoke": 25, "pilot": 75, "full": None}[name]
    cap = int(max_pairs) if max_pairs is not None else default_cap
    if cap is not None and cap <= 0:
        raise ValueError("max_pairs must be positive")
    random_count = min(1 if name == "smoke" else 3, int(random_directions)) if name != "full" else int(random_directions)
    orthogonal_count = min(1 if name == "smoke" else 3, int(orthogonal_directions)) if name != "full" else int(orthogonal_directions)
    if random_count <= 0 or orthogonal_count <= 0:
        raise ValueError("control direction counts must be positive")
    return cap, random_count, orthogonal_count


def validation_identity(discovery_df: pd.DataFrame) -> dict[str, Any]:
    """Return an auditable digest without exposing confirmation rows."""
    if (discovery_df["split"].astype(str) == "confirmation").any():
        raise RuntimeError("confirmation leaked into E01B discovery view")
    validation_ids = sorted(
        discovery_df.loc[
            discovery_df["split"].astype(str) == "validation", "sample_id"
        ].astype(str)
    )
    if not validation_ids:
        raise RuntimeError("E01B validation split is empty")
    return {
        "split": "validation",
        "n": len(validation_ids),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode("utf-8")
        ).hexdigest(),
        "confirmation_accessed": False,
    }


def validate_artifact_shape(
    intervention_rows: pd.DataFrame,
    trace_rows: pd.DataFrame,
    *,
    n_base_examples: int,
    n_treatment_specs: int,
    n_trace_layers: int,
) -> None:
    """Fail before aggregation when E01B raw evidence is incomplete or donor-tainted."""
    required = {
        "base_sample_id",
        "pair_id",
        "condition",
        "target_name",
        "q_base",
        "q_target",
        "q_after",
        "base_yes_no_margin",
        "intervened_yes_no_margin",
        "activation_norm",
        "delta_norm",
        "delta_norm_ratio",
    }
    missing = required - set(intervention_rows.columns)
    if missing:
        raise RuntimeError(f"E01B intervention artifact missing columns: {sorted(missing)}")
    expected_raw = int(n_base_examples) * int(n_treatment_specs)
    expected_trace = expected_raw * int(n_trace_layers)
    if len(intervention_rows) != expected_raw or len(trace_rows) != expected_trace:
        raise RuntimeError("E01B raw evidence row-count mismatch")
    forbidden = {"source_sample_id", "donor_sample_id", "source_hidden"}
    leaked = forbidden & set(intervention_rows.columns)
    if leaked:
        raise RuntimeError(f"donor/source fields leaked into E01B evidence: {sorted(leaked)}")


def find_identity_matched_e01a_snapshot(
    repo_root: Path,
    *,
    model_id: str,
    resolved_revision: str | None,
    split_hash: str,
    expected_sample_ids: list[str],
    expected_layers: list[int],
) -> tuple[tuple[Any, Any, Any], Path] | None:
    """Find a complete E01A snapshot only when every scientific identity matches."""
    root = repo_root / "runs" / "E01A"
    if not root.exists():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        status = StatusFile.load(candidate)
        manifest_path = candidate / "manifest.json"
        if status is None or not status.is_complete() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = manifest.get("model", {})
        dataset = manifest.get("dataset", {})
        if str(model.get("id")) != str(model_id):
            continue
        if str(model.get("resolved_revision")) != str(resolved_revision):
            continue
        if str(dataset.get("split_hash")) != str(split_hash):
            continue
        try:
            snapshot = load_activation_snapshot(
                candidate,
                expected_sample_ids=expected_sample_ids,
                expected_layers=expected_layers,
            )
        except RuntimeError:
            continue
        if snapshot is not None:
            return snapshot, candidate
    return None
