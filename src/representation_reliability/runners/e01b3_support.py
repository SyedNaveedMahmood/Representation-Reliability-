"""Frozen evidence-reuse and identity contracts for E01B-3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runtime.status import StatusFile
from .e01b2_support import e01b2_profile_limits

REQUIRED_E01B2_ARTIFACTS = (
    "intervention_rows.parquet",
    "trace_rows.parquet",
    "source_context_plan.parquet",
    "setpoint_targets.json",
    "probe_metrics.parquet",
    "manifest.json",
    "status.json",
)


def probe_scaler_digest(fits: Mapping[int, Mapping[str, Any]]) -> str:
    """Digest every fitted coefficient, intercept, scaler, and selected C."""
    digest = hashlib.sha256()
    for layer in sorted(map(int, fits)):
        fit = fits[layer]
        digest.update(f"layer={layer}\n".encode())
        arrays = (
            np.asarray(fit["classifier"].coef_, dtype="<f8"),
            np.asarray(fit["classifier"].intercept_, dtype="<f8"),
            np.asarray(fit["scaler_mean"], dtype="<f8"),
            np.asarray(fit["scaler_scale"], dtype="<f8"),
            np.asarray([fit["chosen_C"]], dtype="<f8"),
        )
        for array in arrays:
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def find_frozen_e01b2_run(
    repo_root: Path,
    *,
    model_id: str,
    resolved_revision: str | None,
    profile: str,
    n_pairs: int,
    random_directions: int,
    lambdas: Sequence[float],
    trace_layers: Sequence[int],
) -> Path:
    """Find completed E01B-2 evidence with the exact bounded/full shape."""
    root = repo_root / "runs" / "E01B2"
    if not root.exists():
        raise RuntimeError("E01B-2 evidence root is missing")
    matches: list[Path] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        status = StatusFile.load(candidate)
        metrics_path = candidate / "e01b2_metrics.json"
        manifest_path = candidate / "manifest.json"
        if status is None or not status.is_complete() or not metrics_path.exists():
            continue
        if any(not (candidate / name).exists() for name in REQUIRED_E01B2_ARTIFACTS):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            str(metrics.get("model_id")) == str(model_id)
            and str(metrics.get("resolved_revision")) == str(resolved_revision)
            and str(metrics.get("profile")) == str(profile).lower()
            and int(metrics.get("n_pairs", -1)) == int(n_pairs)
            and int(metrics.get("random_orthogonal_directions", -1)) == int(random_directions)
            and tuple(map(float, metrics.get("context_strengths", [])))
            == tuple(map(float, lambdas))
            and list(map(int, metrics.get("trace_layers", []))) == list(map(int, trace_layers))
            and metrics.get("confirmation_accessed") is False
            and manifest.get("dataset", {}).get("confirmation_accessed") is False
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one compatible E01B-2 run for {model_id}/{profile}; "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_prior_manifest_identity(
    prior_manifest: Mapping[str, Any],
    *,
    model_id: str,
    resolved_revision: str | None,
    tokenizer_revision: str | None,
    candidate_token_ids: Sequence[int],
    split_hash: str,
) -> None:
    model = prior_manifest.get("model", {})
    dataset = prior_manifest.get("dataset", {})
    checks = {
        "model_id": str(model.get("id")) == str(model_id),
        "resolved_revision": str(model.get("resolved_revision")) == str(resolved_revision),
        "tokenizer_revision": str(model.get("tokenizer_revision")) == str(tokenizer_revision),
        "candidate_token_ids": list(map(int, model.get("candidate_token_ids", [])))
        == list(map(int, candidate_token_ids)),
        "split_hash": str(dataset.get("split_hash")) == str(split_hash),
        "confirmation_locked": dataset.get("confirmation_accessed") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"E01B-2 manifest identity mismatch: {failed}")


def validate_source_plan_identity(
    persisted: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    numeric_tolerance: float = 1e-10,
) -> dict[str, float | int]:
    """Require exact plan/source/seed identity and numeric context identity."""
    keys = ["base_sample_id", "condition", "direction_seed"]
    left = persisted.copy()
    right = reconstructed.copy()
    for frame in (left, right):
        frame["direction_seed"] = (
            pd.to_numeric(frame["direction_seed"], errors="coerce").fillna(-1).astype(np.int64)
        )
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise RuntimeError("duplicate E01B-2 source-plan identity")
    merged = left.merge(
        right,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_prior", "_reconstructed"),
        indicator=True,
    )
    mismatch_count = int((merged["_merge"] != "both").sum())
    exact = (
        "base_pair_id",
        "base_relation_family",
        "base_label",
        "context_source_id",
        "context_source_pair_id",
        "context_source_relation_family",
        "context_source_label",
        "context_selection_seed",
        "reference_norm_source",
        "reference_fallback_used",
        "context_vector_fallback_used",
    )
    both = merged[merged["_merge"] == "both"]
    for column in exact:
        lhs = both[f"{column}_prior"].fillna("<NA>").astype(str)
        rhs = both[f"{column}_reconstructed"].fillna("<NA>").astype(str)
        mismatch_count += int((lhs != rhs).sum())
    numeric = (
        "reference_norm",
        "matched_raw_norm",
        "context_raw_norm",
        "context_projected_raw_norm",
        "context_applied_norm",
        "context_dot_truth_direction",
        "context_norm_relative_error",
    )
    max_deviation = 0.0
    for column in numeric:
        deviation = np.abs(
            both[f"{column}_prior"].to_numpy(float)
            - both[f"{column}_reconstructed"].to_numpy(float)
        )
        max_deviation = max(max_deviation, float(np.max(deviation, initial=0.0)))
        mismatch_count += int(np.sum(deviation > float(numeric_tolerance)))
    if mismatch_count:
        raise RuntimeError(
            f"E01B-2 source plan mismatch: {mismatch_count}; max deviation={max_deviation}"
        )
    return {
        "source_plan_mismatch_count": 0,
        "source_plan_max_numeric_deviation": max_deviation,
        "n_plan_rows": len(both),
    }


def validate_e01b3_artifact_shape(
    context_rows: pd.DataFrame,
    context_trace: pd.DataFrame,
    factorial_rows: pd.DataFrame,
    factorial_trace: pd.DataFrame,
    *,
    n_base_examples: int,
    n_context_specs: int,
    n_trace_layers: int,
) -> None:
    expected = int(n_base_examples) * int(n_context_specs)
    if (
        len(context_rows) != expected
        or len(factorial_rows) != expected
        or len(context_trace) != expected * int(n_trace_layers)
        or len(factorial_trace) != expected * int(n_trace_layers)
    ):
        raise RuntimeError("E01B-3 artifact row-count mismatch")
    required = {
        "Y00",
        "Y10",
        "Y01",
        "Y11",
        "Q0",
        "A_context",
        "Q_context",
        "G_interaction",
        "y00_run_id",
        "y10_run_id",
        "y01_run_id",
        "y11_run_id",
    }
    missing = required - set(factorial_rows.columns)
    if missing:
        raise RuntimeError(f"E01B-3 factorial rows missing: {sorted(missing)}")


def e01b3_profile_limits(
    profile: str, max_pairs: int | None, random_directions: int
) -> tuple[int | None, int]:
    """Use the exact E01B-2 bounded/full profile contract."""
    return e01b2_profile_limits(profile, max_pairs, random_directions)
