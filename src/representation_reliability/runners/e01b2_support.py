"""Deterministic source planning and artifact contracts for E01B-2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..interventions.orthogonal_context import orthogonal_component
from ..runtime.status import StatusFile
from .e01a_support import _assert_matched_counterfactual

CONTEXT_STRENGTHS = (0.5, 1.0)
CONTEXT_EPSILON = 1e-12


@dataclass(frozen=True)
class ContextSourcePlan:
    base_sample_id: str
    matched_source_id: str
    same_family_source_id: str
    different_family_source_id: str
    same_label_source_id: str
    selection_seed: int


def e01b2_profile_limits(
    profile: str, max_pairs: int | None, random_directions: int
) -> tuple[int | None, int]:
    name = str(profile).strip().lower()
    if name not in {"smoke", "pilot", "full"}:
        raise ValueError("profile must be smoke, pilot, or full")
    cap = (
        int(max_pairs)
        if max_pairs is not None
        else {"smoke": 25, "pilot": 75, "full": None}[name]
    )
    if cap is not None and cap <= 0:
        raise ValueError("max_pairs must be positive")
    requested = int(random_directions)
    count = requested if name == "full" else min(1 if name == "smoke" else 3, requested)
    if count <= 0:
        raise ValueError("random_directions must be positive")
    return cap, count


def parse_context_strengths(raw: str | Sequence[float]) -> tuple[float, ...]:
    values = (
        tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())
        if isinstance(raw, str)
        else tuple(float(value) for value in raw)
    )
    values = tuple(sorted(set(values)))
    if values != CONTEXT_STRENGTHS:
        raise ValueError(
            f"E01B-2 context strengths are frozen at {CONTEXT_STRENGTHS}, got {values}"
        )
    return values


def _stable_choice(
    ids: Sequence[str], *, seed: int, base_id: str, tag: str
) -> str:
    ordered = sorted(map(str, ids))
    if not ordered:
        raise RuntimeError(f"empty E01B-2 source pool for {base_id}: {tag}")
    digest = hashlib.sha256(f"{seed}|{base_id}|{tag}".encode()).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def build_context_source_plans(
    discovery_df: pd.DataFrame,
    samples_by_id: Mapping[str, Any],
    *,
    base_sample_ids: Sequence[str],
    seed: int,
) -> dict[str, ContextSourcePlan]:
    """Select deterministic discovery-test structured context sources."""
    test = discovery_df[
        discovery_df["split"].astype(str) == "discovery_test"
    ].copy()
    if len(test) == 0:
        raise RuntimeError("E01B-2 discovery-test source pool is empty")
    rows = test.set_index("sample_id", drop=False)
    plans: dict[str, ContextSourcePlan] = {}
    for sid in map(str, base_sample_ids):
        if sid not in rows.index:
            raise KeyError(f"base sample {sid!r} is not in discovery_test")
        base = rows.loc[sid]
        pair_id = str(base["pair_id"])
        relation = str(base["relation"])
        label = int(base["target_label"])
        matched_id = str(samples_by_id[sid].counterfactual_id or "")
        if matched_id not in rows.index:
            raise RuntimeError(f"matched context source missing for {sid}")
        _assert_matched_counterfactual(samples_by_id[sid], samples_by_id[matched_id])

        other_pair = test["pair_id"].astype(str) != pair_id
        same_family = test["relation"].astype(str) == relation
        opposite_label = test["target_label"].astype(int) == 1 - label
        same_label = test["target_label"].astype(int) == label
        different_family = test["relation"].astype(str) != relation

        same_family_ids = test.loc[
            other_pair & same_family & opposite_label, "sample_id"
        ].astype(str).tolist()
        different_family_ids = test.loc[
            other_pair & different_family & opposite_label, "sample_id"
        ].astype(str).tolist()
        same_label_pool = test.loc[
            other_pair & same_family & same_label, "sample_id"
        ].astype(str).tolist()
        if not same_label_pool:
            same_label_pool = test.loc[
                other_pair & same_label, "sample_id"
            ].astype(str).tolist()
        plans[sid] = ContextSourcePlan(
            base_sample_id=sid,
            matched_source_id=matched_id,
            same_family_source_id=_stable_choice(
                same_family_ids, seed=seed, base_id=sid, tag="same_family_opposite"
            ),
            different_family_source_id=_stable_choice(
                different_family_ids,
                seed=seed,
                base_id=sid,
                tag="different_family_opposite",
            ),
            same_label_source_id=_stable_choice(
                same_label_pool, seed=seed, base_id=sid, tag="same_label"
            ),
            selection_seed=int(seed),
        )
    validate_context_source_plans(test, plans)
    return plans


def validate_context_source_plans(
    discovery_test_df: pd.DataFrame,
    plans: Mapping[str, ContextSourcePlan],
) -> None:
    rows = discovery_test_df.set_index("sample_id", drop=False)
    for sid, plan in plans.items():
        base = rows.loc[sid]
        pair = str(base["pair_id"])
        relation = str(base["relation"])
        label = int(base["target_label"])
        matched = rows.loc[plan.matched_source_id]
        same_family = rows.loc[plan.same_family_source_id]
        different = rows.loc[plan.different_family_source_id]
        same_label = rows.loc[plan.same_label_source_id]
        if str(matched["pair_id"]) != pair or int(matched["target_label"]) == label:
            raise RuntimeError(f"invalid matched source plan for {sid}")
        if str(same_family["pair_id"]) == pair or str(same_family["relation"]) != relation:
            raise RuntimeError(f"invalid same-family source plan for {sid}")
        if int(same_family["target_label"]) == label:
            raise RuntimeError(f"same-family source must have target label for {sid}")
        if str(different["pair_id"]) == pair or str(different["relation"]) == relation:
            raise RuntimeError(f"invalid different-family source plan for {sid}")
        if int(different["target_label"]) == label:
            raise RuntimeError(f"different-family source must have target label for {sid}")
        if str(same_label["pair_id"]) == pair or int(same_label["target_label"]) != label:
            raise RuntimeError(f"invalid same-label source plan for {sid}")


def validation_matched_norm_fallback(
    discovery_df: pd.DataFrame,
    samples_by_id: Mapping[str, Any],
    layer_activations: Mapping[str, np.ndarray],
    unit_truth_direction: np.ndarray,
    *,
    epsilon: float = CONTEXT_EPSILON,
) -> dict[str, Any]:
    """Compute the frozen fallback from validation matched contexts only."""
    validation_ids = discovery_df.loc[
        discovery_df["split"].astype(str) == "validation", "sample_id"
    ].astype(str).tolist()
    norms: list[float] = []
    for sid in validation_ids:
        matched_id = str(samples_by_id[sid].counterfactual_id or "")
        if matched_id not in layer_activations:
            raise RuntimeError(f"validation matched activation missing for {sid}")
        norm = float(
            np.linalg.norm(
                orthogonal_component(
                    layer_activations[matched_id],
                    layer_activations[sid],
                    unit_truth_direction,
                )
            )
        )
        if norm >= float(epsilon):
            norms.append(norm)
    if not norms:
        raise RuntimeError("no nondegenerate validation matched-context norms")
    ids_digest = hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest()
    return {
        "median_nondegenerate_matched_orthogonal_norm": float(np.median(norms)),
        "n_validation": len(validation_ids),
        "n_nondegenerate": len(norms),
        "epsilon": float(epsilon),
        "validation_ids_sha256": ids_digest,
        "confirmation_accessed": False,
    }


def find_frozen_e01b1_run(
    repo_root: Path,
    *,
    model_id: str,
    resolved_revision: str | None,
) -> Path:
    """Find the immutable completed full E01B-1 evidence for this checkpoint."""
    root = repo_root / "runs" / "E01B1"
    for candidate in sorted(root.iterdir(), reverse=True):
        status = StatusFile.load(candidate)
        manifest_path = candidate / "manifest.json"
        metrics_path = candidate / "e01b1_metrics.json"
        targets_path = candidate / "setpoint_targets.json"
        if (
            status is None
            or not status.is_complete()
            or not manifest_path.exists()
            or not metrics_path.exists()
            or not targets_path.exists()
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            str(manifest.get("model", {}).get("id")) == str(model_id)
            and str(manifest.get("model", {}).get("resolved_revision"))
            == str(resolved_revision)
            and metrics.get("profile") == "full"
            and metrics.get("confirmation_accessed") is False
        ):
            return candidate
    raise RuntimeError(f"completed full E01B-1 run not found for {model_id}")


def find_e01b1_coordinate_reference(
    repo_root: Path,
    *,
    model_id: str,
    resolved_revision: str | None,
    base_sample_ids: Sequence[str],
) -> tuple[Path, pd.DataFrame]:
    """Find an E01B-1 coordinate run with the identical directed-example set.

    Matching the example set also matches bounded-run batch composition. This
    avoids treating expected BF16 batch-shape drift between a smoke and a full
    sweep as an intervention-implementation discrepancy.
    """
    expected_ids = set(map(str, base_sample_ids))
    root = repo_root / "runs" / "E01B1"
    for candidate in sorted(root.iterdir(), reverse=True):
        status = StatusFile.load(candidate)
        manifest_path = candidate / "manifest.json"
        metrics_path = candidate / "e01b1_metrics.json"
        rows_path = candidate / "intervention_rows.parquet"
        if (
            status is None
            or not status.is_complete()
            or not manifest_path.exists()
            or not metrics_path.exists()
            or not rows_path.exists()
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            str(manifest.get("model", {}).get("id")) != str(model_id)
            or str(manifest.get("model", {}).get("resolved_revision"))
            != str(resolved_revision)
            or metrics.get("confirmation_accessed") is not False
        ):
            continue
        rows = pd.read_parquet(rows_path)
        reference = rows[
            rows["condition"] == "source_free_opposite_class_median"
        ][["base_sample_id", "q_target", "intervened_yes_no_margin"]].copy()
        if reference["base_sample_id"].duplicated().any():
            raise RuntimeError(f"duplicate E01B-1 coordinate rows in {candidate}")
        if set(reference["base_sample_id"].astype(str)) == expected_ids:
            return candidate, reference
    raise RuntimeError(
        f"matching E01B-1 coordinate reference not found for {model_id} "
        f"with {len(expected_ids)} examples"
    )


def validate_e01b2_artifact_shape(
    raw: pd.DataFrame,
    trace: pd.DataFrame,
    *,
    n_base_examples: int,
    n_specs: int,
    n_trace_layers: int,
) -> None:
    required = {
        "base_sample_id",
        "pair_id",
        "condition",
        "lambda_context",
        "q_base",
        "q_target",
        "q_after",
        "context_reference_norm",
        "context_applied_norm",
        "context_dot_truth_direction",
        "semantic_delta_norm",
        "context_delta_norm",
        "total_delta_norm",
        "delta_margin_toward_target",
        "context_increment_vs_coordinate_only",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"E01B-2 artifact missing columns: {sorted(missing)}")
    expected = int(n_base_examples) * int(n_specs)
    if len(raw) != expected or len(trace) != expected * int(n_trace_layers):
        raise RuntimeError("E01B-2 artifact row-count mismatch")
