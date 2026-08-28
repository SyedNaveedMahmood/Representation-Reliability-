"""Support utilities for E01A causal-conversion orchestration.

This module deliberately contains deterministic, model-agnostic bookkeeping:
alpha profiles, matched/control source selection, batched activation extraction,
and batched intervention execution. Scientific intervention math lives in
``interventions.truth_coordinate`` and model hooks live in
``adapters.intervention``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from safetensors.numpy import load_file, save_file

from ..adapters.intervention import (
    forward_resid_post_intervention,
    forward_selected_token_logits,
)
from ..extraction.activations import build_token_selections
from ..runtime.status import atomic_write_json

ALPHA_PROFILES: dict[str, tuple[float, ...]] = {
    "smoke": (0.0, 1.0),
    "pilot": (-0.5, 0.0, 0.5, 1.0, 1.5),
    "full": (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0),
}


@dataclass(frozen=True)
class SourcePlan:
    base_sample_id: str
    pair_id: str
    matched_source_id: str
    same_label_source_id: str
    shuffled_source_id: str
    selection_seed: int


def alpha_profile(name: str) -> tuple[float, ...]:
    key = str(name).strip().lower()
    if key not in ALPHA_PROFILES:
        raise ValueError(f"unknown E01A profile {name!r}; choose {sorted(ALPHA_PROFILES)}")
    return ALPHA_PROFILES[key]


def parse_trace_layers(
    raw: str | Sequence[int], *, intervention_layer: int, num_layers: int
) -> list[int]:
    if isinstance(raw, str):
        pieces = [p.strip() for p in raw.split(",") if p.strip()]
        layers = [int(p) for p in pieces]
    else:
        layers = [int(x) for x in raw]
    layers = sorted({int(intervention_layer), *layers})
    bad = [x for x in layers if x < int(intervention_layer) or x >= int(num_layers)]
    if bad:
        raise ValueError(
            "trace layers must be at/after the intervention and within the model: "
            f"{bad}; num_layers={num_layers}"
        )
    return layers


def deterministic_subset_pair_ids(
    discovery_df: pd.DataFrame,
    *,
    max_pairs: int | None,
    seed: int,
) -> list[str]:
    test = discovery_df[discovery_df["split"] == "discovery_test"]
    pair_ids = sorted(test["pair_id"].astype(str).unique().tolist())
    if max_pairs is None or int(max_pairs) >= len(pair_ids):
        return pair_ids
    if int(max_pairs) <= 0:
        raise ValueError("max_pairs must be positive when provided")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(np.asarray(pair_ids, dtype=object), size=int(max_pairs), replace=False)
    return sorted(map(str, chosen.tolist()))


def _stable_index(seed: int, base_id: str, label: int, n: int, tag: str) -> int:
    payload = f"{seed}|{base_id}|{label}|{tag}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % int(n)


def select_same_label_sources(rows: pd.DataFrame, *, seed: int = 0) -> dict[str, str]:
    """Choose deterministic same-label controls outside each base pair."""
    selected: dict[str, str] = {}
    for base in rows.itertuples(index=False):
        sid = str(base.sample_id)
        label = int(base.target_label)
        pair_id = str(base.pair_id)
        relation = str(base.relation)
        eligible = rows[
            (rows["target_label"].astype(int) == label)
            & (rows["pair_id"].astype(str) != pair_id)
            & (rows["sample_id"].astype(str) != sid)
        ]
        matched_relation = eligible[eligible["relation"].astype(str) == relation]
        pool = matched_relation if len(matched_relation) else eligible
        if len(pool) == 0:
            raise RuntimeError(f"insufficient same-label control-source pool for {sid}")
        ids = sorted(pool["sample_id"].astype(str).tolist())
        selected[sid] = ids[_stable_index(seed, sid, label, len(ids), "same")]
    return selected


def select_shuffled_opposite_sources(rows: pd.DataFrame, *, seed: int) -> dict[str, str]:
    """Choose deterministic opposite-label controls outside each base pair."""
    selected: dict[str, str] = {}
    for base in rows.itertuples(index=False):
        sid = str(base.sample_id)
        label = int(base.target_label)
        pair_id = str(base.pair_id)
        relation = str(base.relation)
        eligible = rows[
            (rows["target_label"].astype(int) != label)
            & (rows["pair_id"].astype(str) != pair_id)
            & (rows["sample_id"].astype(str) != sid)
        ]
        matched_relation = eligible[eligible["relation"].astype(str) == relation]
        pool = matched_relation if len(matched_relation) else eligible
        if len(pool) == 0:
            raise RuntimeError(f"insufficient shuffled control-source pool for {sid}")
        ids = sorted(pool["sample_id"].astype(str).tolist())
        selected[sid] = ids[_stable_index(seed, sid, 1 - label, len(ids), "opp")]
    return selected


def _assert_matched_counterfactual(base: Any, matched: Any) -> None:
    """Fail if an authoritative twin violates the generator's nuisance match."""
    base_id = str(base.sample_id)
    matched_id = str(matched.sample_id)
    if str(base.counterfactual_id or "") != matched_id:
        raise RuntimeError(f"counterfactual_id mismatch for {base_id}")
    if str(matched.counterfactual_id or "") != base_id:
        raise RuntimeError(f"counterfactual link is not reciprocal for {base_id}")
    if str(base.pair_id) != str(matched.pair_id):
        raise RuntimeError(f"counterfactual pair_id mismatch for {base_id}")
    if int(base.target_label) == int(matched.target_label):
        raise RuntimeError(f"counterfactual source has same label for {base_id}")
    if int(base.expected_counterfactual_label) != int(matched.target_label):
        raise RuntimeError(f"expected counterfactual label mismatch for {base_id}")

    same_fields = (
        "premise",
        "queried_word",
        "entity_a",
        "entity_b",
        "relation",
        "template_id",
        "question_variant",
    )
    mismatched = [
        field for field in same_fields if base.metadata.get(field) != matched.metadata.get(field)
    ]
    if mismatched:
        raise RuntimeError(f"counterfactual nuisance mismatch for {base_id}: {mismatched}")
    if base.metadata.get("queried_side") == matched.metadata.get("queried_side"):
        raise RuntimeError(f"counterfactual entity order did not reverse for {base_id}")


def build_source_plans(
    discovery_df: pd.DataFrame,
    samples_by_id: Mapping[str, Any],
    *,
    base_sample_ids: Sequence[str],
    seed: int,
) -> dict[str, SourcePlan]:
    """Build matched, same-label and shuffled-opposite discovery-test sources.

    Controls are relation-family matched whenever a candidate exists. They are
    selected deterministically and never use the matched pair as a control.
    """
    test = discovery_df[discovery_df["split"] == "discovery_test"].copy()
    rows = test.set_index("sample_id", drop=False)
    same_sources = select_same_label_sources(test, seed=int(seed))
    shuffled_sources = select_shuffled_opposite_sources(test, seed=int(seed))
    plans: dict[str, SourcePlan] = {}
    for sid in map(str, base_sample_ids):
        if sid not in rows.index:
            raise KeyError(f"base sample {sid!r} is not in discovery_test")
        sample = samples_by_id[sid]
        matched = str(sample.counterfactual_id or "")
        if not matched or matched not in rows.index:
            raise RuntimeError(f"matched counterfactual missing for {sid}")
        base_row = rows.loc[sid]
        matched_row = rows.loc[matched]
        _assert_matched_counterfactual(sample, samples_by_id[matched])
        base_label = int(base_row["target_label"])
        pair_id = str(base_row["pair_id"])
        if int(matched_row["target_label"]) == base_label:
            raise RuntimeError(f"counterfactual source has same label for {sid}")
        if str(matched_row["pair_id"]) != pair_id:
            raise RuntimeError(f"counterfactual pair_id mismatch for {sid}")

        same_id = same_sources[sid]
        opp_id = shuffled_sources[sid]
        plans[sid] = SourcePlan(
            base_sample_id=sid,
            pair_id=pair_id,
            matched_source_id=matched,
            same_label_source_id=same_id,
            shuffled_source_id=opp_id,
            selection_seed=int(seed),
        )
    return plans


def extract_resid_post_layers(
    adapter,
    samples: Sequence[Any],
    *,
    layers: Sequence[int],
    token_selector: str = "last_prompt",
    batch_size: int = 8,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[str, int],
    dict[str, dict[str, Any]],
]:
    """Extract selected resid_post layers and resolved token indices in batches."""
    unique_layers = sorted(set(map(int, layers)))
    requests = [("resid_post", layer) for layer in unique_layers]
    store: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in unique_layers}
    token_indices: dict[str, int] = {}
    token_sites: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(samples), stride):
        chunk = list(samples[start : start + stride])
        prompts = [s.prompt for s in chunk]
        indices: list[int] = []
        for sample in chunk:
            resolved = build_token_selections(adapter, sample, [token_selector])[0]
            indices.append(int(resolved.index))
            token_indices[str(sample.sample_id)] = int(resolved.index)
            token_sites[str(sample.sample_id)] = resolved.as_dict()
        out = adapter.extract(prompts, requests=requests, token_indices=indices)
        for layer in unique_layers:
            arr = np.asarray(out[("resid_post", layer)], dtype=np.float64)
            for sample, vector in zip(chunk, arr):
                store[layer][str(sample.sample_id)] = vector.copy()
    expected = {str(s.sample_id) for s in samples}
    if set(token_indices) != expected:
        raise RuntimeError("token-index extraction sample identity mismatch")
    for layer in unique_layers:
        if set(store[layer]) != expected:
            raise RuntimeError(f"activation extraction sample identity mismatch at layer {layer}")
    return store, token_indices, token_sites


def save_activation_snapshot(
    run_dir: str | Path,
    activations: Mapping[int, Mapping[str, np.ndarray]],
    *,
    sample_ids: Sequence[str],
    token_indices: Mapping[str, int],
    token_sites: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically save the expensive E01A discovery activation snapshot."""
    root = Path(run_dir)
    tensor_path = root / "activation_snapshot.safetensors"
    temp_path = root / ".activation_snapshot.tmp.safetensors"
    ids = list(map(str, sample_ids))
    tensors = {
        f"layer_{int(layer)}": np.stack(
            [np.asarray(by_id[sid], dtype=np.float32) for sid in ids]
        )
        for layer, by_id in sorted(activations.items())
    }
    if temp_path.exists():
        temp_path.unlink()
    save_file(tensors, temp_path)
    os.replace(temp_path, tensor_path)
    atomic_write_json(
        root / "activation_snapshot.json",
        {
            "sample_ids": ids,
            "layers": sorted(map(int, activations)),
            "token_indices": {sid: int(token_indices[sid]) for sid in ids},
            "token_sites": {sid: dict(token_sites[sid]) for sid in ids},
        },
    )


def load_activation_snapshot(
    run_dir: str | Path,
    *,
    expected_sample_ids: Sequence[str],
    expected_layers: Sequence[int],
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[str, int],
    dict[str, dict[str, Any]],
] | None:
    """Load a complete identity-bound snapshot, or return None if absent."""
    root = Path(run_dir)
    tensor_path = root / "activation_snapshot.safetensors"
    metadata_path = root / "activation_snapshot.json"
    if not tensor_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ids = list(map(str, expected_sample_ids))
    layers = sorted(set(map(int, expected_layers)))
    if metadata.get("sample_ids") != ids or metadata.get("layers") != layers:
        raise RuntimeError("E01A activation snapshot identity mismatch")
    tensors = load_file(tensor_path)
    expected_keys = {f"layer_{layer}" for layer in layers}
    if set(tensors) != expected_keys:
        raise RuntimeError("E01A activation snapshot layer mismatch")
    activations: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        matrix = np.asarray(tensors[f"layer_{layer}"], dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) != len(ids):
            raise RuntimeError("E01A activation snapshot shape mismatch")
        activations[layer] = {
            sid: matrix[row].copy() for row, sid in enumerate(ids)
        }
    token_indices = {
        str(sid): int(index)
        for sid, index in metadata["token_indices"].items()
    }
    token_sites = {
        str(sid): dict(site) for sid, site in metadata["token_sites"].items()
    }
    if set(token_indices) != set(ids) or set(token_sites) != set(ids):
        raise RuntimeError("E01A activation snapshot token-site mismatch")
    return activations, token_indices, token_sites


def run_intervention_batches(
    adapter,
    samples_by_id: Mapping[str, Any],
    base_sample_ids: Sequence[str],
    *,
    layer: int,
    token_indices: Mapping[str, int],
    deltas_by_id: Mapping[str, np.ndarray],
    output_token_ids: Sequence[int],
    capture_layers: Sequence[int],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Execute one intervention condition over base IDs with exact row identity."""
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    ids = list(map(str, base_sample_ids))
    for start in range(0, len(ids), stride):
        chunk_ids = ids[start : start + stride]
        prompts = [samples_by_id[sid].prompt for sid in chunk_ids]
        idx = [int(token_indices[sid]) for sid in chunk_ids]
        deltas = np.stack([np.asarray(deltas_by_id[sid], dtype=np.float64) for sid in chunk_ids])
        out = forward_resid_post_intervention(
            adapter,
            prompts,
            layer=int(layer),
            token_indices=idx,
            deltas=deltas,
            output_token_ids=list(map(int, output_token_ids)),
            capture_layers=list(map(int, capture_layers)),
        )
        selected = np.asarray(out["selected_logits"], dtype=np.float64)
        for row, sid in enumerate(chunk_ids):
            result[sid] = {
                "selected_logits": selected[row].copy(),
                "captured": {
                    int(trace_layer): np.asarray(trace_values[row], dtype=np.float64).copy()
                    for trace_layer, trace_values in out["captured"].items()
                },
            }
    if set(result) != set(ids):
        raise RuntimeError("intervention result sample identity mismatch")
    return result


def run_unintervened_batches(
    adapter,
    samples_by_id: Mapping[str, Any],
    base_sample_ids: Sequence[str],
    *,
    token_indices: Mapping[str, int],
    output_token_ids: Sequence[int],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Execute an unhooked baseline forward with exact row identity."""
    result: dict[str, np.ndarray] = {}
    stride = max(1, int(batch_size))
    ids = list(map(str, base_sample_ids))
    for start in range(0, len(ids), stride):
        chunk_ids = ids[start : start + stride]
        logits = forward_selected_token_logits(
            adapter,
            [samples_by_id[sid].prompt for sid in chunk_ids],
            token_indices=[int(token_indices[sid]) for sid in chunk_ids],
            output_token_ids=output_token_ids,
        )
        for row, sid in enumerate(chunk_ids):
            result[sid] = np.asarray(logits[row], dtype=np.float64).copy()
    if set(result) != set(ids):
        raise RuntimeError("unintervened result sample identity mismatch")
    return result


def intervention_base_activations(
    clean_results: Mapping[str, Mapping[str, Any]],
    base_sample_ids: Sequence[str],
    *,
    layer: int,
) -> dict[str, np.ndarray]:
    """Use base states captured with the exact intervention batch shapes."""
    bases: dict[str, np.ndarray] = {}
    for sid in map(str, base_sample_ids):
        try:
            vector = clean_results[sid]["captured"][int(layer)]
        except KeyError as exc:
            raise RuntimeError(
                f"clean intervention-layer activation missing for {sid}"
            ) from exc
        bases[sid] = np.asarray(vector, dtype=np.float64).copy()
    return bases
