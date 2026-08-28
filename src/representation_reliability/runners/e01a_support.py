"""Support utilities for E01A causal-conversion orchestration.

This module deliberately contains deterministic, model-agnostic bookkeeping:
alpha profiles, matched/control source selection, batched activation extraction,
and batched intervention execution. Scientific intervention math lives in
``interventions.truth_coordinate`` and model hooks live in
``adapters.intervention``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..adapters.intervention import forward_resid_post_intervention
from ..extraction.activations import build_token_selections


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
    layers = sorted(set([int(intervention_layer), *layers]))
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
    payload = f"{seed}|{base_id}|{label}|{tag}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % int(n)


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
        base_label = int(base_row["target_label"])
        relation = str(base_row["relation"])
        pair_id = str(base_row["pair_id"])
        if int(matched_row["target_label"]) == base_label:
            raise RuntimeError(f"counterfactual source has same label for {sid}")
        if str(matched_row["pair_id"]) != pair_id:
            raise RuntimeError(f"counterfactual pair_id mismatch for {sid}")

        same = test[
            (test["relation"].astype(str) == relation)
            & (test["target_label"].astype(int) == base_label)
            & (test["pair_id"].astype(str) != pair_id)
            & (test["sample_id"].astype(str) != sid)
        ]
        if len(same) == 0:
            same = test[
                (test["target_label"].astype(int) == base_label)
                & (test["pair_id"].astype(str) != pair_id)
                & (test["sample_id"].astype(str) != sid)
            ]
        opp = test[
            (test["relation"].astype(str) == relation)
            & (test["target_label"].astype(int) != base_label)
            & (test["pair_id"].astype(str) != pair_id)
            & (test["sample_id"].astype(str) != matched)
        ]
        if len(opp) == 0:
            opp = test[
                (test["target_label"].astype(int) != base_label)
                & (test["pair_id"].astype(str) != pair_id)
                & (test["sample_id"].astype(str) != matched)
            ]
        if len(same) == 0 or len(opp) == 0:
            raise RuntimeError(f"insufficient control-source pool for {sid}")
        same_ids = sorted(same["sample_id"].astype(str).tolist())
        opp_ids = sorted(opp["sample_id"].astype(str).tolist())
        same_id = same_ids[_stable_index(seed, sid, base_label, len(same_ids), "same")]
        opp_id = opp_ids[_stable_index(seed, sid, 1 - base_label, len(opp_ids), "opp")]
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
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, int]]:
    """Extract selected resid_post layers and resolved token indices in batches."""
    unique_layers = sorted(set(map(int, layers)))
    requests = [("resid_post", layer) for layer in unique_layers]
    store: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in unique_layers}
    token_indices: dict[str, int] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(samples), stride):
        chunk = list(samples[start : start + stride])
        prompts = [s.prompt for s in chunk]
        indices: list[int] = []
        for sample in chunk:
            resolved = build_token_selections(adapter, sample, [token_selector])[0]
            indices.append(int(resolved.index))
            token_indices[str(sample.sample_id)] = int(resolved.index)
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
    return store, token_indices


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
        deltas = np.stack(
            [np.asarray(deltas_by_id[sid], dtype=np.float64) for sid in chunk_ids]
        )
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
