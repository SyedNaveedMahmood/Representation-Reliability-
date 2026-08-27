"""Extraction runner: wires config + adapter into the sharded cache."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..adapters.hf import HFAdapter
from ..extraction.activations import extract_dataset_activations

logger = logging.getLogger(__name__)


def expand_layer_plan(cfg, num_layers: int) -> list[int]:
    plan = cfg.representation.layers
    if plan == "all":
        return list(range(num_layers))
    layers = sorted({int(x) for x in plan})
    bad = [l for l in layers if l < 0 or l >= num_layers]
    if bad:
        raise ValueError(f"configured layers {bad} out of range for {num_layers}-layer model")
    return layers


def load_adapter(cfg) -> HFAdapter:
    adapter = HFAdapter(cfg.model, cfg.runtime)
    adapter.load()
    logger.info(
        "loaded %s (%d layers, hidden=%d)", cfg.model.id,
        adapter.num_layers, adapter.hidden_size,
    )
    # bf16 weights imply autocast-free inference; keep inference_mode inside calls.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return adapter


def run_extraction_stage(
    cfg,
    adapter: HFAdapter,
    samples,                      # sequence[Sample]
    cache_dir: str | Path,
    split_of,                     # Callable[[sample_id], str]
    identity: dict | None = None,
    resume: bool | None = None,
):
    resume = bool(cfg.runtime.resume if resume is None else resume)
    layers = expand_layer_plan(cfg, adapter.num_layers)
    token_selectors = list(cfg.representation.token_selectors)

    if identity is None:
        raise ValueError("schema-v2 extraction requires an explicit cache identity")

    index_df, info = extract_dataset_activations(
        adapter,
        samples,
        cache_dir,
        sites=list(cfg.representation.sites),
        layers=layers,
        token_selectors=token_selectors,
        shard_size=int(cfg.storage.shard_size),
        batch_size=int(cfg.runtime.batch_size),
        split_of=split_of,
        identity=identity,
        resume=resume,
    )
    return index_df, {**info, "layers": layers, "token_selectors": token_selectors}
