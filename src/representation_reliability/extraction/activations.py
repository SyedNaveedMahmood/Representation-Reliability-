"""Extraction orchestration: sharded, atomic, resumable activation caching."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..adapters.hf import HFAdapter
from ..contracts import Sample
from ..runtime.manifest import prompt_hash
from .cache import (
    ActivationCacheReader,
    ActivationShardWriter,
    completed_shard_ids,
    shard_dir,
)
from .token_selection import ResolvedToken, resolve_token_selection

logger = logging.getLogger(__name__)


def build_token_selections(
    adapter: HFAdapter,
    sample: Sample,
    selector_names: Sequence[str],
) -> list[ResolvedToken]:
    """Resolve every requested selector for one sample."""
    resolved: list[ResolvedToken] = []
    for name in selector_names:
        resolved.append(
            resolve_token_selection(
                strategy=name,
                tokenizer=adapter.tokenizer,
                prompt_text=sample.prompt,
                chat_template_used=bool(sample.metadata.get("chat_template_used", False)),
                target_text=sample.metadata.get("target_text"),
            )
        )
    return resolved


def extract_dataset_activations(
    adapter: HFAdapter,
    samples: Sequence[Sample],
    cache_dir: str | Path,
    *,
    sites: Sequence[str],
    layers: Sequence[int],
    token_selectors: Sequence[str],
    shard_size: int,
    batch_size: int,
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    split_of: Callable[[str], str],
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract activations into a sharded portable cache; returns (index_df, info).

    Work unit = one (sample, token_selector). Each batched forward pass yields
    ``len(requests)`` activation rows per unit. Resume semantics: the longest
    contiguous prefix of complete shards is kept; any later (stale) shard dirs
    are pruned and deterministically rewritten.
    """
    requests: list[tuple[str, int]] = [
        (site, layer) for site in sites for layer in layers
    ]
    # Deterministic ordering: site-major then layer.
    requests.sort()

    units: list[tuple[int, str]] = []
    for si in range(len(samples)):
        for sel_name in token_selectors:
            units.append((si, sel_name))
    total_units = len(units)

    # Shard boundaries must be deterministic in WORK-UNIT space for exact
    # resumption: shard k always holds units [k*units_per_shard, ...)
    rows_per_unit = len(requests)
    units_per_shard = max(1, int(shard_size) // rows_per_unit)

    all_ids = set(completed_shard_ids(cache_dir) if resume else [])
    prefix_count = 0
    while prefix_count in all_ids:
        prefix_count += 1
    _prune_stale_shards(cache_dir, prefix_count)
    start_unit = prefix_count * units_per_shard

    writer = ActivationShardWriter(cache_dir, shard_size=shard_size)
    writer.shard_count = prefix_count
    units_since_flush = 0

    started = _dt.datetime.now(_dt.timezone.utc)
    n_rows_before = sum(
        _marker_rows(cache_dir, sid) for sid in range(prefix_count)
    )

    def flush_batch(batch_prompts, batch_indices, batch_metas) -> int:
        """Run forward pass + write activation rows. Returns rows written."""
        result = adapter.extract(
            prompts=batch_prompts, requests=requests, token_indices=batch_indices
        )
        flat_requests = sorted(result.keys(), key=lambda r: (r[0], r[1]))
        written = 0
        for row_i in range(len(batch_prompts)):
            meta0 = batch_metas[row_i]
            for (site, layer) in flat_requests:
                vec = result[(site, layer)][row_i]
                meta = {
                    "sample_id": meta0["sample_id"],
                    "model_id": model_id,
                    "model_revision": model_revision or "unspecified",
                    "tokenizer_revision": tokenizer_revision or "unspecified",
                    "site": site,
                    "layer": int(layer),
                    "native_module_name": adapter.resolve_site(
                        site, layer
                    ).native_module_name,
                    "token_selector": meta0["token_selector"],
                    "token_index": meta0["token_index"],
                    "token_id": meta0["token_id"],
                    "token_text": meta0["token_text"],
                    "char_start": meta0.get("char_start"),
                    "char_end": meta0.get("char_end"),
                    "sequence_length": meta0["sequence_length"],
                    "chat_template_used": meta0["chat_template_used"],
                    "prompt_hash": meta0["prompt_hash"],
                    "split": meta0["split"],
                    "tensor_file": f"shard_{writer.shard_count:05d}/activations.safetensors",
                }
                writer.add(vec, meta)
                written += 1
        return written

    batch_prompts: list[str] = []
    batch_indices: list[int] = []
    batch_metas: list[dict[str, Any]] = []
    units_processed = 0
    rows_written = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(
            initial=start_unit, total=total_units, desc="extract",
            unit="unit", dynamic_ncols=True,
        )
    except ImportError:  # pragma: no cover
        pbar = None

    def flush_now() -> None:
        nonlocal units_processed, rows_written
        if not batch_prompts:
            return
        rows_written += flush_batch(batch_prompts, batch_indices, batch_metas)
        n_units = len(batch_prompts)
        units_processed += n_units
        if pbar is not None:
            pbar.update(n_units)
        batch_prompts.clear()
        batch_indices.clear()
        batch_metas.clear()

    for ui in range(start_unit, total_units):
        si, sel_name = units[ui]
        sample = samples[si]
        resolved = build_token_selections(adapter, sample, [sel_name])[0]
        batch_prompts.append(sample.prompt)
        batch_indices.append(int(resolved.index))
        batch_metas.append({
            "sample_id": sample.sample_id,
            "token_selector": resolved.strategy,
            "token_index": int(resolved.index),
            "token_id": int(resolved.token_id),
            "token_text": resolved.token_text,
            "char_start": resolved.char_start,
            "char_end": resolved.char_end,
            "sequence_length": int(resolved.sequence_length),
            "chat_template_used": bool(resolved.chat_template_used),
            "prompt_hash": prompt_hash(sample.prompt),
            "split": split_of(sample.sample_id),
        })
        units_since_flush += 1
        if len(batch_prompts) >= batch_size:
            flush_now()
        if units_since_flush >= units_per_shard:
            writer.flush()
            units_since_flush = 0
    flush_now()
    if pbar is not None:
        pbar.close()

    # Flush the trailing partial shard (units after the last full shard).
    writer.flush()

    rows_total = sum(_marker_rows(cache_dir, sid) for sid in completed_shard_ids(cache_dir))
    elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()

    reader = ActivationCacheReader(cache_dir)
    index_df = reader.index()
    info = {
        "started_at": started.isoformat(),
        "elapsed_s": elapsed,
        "units_total": total_units,
        "units_processed_this_call": units_processed,
        "rows_before_this_call": int(n_rows_before),
        "rows_written_this_call": int(rows_written),
        "rows_cached_total": int(rows_total),
        "expected_rows_per_sample": len(requests) * len(token_selectors),
        "complete_shards": len(completed_shard_ids(cache_dir)),
    }
    logger.info("extraction finished: %s", info)
    return index_df, info


def _prune_stale_shards(cache_dir: str | Path, keep_prefix_count: int) -> None:
    """Remove non-complete or out-of-prefix shard dirs.

    Everything from ``keep_prefix_count`` onward is deterministically
    reproducible from the sample order, so pruning is safe.
    """
    import shutil

    root = Path(cache_dir)
    if not root.exists():
        return
    for child in sorted(root.glob("shard_*")):
        try:
            idx = int(child.name.split("_")[-1])
        except ValueError:
            continue
        if idx < keep_prefix_count:
            continue
        shutil.rmtree(child, ignore_errors=True)


def _marker_rows(cache_dir, sid: int) -> int:
    with (shard_dir(Path(cache_dir), sid) / "_complete.json").open(encoding="utf-8") as fh:
        return int(json.load(fh)["n_rows"])

