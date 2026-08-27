"""Extraction orchestration: sharded, atomic, resumable activation caching.

Cache schema v2 integrity rules enforced here:

- A physical shard always holds an EXACT contiguous work-unit range; its
  ``_complete.json`` records ``(unit_start, unit_end_exclusive)``.
- Batches never cross a declared shard boundary silently: when the next work
  unit belongs to a new shard, all pending forward-pass work is flushed into
  the current shard first and the shard is closed with its truthful range.
- Resume derives its next work unit from validated marker metadata
  (contiguous chained unit ranges), never from assumed arithmetic.
- Cache reuse requires a matching schema-v2 ``cache_identity.json``
  (dataset content, model/tokenizer resolved revisions, site/layer/selector
  plan, dtype, tokenization settings); mismatch refuses loudly and legacy
  caches without identity are never adopted.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import Sample
from ..runtime.manifest import prompt_hash
from .cache import (
    ActivationCacheReader,
    ActivationShardWriter,
    completed_shard_ids,
    ensure_cache_identity,
    read_marker,
)
from .token_selection import ResolvedToken, resolve_token_selection

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 2


def dataset_content_hash(samples: Sequence[Sample]) -> str:
    """Order-sensitive hash over every sample's id and raw prompt text."""
    h = hashlib.sha256()
    for s in samples:
        h.update(f"{s.sample_id}\x00{s.prompt}\x1e".encode())
    return h.hexdigest()


def build_cache_identity(
    *,
    experiment_id: str,
    samples: Sequence[Sample],
    model_id: str,
    model_resolved_revision: str | None,
    tokenizer_id: str,
    tokenizer_resolved_revision: str | None,
    sites: Sequence[str],
    layers: Sequence[int],
    token_selectors: Sequence[str],
    model_dtype: str,
    storage_dtype: str = "float32",
    tokenization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the strict semantic identity of an activation cache."""
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_content_hash": dataset_content_hash(samples),
        "model_id": model_id,
        "model_resolved_revision": model_resolved_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_resolved_revision": tokenizer_resolved_revision,
        "sites": sorted(set(sites)),
        "layers": [int(x) for x in layers],
        "token_selectors": sorted(set(token_selectors)),
        "model_dtype": model_dtype,
        "storage_dtype": storage_dtype,
        "tokenization": {
            "add_special_tokens": True,
            "padding_side": "right",
            **(tokenization or {}),
        },
    }


def build_token_selections(
    adapter,
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


def _validated_resume_state(cache_dir: Path, rows_per_unit: int) -> tuple[int, int]:
    """Validate the contiguous marker chain; return (prefix_shards, resume_unit).

    Raises loudly if complete shards exist whose unit ranges do not chain
    exactly from zero or whose row counts contradict their recorded ranges.
    """
    ids = sorted(completed_shard_ids(cache_dir))
    expected_sid = 0
    unit_cursor = 0
    for sid in ids:
        if sid != expected_sid:
            break  # gap -> prefix ends here
        m = read_marker(cache_dir, sid)
        start, end = int(m["unit_start"]), int(m["unit_end_exclusive"])
        if start != unit_cursor:
            raise RuntimeError(
                f"shard {sid} unit range {start}..{end} does not chain from "
                f"cursor {unit_cursor}; cache state inconsistent"
            )
        expected_rows = (end - start) * rows_per_unit
        if end > start and int(m["n_rows"]) != expected_rows:
            raise RuntimeError(
                f"shard {sid} holds {m['n_rows']} rows but its unit range "
                f"implies {expected_rows}; cache state inconsistent"
            )
        if end == start and int(m["n_rows"]) != 0:
            raise RuntimeError(f"shard {sid} has empty range with rows")
        unit_cursor = end
        expected_sid += 1
    return expected_sid, unit_cursor


def _prune_stale_shards(cache_dir: str | Path, keep_prefix_count: int) -> None:
    """Remove incomplete or out-of-prefix shard dirs (deterministically rewritable)."""
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


def extract_dataset_activations(
    adapter,
    samples: Sequence[Sample],
    cache_dir: str | Path,
    *,
    sites: Sequence[str],
    layers: Sequence[int],
    token_selectors: Sequence[str],
    shard_size: int,
    batch_size: int,
    split_of: Callable[[str], str],
    identity: dict[str, Any],
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract activations into a schema-v2 sharded cache; returns (index_df, info).

    Work unit = one ``(sample_index, token_selector)`` pair; each unit yields
    exactly ``len(requests)`` activation rows. Shard ``k`` deterministically
    covers work units ``[k*units_per_shard, (k+1)*units_per_shard)`` except a
    possibly smaller final shard; every marker records its truthful range and
    resume is validated against chained markers.
    """
    requests = sorted({(site, layer) for site in sites for layer in layers})
    if not requests:
        raise ValueError("no extraction requests")
    rows_per_unit = len(requests)

    units: list[tuple[int, str]] = []
    for si in range(len(samples)):
        for sel_name in token_selectors:
            units.append((si, sel_name))
    total_units = len(units)
    if total_units == 0:
        raise ValueError("no work units")

    units_per_shard = max(1, int(shard_size) // rows_per_unit)

    cache_dir_path = Path(cache_dir)
    ensure_cache_identity(cache_dir_path, identity)

    prefix_shards, marker_resume_unit = _validated_resume_state(
        cache_dir_path, rows_per_unit
    )
    start_unit = marker_resume_unit if resume else 0
    if not resume and completed_shard_ids(cache_dir_path):
        raise RuntimeError("resume=False on a non-empty cache is forbidden")
    _prune_stale_shards(cache_dir_path, prefix_shards)
    if resume and start_unit > 0:
        logger.info(
            "resume: %d validated shard(s); continuing at work unit %d",
            prefix_shards, start_unit,
        )

    writer = ActivationShardWriter(cache_dir_path, shard_size=shard_size)
    writer.shard_count = prefix_shards

    started = _dt.datetime.now(_dt.timezone.utc)
    n_rows_before = sum(
        int(read_marker(cache_dir_path, sid)["n_rows"])
        for sid in range(prefix_shards)
    )

    batch_prompts: list[str] = []
    batch_indices: list[int] = []
    batch_metas: list[dict[str, Any]] = []
    units_processed = 0
    rows_written = 0
    shard_first_unit = start_unit

    def flush_now() -> None:
        nonlocal units_processed, rows_written
        if not batch_prompts:
            return
        result = adapter.extract(
            prompts=batch_prompts, requests=requests, token_indices=batch_indices
        )
        flat_requests = sorted(result.keys(), key=lambda r: (r[0], r[1]))
        for row_i in range(len(batch_prompts)):
            meta0 = batch_metas[row_i]
            for (site, layer) in flat_requests:
                vec = result[(site, layer)][row_i]
                meta = {
                    "sample_id": meta0["sample_id"],
                    "model_id": identity["model_id"],
                    "model_revision": identity["model_resolved_revision"] or "unresolved",
                    "tokenizer_revision": (
                        identity["tokenizer_resolved_revision"] or "unresolved"
                    ),
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
                rows_written += 1
        units_processed += len(batch_prompts)
        batch_prompts.clear()
        batch_indices.clear()
        batch_metas.clear()

    try:
        from tqdm import tqdm

        pbar: Any = tqdm(initial=start_unit, total=total_units, desc="extract",
                         unit="unit", dynamic_ncols=True)
    except ImportError:  # pragma: no cover
        pbar = None

    for ui in range(start_unit, total_units):
        rel_idx = ui - start_unit
        if rel_idx > 0 and rel_idx % units_per_shard == 0:
            # Shard boundary crossing at an exact deterministic work-unit
            # line: flush ALL pending work (all belongs to closing shard),
            # then close that shard with its truthful range.
            flush_now()
            writer.close_shard(shard_first_unit, ui)
            shard_first_unit = ui

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
        if len(batch_prompts) >= batch_size:
            flush_now()
        if pbar is not None:
            pbar.update(1)

    flush_now()
    if total_units > shard_first_unit:
        writer.close_shard(shard_first_unit, total_units)
    if pbar is not None:
        pbar.close()

    rows_total = sum(
        int(read_marker(cache_dir_path, sid)["n_rows"])
        for sid in completed_shard_ids(cache_dir_path)
    )
    elapsed = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()

    reader = ActivationCacheReader(cache_dir_path)
    index_df = reader.index()
    info = {
        "started_at": started.isoformat(),
        "elapsed_s": elapsed,
        "units_total": total_units,
        "units_processed_this_call": units_processed,
        "rows_before_this_call": int(n_rows_before),
        "rows_written_this_call": int(rows_written),
        "rows_cached_total": int(rows_total),
        "expected_rows_per_unit": len(requests),
        "units_per_shard": units_per_shard,
        "complete_shards": len(completed_shard_ids(cache_dir_path)),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }
    logger.info("extraction finished: %s", info)
    return index_df, info




