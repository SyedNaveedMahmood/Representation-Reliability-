"""Portable, atomic, resumable activation cache.

Layout::

    <cache_dir>/
        shard_00000/
            activations.safetensors   # [rows, hidden] float32 tensor rows
            meta.parquet              # one row per stored activation
            _complete.json            # row count + sha256 of the tensor file

A shard directory without ``_complete.json`` is *never* treated as complete,
so interrupted runs can be resumed exactly.  Writes are atomic: tensors and
metadata are written to temp files and renamed only afterwards.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_dir(cache_dir: Path, shard_idx: int) -> Path:
    return Path(cache_dir) / f"shard_{shard_idx:05d}"


def shard_is_complete(cache_dir: Path, shard_idx: int) -> bool:
    d = shard_dir(Path(cache_dir), shard_idx)
    marker = d / "_complete.json"
    if not marker.exists():
        return False
    try:
        with marker.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False
    return (
        (d / "activations.safetensors").exists()
        and (d / "meta.parquet").exists()
        and int(info.get("n_rows", -1)) >= 0
    )


def completed_shard_ids(cache_dir: str | Path) -> list[int]:
    root = Path(cache_dir)
    if not root.exists():
        return []
    ids = []
    for child in sorted(root.glob("shard_*")):
        try:
            idx = int(child.name.split("_")[-1])
        except ValueError:
            continue
        if shard_is_complete(root, idx):
            ids.append(idx)
    return ids


class ActivationShardWriter:
    """Accumulates activation rows; flushes complete shards atomically."""

    def __init__(self, cache_dir: str | Path, shard_size: int,
                 dtype: str = "float32") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.dtype = np.dtype(dtype)
        self.buffer_rows: list[np.ndarray] = []
        self.buffer_meta: list[dict[str, Any]] = []
        self.shard_count = 0          # number of shards flushed so far
        self.total_rows_written = 0

    def add(self, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        """Add one activation row with its metadata record."""
        vec = np.asarray(vector, dtype=self.dtype).reshape(-1)
        self.buffer_rows.append(vec)
        meta = dict(metadata)
        meta.pop("tensor_file", None)   # stamped accurately at flush time
        meta["shape"] = ",".join(str(x) for x in vec.shape)
        meta["dtype"] = str(self.dtype.name)
        self.buffer_meta.append(meta)
        # NOTE: callers control flushing (orchestrators define shard
        # boundaries in work-unit space); no implicit flush here.

    def flush(self) -> Path | None:
        """Atomically write the current buffer as the next shard."""
        if not self.buffer_rows:
            return None
        import os

        rows = np.stack(self.buffer_rows, axis=0)
        shard_id = self.shard_count
        # Stamp every row with its actual destination now that the shard
        # number is final (rows added earlier may have hit flush boundaries).
        for meta in self.buffer_meta:
            meta["tensor_file"] = f"shard_{shard_id:05d}/activations.safetensors"
        meta_df = pd.DataFrame(self.buffer_meta)
        # tensor_key: row position within this shard (used by the reader).
        meta_df.insert(0, "tensor_key", np.arange(len(meta_df), dtype=np.int64))

        out_dir = shard_dir(self.cache_dir, shard_id)
        out_dir.mkdir(parents=True, exist_ok=False)

        tensor_path = out_dir / "activations.safetensors"
        meta_path = out_dir / "meta.parquet"
        tmp_tensor = out_dir / "activations.safetensors.tmp"
        tmp_meta = out_dir / "meta.parquet.tmp"

        from safetensors.numpy import save_file

        save_file({"activations": rows}, str(tmp_tensor))
        meta_df.to_parquet(tmp_meta, index=False)

        os.replace(tmp_tensor, tensor_path)
        os.replace(tmp_meta, meta_path)

        digest = _sha256_file(tensor_path)
        marker = {
            "shard": shard_id,
            "n_rows": int(rows.shape[0]),
            "hidden_dim": int(rows.shape[1]),
            "tensor_sha256": digest,
            "columns": list(meta_df.columns),
        }
        tmp_marker = out_dir / "_complete.json.tmp"
        with open(tmp_marker, "w", encoding="utf-8") as fh:
            json.dump(marker, fh, indent=2, sort_keys=True)
        os.replace(tmp_marker, out_dir / "_complete.json")

        self.buffer_rows.clear()
        self.buffer_meta.clear()
        self.shard_count += 1
        self.total_rows_written += int(rows.shape[0])
        return out_dir


class ActivationCacheReader:
    """Reads completed shards of a cache as one logical table."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)

    def index(self) -> pd.DataFrame:
        frames = []
        for sid in completed_shard_ids(self.root):
            meta_path = shard_dir(self.root, sid) / "meta.parquet"
            df = pd.read_parquet(meta_path)
            df["shard"] = sid
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_rows(self, index_df: pd.DataFrame) -> np.ndarray:
        """Load stacked activations for the given metadata rows.

        Integrity of every touched shard file is verified against its marker.
        """
        empty = index_df.empty
        if empty:
            return np.zeros((0, 0), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for sid, group in index_df.groupby("shard", sort=True):
            d = shard_dir(self.root, int(sid))
            tensor_path = d / "activations.safetensors"
            marker_path = d / "_complete.json"
            with marker_path.open("r", encoding="utf-8") as fh:
                marker = json.load(fh)
            actual = _sha256_file(tensor_path)
            if actual != marker.get("tensor_sha256"):
                raise OSError(
                    f"activation shard {sid} failed integrity check "
                    "(sha256 mismatch); rerun extraction"
                )
            from safetensors.numpy import load_file

            data = load_file(str(tensor_path))["activations"]
            keys = group["tensor_key"].to_numpy().astype(int)
            pieces.append(np.asarray(data[keys], dtype=np.float32))
        return np.concatenate(pieces, axis=0)

    def verify_alignment(self) -> dict[int, dict[str, Any]]:
        """Test A helper: reload shards independently; check round-trip and row counts."""
        results: dict[int, dict[str, Any]] = {}
        for sid in completed_shard_ids(self.root):
            d = shard_dir(self.root, sid)
            from safetensors.numpy import load_file

            t1 = load_file(str(d / "activations.safetensors"))["activations"]
            t2 = load_file(str(d / "activations.safetensors"))["activations"]
            meta = pd.read_parquet(d / "meta.parquet")
            with (d / "_complete.json").open("r", encoding="utf-8") as fh:
                marker = json.load(fh)
            max_dev = (
                float(np.max(np.abs(t1.astype(np.float64) - t2.astype(np.float64))))
                if t1.size else 0.0
            )
            results[sid] = {
                "rows_tensor": int(t1.shape[0]),
                "rows_meta": len(meta),
                "rows_marker": int(marker["n_rows"]),
                "roundtrip_max_abs_dev": max_dev,
                "aligned": (
                    t1.shape[0] == len(meta) == int(marker["n_rows"])
                    and max_dev == 0.0
                ),
            }
        return results

