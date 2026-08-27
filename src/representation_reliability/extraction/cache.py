"""Portable, atomic, resumable activation cache (schema v2).

Layout::

    <cache_dir>/
        cache_identity.json        # strict semantic identity; reuse refused on mismatch
        shard_00000/
            activations.safetensors   # [rows, hidden] float32 tensor rows
            meta.parquet              # one row per stored activation
            _complete.json            # n_rows, sha256(tensor), unit range

Contracts enforced here:

- *Order*: ``ActivationCacheReader.load_rows(requested)`` returns a matrix whose
  row ``i`` corresponds EXACTLY to ``requested.iloc[i]``, in the caller's own
  row order (``tensor_key`` is shard-local and never drives global order).
- *Integrity*: every touched shard is sha256-checked against its marker at
  least once per reader instance; the validated tensor is cached afterwards.
- *Identity*: shard markers carry the work-unit range they hold, so resume is
  derived from recorded fact, not arithmetic assumptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


CACHE_IDENTITY_FILE = "cache_identity.json"
REQUIRED_MARKER_KEYS = ("unit_start", "unit_end_exclusive")


def shard_dir(cache_dir: Path, shard_idx: int) -> Path:
    return Path(cache_dir) / f"shard_{shard_idx:05d}"


class LegacyCacheError(RuntimeError):
    """Raised when a pre-v2 cache directory cannot be safely interpreted."""


class CacheIdentityMismatchError(RuntimeError):
    """Raised when a cache's semantic identity disagrees with the request."""


def shard_is_complete(cache_dir: Path, shard_idx: int) -> bool:
    d = shard_dir(Path(cache_dir), shard_idx)
    marker_path = d / "_complete.json"
    if not marker_path.exists():
        return False
    try:
        with marker_path.open("r", encoding="utf-8") as fh:
            info = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False
    if not all(k in info for k in REQUIRED_MARKER_KEYS):
        return False   # pre-v2 marker without unit range -> not interpretable
    return (
        (d / "activations.safetensors").exists()
        and (d / "meta.parquet").exists()
        and int(info.get("n_rows", -1)) >= 0
        and int(info["unit_end_exclusive"]) >= int(info["unit_start"])
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


def read_marker(cache_dir: str | Path, shard_idx: int) -> dict[str, Any]:
    with (shard_dir(Path(cache_dir), shard_idx) / "_complete.json").open(
        encoding="utf-8"
    ) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Cache identity (schema v2)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def ensure_cache_identity(cache_dir: str | Path, expected: dict[str, Any]) -> dict:
    """Validate (or create) the cache semantic-identity manifest.

    Reuse is REFUSED on:
    - identity mismatch with what the current run expects;
    - a legacy cache (completed shards but no identity manifest), which cannot
      prove semantic compatibility.
    """
    root = Path(cache_dir)
    path = root / CACHE_IDENTITY_FILE
    has_shards = any(root.glob("shard_*")) if root.exists() else False
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored != expected:
            diffs = {
                k: {"stored": stored.get(k), "expected": expected.get(k)}
                for k in sorted(set(stored) | set(expected))
                if stored.get(k) != expected.get(k)
            }
            raise CacheIdentityMismatchError(
                f"cache identity mismatch in {root}; refusing reuse: {diffs}"
            )
        return stored
    if has_shards:
        raise LegacyCacheError(
            f"{root} contains pre-schema-v2 shards without a cache identity "
            "manifest; refusing silent adoption. Use a fresh cache location."
        )
    atomic_write_json(path, expected)
    return expected


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(payload))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise



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

    def close_shard(
        self,
        unit_start: int,
        unit_end_exclusive: int,
    ) -> Path | None:
        """Atomically write the current buffer as the next shard.

        The marker records the exact work-unit range held by this shard so
        that resume derives from recorded fact, never arithmetic assumptions.
        """
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
            "unit_start": int(unit_start),
            "unit_end_exclusive": int(unit_end_exclusive),
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
    """Reads completed shards of a cache as one logical table.

    Order contract: ``load_rows(requested)[i]`` corresponds EXACTLY to
    ``requested.iloc[i]`` in the caller's given row order. Shard digests are
    verified once per reader instance and the validated tensor is cached.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)
        self._verified_shards: dict[int, np.ndarray] = {}

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

    def _verified_shard(self, sid: int) -> np.ndarray:
        """Load a shard once; verify its sha256 against the marker first."""
        if sid in self._verified_shards:
            return self._verified_shards[sid]
        from safetensors.numpy import load_file

        d = shard_dir(self.root, sid)
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
        data = np.asarray(load_file(str(tensor_path))["activations"], dtype=np.float32)
        meta_rows = len(pd.read_parquet(d / "meta.parquet"))
        if not (data.shape[0] == int(marker["n_rows"]) == meta_rows):
            raise OSError(
                f"activation shard {sid} inconsistent: tensor rows "
                f"{data.shape[0]}, marker {marker['n_rows']}, meta {meta_rows}"
            )
        self._verified_shards[sid] = data
        return data

    def load_rows(self, requested: pd.DataFrame) -> np.ndarray:
        """Return activations for ``requested`` rows IN THE SAME ORDER.

        Row ``i`` of the output belongs to metadata row ``i`` of ``requested``,
        regardless of how rows are interleaved across shards.
        """
        n = len(requested)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)
        req = requested.copy()
        req["_req_pos"] = np.arange(n)
        out: np.ndarray | None = None
        for sid, group in req.groupby("shard", sort=True):
            arr = self._verified_shard(int(sid))
            if out is None:
                out = np.zeros((n, arr.shape[1]), dtype=np.float32)
            pos = group["_req_pos"].to_numpy()
            keys = group["tensor_key"].to_numpy().astype(int)
            if keys.min() < 0 or keys.max() >= arr.shape[0]:
                raise IndexError(f"tensor_key out of range in shard {sid}")
            out[pos] = arr[keys]
        assert out is not None
        return out

    def verify_reload_consistency(self) -> dict[int, dict[str, Any]]:
        """Diagnostic: a second independent load matches the validated load."""
        results: dict[int, dict[str, Any]] = {}
        for sid in completed_shard_ids(self.root):
            d = shard_dir(self.root, sid)
            from safetensors.numpy import load_file

            t1 = self._verified_shard(sid)
            t2 = load_file(str(d / "activations.safetensors"))["activations"]
            meta = pd.read_parquet(d / "meta.parquet")
            marker = read_marker(self.root, sid)
            max_dev = (
                float(np.max(np.abs(t1.astype(np.float64) - t2.astype(np.float64))))
                if t1.size else 0.0
            )
            results[sid] = {
                "rows_tensor": int(t1.shape[0]),
                "rows_meta": len(meta),
                "rows_marker": int(marker["n_rows"]),
                "repeat_load_max_abs_dev": max_dev,
                "aligned": (
                    t1.shape[0] == len(meta) == int(marker["n_rows"])
                    and max_dev == 0.0
                ),
            }
        return results

