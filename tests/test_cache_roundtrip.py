import json

import numpy as np
import pandas as pd
import pytest

from representation_reliability.extraction.cache import (
    ActivationCacheReader,
    ActivationShardWriter,
    completed_shard_ids,
    shard_is_complete,
)


def test_roundtrip_numeric_equality_and_alignment(tmp_path):
    writer = ActivationShardWriter(tmp_path / "cache", shard_size=4)
    rng = np.random.default_rng(0)
    metas = []
    vecs = []
    for i in range(10):
        v = rng.standard_normal(16).astype(np.float32)
        meta = {
            "sample_id": f"s{i}", "site": "resid_post", "layer": i % 3,
            "token_selector": "last_prompt", "token_index": 5,
            "token_id": 7, "token_text": "x",
            "split": ["train", "validation"][i % 2],
            "prompt_hash": f"ph{i}",
        }
        writer.add(v, meta)
        vecs.append(v)
        metas.append(meta)

    # Deterministic unit ranges; here 1 unit == 1 row for the test scaffold.
    writer.close_shard(unit_start=0, unit_end_exclusive=10)
    assert completed_shard_ids(tmp_path / "cache")

    reader = ActivationCacheReader(tmp_path / "cache")
    index_df = reader.index()
    assert len(index_df) == 10
    # sample id / metadata alignment in stored order
    assert index_df["sample_id"].tolist() == [m["sample_id"] for m in metas]
    X = reader.load_rows(index_df)
    assert np.allclose(X, np.stack(vecs), atol=1e-6)

    diag = reader.verify_reload_consistency()
    for sid, info in diag.items():
        assert info["aligned"], sid
        assert info["repeat_load_max_abs_dev"] == 0.0


def test_incomplete_shard_is_not_complete(tmp_path):
    cache_dir = tmp_path / "cache"
    w = ActivationShardWriter(cache_dir, shard_size=2)
    w.add(np.zeros(4, dtype=np.float32), {"sample_id": "a"})
    w.close_shard(unit_start=0, unit_end_exclusive=1)
    # remove marker -> must no longer count as complete
    shard0 = cache_dir / "shard_00000"
    (shard0 / "_complete.json").unlink()
    assert not shard_is_complete(cache_dir, 0)
    assert completed_shard_ids(cache_dir) == []


def test_pre_v2_marker_without_unit_range_not_interpretable(tmp_path):
    """Old Phase-0A markers (no unit range) must never count as complete."""
    cache_dir = tmp_path / "cache"
    w = ActivationShardWriter(cache_dir, shard_size=2)
    w.add(np.ones(4, dtype=np.float32), {"sample_id": "a"})
    w.add(np.zeros(4, dtype=np.float32), {"sample_id": "b"})
    w.close_shard(unit_start=0, unit_end_exclusive=2)
    marker_path = cache_dir / "shard_00000" / "_complete.json"
    marker = json.loads(marker_path.read_text())
    for key in ("unit_start", "unit_end_exclusive"):
        del marker[key]
    marker_path.write_text(json.dumps(marker))
    assert completed_shard_ids(cache_dir) == []


def test_corrupt_shard_detected_on_read(tmp_path):
    cache_dir = tmp_path / "cache"
    w = ActivationShardWriter(cache_dir, shard_size=2)
    w.add(np.ones(4, dtype=np.float32), {"sample_id": "a"})
    w.add(np.zeros(4, dtype=np.float32), {"sample_id": "b"})
    w.close_shard(unit_start=0, unit_end_exclusive=2)   # explicit flush is the contract now
    tensor_path = cache_dir / "shard_00000" / "activations.safetensors"
    data = bytearray(tensor_path.read_bytes())
    data[100] ^= 0xFF   # flip some bytes
    tensor_path.write_bytes(bytes(data))
    reader = ActivationCacheReader(cache_dir)
    idx = reader.index()
    with pytest.raises(IOError):
        reader.load_rows(idx)


def test_writer_dtype_recorded(tmp_path):
    w = ActivationShardWriter(tmp_path / "c", shard_size=8)
    w.add(np.arange(4, dtype=np.float64), {"sample_id": "s"})
    w.close_shard(unit_start=0, unit_end_exclusive=1)
    df = pd.read_parquet(tmp_path / "c" / "shard_00000" / "meta.parquet")
    assert (df["dtype"] == "float32").all()


def test_true_source_vector_roundtrip_multi_shard(tmp_path):
    """Write -> disk -> logical-reader round trip across several shards.

    Compares loaded vectors against the ORIGINAL source vectors (per sample
    id), which is what 'round trip' must mean — not a second read of the
    same file.
    """
    rng = np.random.default_rng(3)
    cache_dir = tmp_path / "cache"
    w = ActivationShardWriter(cache_dir, shard_size=3)
    sid_to_source: dict[str, np.ndarray] = {}
    unit_cursor = 0
    for i in range(11):                     # forces a partial final shard
        v = rng.standard_normal(8).astype(np.float32)
        sid = f"s{i}"
        sid_to_source[sid] = v
        w.add(v, {"sample_id": sid, "site": "resid_post", "layer": 1,
                  "token_selector": "last_prompt", "split": "train"})
        if (i + 1) % 4 == 0:                # scaffold: 1 unit == 1 row
            w.close_shard(unit_start=unit_cursor, unit_end_exclusive=i + 1)
            unit_cursor = i + 1
    w.close_shard(unit_start=unit_cursor, unit_end_exclusive=11)

    reader = ActivationCacheReader(cache_dir)
    index_df = reader.index()
    assert index_df["shard"].nunique() >= 3
    # Request rows in an order that interleaves shards.
    shuffled = index_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    X = reader.load_rows(shuffled)
    expected = np.stack([sid_to_source[sid] for sid in shuffled["sample_id"]])
    assert X.shape == expected.shape
    assert np.array_equal(X, expected)      # bit-exact fp32 round trip
