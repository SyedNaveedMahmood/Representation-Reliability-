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

    # Trailing partial shard flush (orchestrators always do this).
    w = writer.flush()
    assert w is not None

    ids = completed_shard_ids(tmp_path / "cache")
    assert len(ids) >= 1

    reader = ActivationCacheReader(tmp_path / "cache")
    index_df = reader.index()
    assert len(index_df) == 10
    # sample id / metadata alignment
    assert index_df["sample_id"].tolist() == [m["sample_id"] for m in metas]
    X = reader.load_rows(index_df.sort_values("tensor_key"))
    assert np.allclose(X, np.stack(vecs), atol=1e-6)

    diag = reader.verify_alignment()
    for sid, info in diag.items():
        assert info["aligned"], sid
        assert info["roundtrip_max_abs_dev"] == 0.0


def test_incomplete_shard_is_not_complete(tmp_path):
    cache_dir = tmp_path / "cache"
    w = ActivationShardWriter(cache_dir, shard_size=2)
    w.add(np.zeros(4, dtype=np.float32), {"sample_id": "a"})
    w.flush()
    # remove marker -> must no longer count as complete
    shard0 = cache_dir / "shard_00000"
    (shard0 / "_complete.json").unlink()
    assert not shard_is_complete(cache_dir, 0)
    assert completed_shard_ids(cache_dir) == []


def test_corrupt_shard_detected_on_read(tmp_path):
    cache_dir = tmp_path / "cache"
    w = ActivationShardWriter(cache_dir, shard_size=2)
    w.add(np.ones(4, dtype=np.float32), {"sample_id": "a"})
    w.add(np.zeros(4, dtype=np.float32), {"sample_id": "b"})
    w.flush()   # explicit flush is the contract now
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
    w.flush()
    df = pd.read_parquet(tmp_path / "c" / "shard_00000" / "meta.parquet")
    assert (df["dtype"] == "float32").all()
