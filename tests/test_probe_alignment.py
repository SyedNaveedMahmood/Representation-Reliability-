"""Regression tests for activation/metadata row alignment across shards.

Bug under audit: ``tensor_key`` is the row position WITHIN a shard and repeats
across shards, but ``ShardedMatrixLoader.rows_for`` sorted the selected
metadata globally by ``tensor_key`` while concatenating activations
shard-major — pairing X rows with the wrong samples whenever >1 shard exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from representation_reliability.extraction.cache import ActivationShardWriter
from representation_reliability.runners.probe import (
    ShardedMatrixLoader,
    design_matrices_for,
)

N_SAMPLES = 12           # 6 facts x 2 twin labels
HIDDEN = 4
SELECTORS = ["last_prompt", "target_span_last"]
LAYER = 3


def _expected_vector(sample_number: int, selector_idx: int) -> np.ndarray:
    """Each vector encodes its source sample identity directly."""
    v = np.zeros(HIDDEN, dtype=np.float32)
    v[0] = float(sample_number)
    v[1] = float(sample_number * 10)
    v[2] = float(selector_idx)
    v[3] = 77.0
    return v


def _build_cache(tmp_path):
    """Write 3 physical shards whose rows interleave shards in the index."""
    rng = np.random.default_rng(0)
    vecs = {}
    metas = []
    w = ActivationShardWriter(tmp_path / "cache", shard_size=8)
    unit_rows = []
    for s in range(N_SAMPLES):
        for sel_i, selector in enumerate(SELECTORS):
            unit_rows.append((s, selector))
    # Insertion order: unit-major (sample, selector); each unit -> one row.
    for k, (s, selector) in enumerate(unit_rows):
        vec = _expected_vector(s, SELECTORS.index(selector)) + rng.normal(
            0, 1e-4, HIDDEN
        ).astype(np.float32)
        vecs[(s, selector)] = vec
        w.add(vec, {
            "sample_id": f"s{s:03d}",
            "site": "resid_post",
            "layer": LAYER,
            "token_selector": selector,
            "split": ("train" if s % 2 == 0 else "discovery_test"),
        })
        if (k + 1) % 8 == 0:
            w.close_shard(unit_start=k - 7, unit_end_exclusive=k + 1)
    if (k + 1) % 8 != 0:
        w.close_shard(unit_start=(k + 1) - ((k + 1) % 8),
                      unit_end_exclusive=k + 1)
    return vecs


def _fake_index(index_df):
    # mimic reader.index(): insertion order, shard column from grouping
    return index_df.reset_index(drop=True)


def test_multi_shard_rows_match_their_metadata(tmp_path):
    _build_cache(tmp_path)
    from representation_reliability.extraction.cache import ActivationCacheReader

    reader = ActivationCacheReader(tmp_path / "cache")
    index_df = reader.index()
    assert index_df["shard"].nunique() >= 3, "test requires >=3 physical shards"

    loader = ShardedMatrixLoader(tmp_path / "cache", index_df)
    label_of = {f"s{s:03d}": (1 if s % 4 < 2 else 0) for s in range(N_SAMPLES)}

    mats = design_matrices_for(
        loader, index_df, "last_prompt", LAYER, "resid_post",
        lambda sid: label_of[sid],
    )
    assert set(mats.keys()) == {"train", "discovery_test"}
    for split_name, block in mats.items():
        meta_ids = block["sample_ids"]
        X = block["X"]
        assert len(meta_ids) == X.shape[0]
        for i, sid in enumerate(meta_ids):
            expected = _expected_vector(int(sid[1:]), SELECTORS.index("last_prompt"))
            # tolerance: cached values equal written vectors bit-exactly (fp32 store)
            assert np.allclose(X[i], expected, atol=5e-4), (
                f"row {i} (sample {sid}) is paired with another sample's vector"
            )


def test_probe_path_verifies_shard_integrity(tmp_path):
    """A corrupted shard must fail loudly in the scientific probe path too."""
    _build_cache(tmp_path)
    from representation_reliability.extraction.cache import ActivationCacheReader

    reader = ActivationCacheReader(tmp_path / "cache")
    index_df = reader.index()
    # corrupt one shard's tensor file
    sid = index_df["shard"].unique()[1]
    d = tmp_path / "cache" / f"shard_{sid:05d}"
    tensor_path = d / "activations.safetensors"
    data = bytearray(tensor_path.read_bytes())
    data[120] ^= 0xFF
    tensor_path.write_bytes(bytes(data))

    loader = ShardedMatrixLoader(tmp_path / "cache", index_df)
    mask = (index_df["site"] == "resid_post") & (index_df["layer"] == LAYER) & (
        index_df["token_selector"] == "last_prompt"
    )
    with pytest.raises(OSError, match="integrity"):
        loader.rows_for(mask)


def test_exact_identity_validation_against_expected_samples(tmp_path):
    _build_cache(tmp_path)
    from representation_reliability.extraction.cache import ActivationCacheReader

    reader = ActivationCacheReader(tmp_path / "cache")
    index_df = reader.index()
    loader = ShardedMatrixLoader(tmp_path / "cache", index_df)

    expected = {
        "train": [f"s{s:03d}" for s in range(N_SAMPLES) if s % 2 == 0],
        "discovery_test": [f"s{s:03d}" for s in range(N_SAMPLES) if s % 2 == 1],
    }
    mats = design_matrices_for(
        loader, index_df, "last_prompt", LAYER, "resid_post",
        lambda sid: int(sid[1:]) % 2,
        expected_ids_by_split=expected,
    )
    for split_name, block in mats.items():
        assert sorted(block["sample_ids"]) == sorted(expected[split_name])
        assert len(set(block["sample_ids"])) == len(block["sample_ids"])

    bad = dict(expected)
    bad["train"] = expected["train"][:-1]          # a missing sample
    with pytest.raises(RuntimeError, match="missing"):
        design_matrices_for(
            loader, index_df, "last_prompt", LAYER, "resid_post",
            lambda sid: int(sid[1:]) % 2, expected_ids_by_split=bad,
        )




def test_two_sites_are_never_mixed(tmp_path):
    """site must be part of the design-matrix identity."""
    w = ActivationShardWriter(tmp_path / "cache", shard_size=4)
    rows = []
    for site_i, site in enumerate(["resid_post", "mlp_out"]):
        for s in range(4):
            vec = np.full(HIDDEN, 100.0 * site_i + s, dtype=np.float32)
            w.add(vec, {
                "sample_id": f"s{s}", "site": site, "layer": 1,
                "token_selector": "last_prompt", "split": "train",
            })
            rows.append((site, vec))
    w.close_shard(unit_start=0, unit_end_exclusive=8)
    from representation_reliability.extraction.cache import ActivationCacheReader

    reader = ActivationCacheReader(tmp_path / "cache")
    index_df = reader.index()
    loader = ShardedMatrixLoader(tmp_path / "cache", index_df)
    mats_a = design_matrices_for(
        loader, index_df, "last_prompt", 1, "resid_post",
        lambda sid: int(sid[1:]) % 2,
    )
    mats_b = design_matrices_for(
        loader, index_df, "last_prompt", 1, "mlp_out",
        lambda sid: int(sid[1:]) % 2,
    )
    xa = mats_a["train"]["X"]
    xb = mats_b["train"]["X"]
    assert xa.shape == xb.shape == (4, HIDDEN)
    # each site returns exactly its own vectors
    assert np.allclose(xa[:, 0], [0, 1, 2, 3])
    assert np.allclose(xb[:, 0], [100, 101, 102, 103])
