"""Adversarial resume & cache-identity tests with a deterministic fake adapter.

Identity property tested: a resumed cache must be logically identical to a
fresh clean extraction — same work units, sample IDs, selectors, sites,
layers, vectors and row cardinality — under batch-size/shard-geometry regimes
where batch boundaries cross declared shard boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from representation_reliability.contracts import Sample
from representation_reliability.extraction.activations import (
    build_cache_identity,
    dataset_content_hash,
    extract_dataset_activations,
)
from representation_reliability.extraction.cache import (
    CacheIdentityMismatchError,
    completed_shard_ids,
    read_marker,
)

N_LAYERS = 2
HIDDEN = 4
SITES = ["resid_post"]
SELECTORS = ["last_prompt", "target_span_last"]
N_SAMPLES = 25


class _FakeSiteResolution:
    def __init__(self, name: str) -> None:
        self.native_module_name = name


class FakeAdapter:
    """Deterministic: vector = f(sample number, site, layer, resolved index).

    The activation extractor only receives ``(prompts, requests,
    token_indices)``, so the fake parses the sample number out of each prompt;
    ``last_prompt`` and ``target_span_last`` resolve to different indices by
    construction, which the logical-state comparison validates exactly.
    """

    model_id = "fake/model"
    num_layers = N_LAYERS
    hidden_size = HIDDEN

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def resolve_site(self, site: str, layer: int):
        return _FakeSiteResolution(f"fake.{site}.{layer}")

    def extract(self, prompts, requests, token_indices):
        out: dict[tuple[str, int], np.ndarray] = {}
        layers_in_requests = sorted({l for (_s, l) in requests})
        sites_in_requests = sorted({s for (s, _l) in requests})
        for site in sites_in_requests:
            for layer in layers_in_requests:
                rows = np.zeros((len(prompts), HIDDEN), dtype=np.float32)
                for i, prompt in enumerate(prompts):
                    s_idx = int(prompt.split("sample=")[1].split(";")[0])
                    v = np.zeros(HIDDEN, dtype=np.float32)
                    v[0] = float(s_idx)
                    v[1] = float(s_idx * 10)
                    v[2] = float(layer)
                    v[3] = float(token_indices[i]) / 1000.0
                    rows[i] = v
                out[(site, layer)] = rows
        return out


def _make_samples():
    samples = []
    for s in range(N_SAMPLES):
        word = "north" if s % 2 == 0 else "south"
        question = f"Is A {word} B?"
        prompt = (
            f"Premise: sample={s}; A is north of B.\n"
            f"Question: {question}\nAnswer:"
        )
        samples.append(Sample(
            sample_id=f"s{s:03d}",
            prompt=prompt,
            target_label=s % 2,
            task_name="synthetic_relations",
            metadata={"target_text": question, "chat_template_used": False},
        ))
    return samples


def _split_of(sid: str) -> str:
    n = int(sid[1:])
    return ["train", "validation", "discovery_test"][n % 3]


def _identity(samples):
    return build_cache_identity(
        experiment_id="E00TEST",
        samples=samples,
        model_id=FakeAdapter.model_id,
        model_resolved_revision=None,
        tokenizer_id=FakeAdapter.model_id,
        tokenizer_resolved_revision=None,
        sites=SITES,
        layers=list(range(N_LAYERS)),
        token_selectors=SELECTORS,
        model_dtype="float32",
    )


def _run_extract(cache_dir, adapter, samples, *, batch_size, shard_size=18,
                 resume=True):
    return extract_dataset_activations(
        adapter, samples, cache_dir,
        sites=SITES, layers=list(range(N_LAYERS)), token_selectors=SELECTORS,
        shard_size=shard_size, batch_size=batch_size, split_of=_split_of,
        identity=_identity(samples), resume=resume,
    )


def _logical_state(cache_dir):
    """Full (sample_id, selector, site, layer) -> vector mapping from disk."""
    from safetensors.numpy import load_file

    records = []
    for sid in completed_shard_ids(cache_dir):
        d = cache_dir / f"shard_{sid:05d}"
        meta = pd.read_parquet(d / "meta.parquet")
        arr = load_file(str(d / "activations.safetensors"))["activations"]
        assert len(meta) == arr.shape[0]
        for r in range(len(meta)):
            m = meta.iloc[r]
            records.append({
                "sample_id": str(m["sample_id"]),
                "token_selector": str(m["token_selector"]),
                "site": str(m["site"]),
                "layer": int(m["layer"]),
                "token_index": int(m["token_index"]),
                "vector": arr[r].tolist(),
            })
    return pd.DataFrame(records).sort_values(
        ["sample_id", "token_selector", "site", "layer"]
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def fake_adapter():
    return FakeAdapter()


def _assert_same_state(a: pd.DataFrame, b: pd.DataFrame):
    assert len(a) == len(b), f"row cardinality differs: {len(a)} vs {len(b)}"
    cols = ["sample_id", "token_selector", "site", "layer", "token_index"]
    assert a[cols].equals(b[cols]), "metadata identity differs"
    va = np.stack(a["vector"].tolist())
    vb = np.stack(b["vector"].tolist())
    assert np.array_equal(va, vb), "vectors differ between resumed and clean cache"


@pytest.mark.parametrize("batch_size", [8, 16], ids=["bs8_lt_units9", "bs16_gt_units9"])
def test_resume_identity_deleted_middle_shard(tmp_path, fake_adapter, batch_size):
    """Cases A/B/D: batch < and > units_per_shard; deleted middle shard."""
    samples = _make_samples()
    # units_per_shard = 18 // rows_per_unit(=2 sites*2 layers) = 9
    clean_dir = tmp_path / "clean"
    _run_extract(clean_dir, fake_adapter, samples, batch_size=batch_size)
    clean = _logical_state(clean_dir)
    assert len(clean) == N_SAMPLES * len(SELECTORS) * N_LAYERS

    resumed_dir = tmp_path / f"resumed_bs{batch_size}"
    _run_extract(resumed_dir, fake_adapter, samples, batch_size=batch_size)

    markers = [read_marker(resumed_dir, s) for s in completed_shard_ids(resumed_dir)]
    ranges = [(m["unit_start"], m["unit_end_exclusive"]) for m in markers]
    for (_, prev_end), (start, _) in zip(ranges, ranges[1:]):
        assert start == prev_end, "shard unit ranges must chain exactly"

    # Delete a MIDDLE shard, resume; must be identical to clean state.
    middle = sorted(completed_shard_ids(resumed_dir))[len(ranges) // 2]
    import shutil

    shutil.rmtree(resumed_dir / f"shard_{middle:05d}")
    _run_extract(resumed_dir, fake_adapter, samples, batch_size=batch_size)
    _assert_same_state(clean, _logical_state(resumed_dir))


def test_partial_final_shard_then_resume(tmp_path, fake_adapter):
    """Case C: non-divisible final shard; deleting it and resuming stays exact."""
    samples = _make_samples()
    d1, d2 = tmp_path / "c1", tmp_path / "c2"
    _, info = _run_extract(d1, fake_adapter, samples, batch_size=5)
    total_units = info["units_total"]
    ids = sorted(completed_shard_ids(d1))
    last_marker = read_marker(d1, ids[-1])
    assert last_marker["unit_end_exclusive"] == total_units
    assert last_marker["unit_end_exclusive"] - last_marker["unit_start"] < 9 \
        or last_marker["unit_start"] == 0

    import shutil

    shutil.rmtree(d2 / f"shard_{ids[-1]:05d}") if False else None
    # build a second cache identical to the first...
    _run_extract(d2, fake_adapter, samples, batch_size=5)
    # ...delete its final partial shard...
    shutil.rmtree(d2 / f"shard_{ids[-1]:05d}")
    # ...and resume; result must equal the untouched clean cache.
    _run_extract(d2, fake_adapter, samples, batch_size=5)
    _assert_same_state(_logical_state(d1), _logical_state(d2))


def test_cache_identity_mismatch_refuses_reuse(tmp_path, fake_adapter):
    samples = _make_samples()
    d = tmp_path / "cache"
    _run_extract(d, fake_adapter, samples, batch_size=8)
    other = dict(_identity(samples))
    other["dataset_content_hash"] = "0" * 64
    with pytest.raises(CacheIdentityMismatchError):
        extract_dataset_activations(
            fake_adapter, samples, d,
            sites=SITES, layers=list(range(N_LAYERS)),
            token_selectors=SELECTORS, shard_size=18, batch_size=8,
            split_of=_split_of, identity=other, resume=True,
        )


def test_legacy_cache_without_identity_never_adopted(tmp_path, fake_adapter):
    """A pre-v2 directory (shards but no identity manifest) must refuse reuse."""
    samples = _make_samples()
    d = tmp_path / "legacy"
    _run_extract(d, fake_adapter, samples, batch_size=8)
    (d / "cache_identity.json").unlink()
    from representation_reliability.extraction.cache import LegacyCacheError

    with pytest.raises(LegacyCacheError):
        _run_extract(d, fake_adapter, samples, batch_size=8)


def test_identity_covers_dataset_content_and_geometry():
    s = _make_samples()
    i1 = _identity(s)
    mutated = list(s)
    from dataclasses import replace as dc_replace

    mutated[0] = dc_replace(mutated[0], prompt="changed prompt")
    i2 = _identity(mutated)
    assert i1["dataset_content_hash"] != i2["dataset_content_hash"]
    assert dataset_content_hash(s) == dataset_content_hash(list(s))

