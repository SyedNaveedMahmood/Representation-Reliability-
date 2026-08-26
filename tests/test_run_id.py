from representation_reliability.runtime.run_id import (
    allocate_run_dir,
    make_run_id,
)


def test_run_id_deterministic():
    a = make_run_id("E00", "deadbeef" * 8, seed=20260827,
                    model_revision="main", dataset_split_hash="abc123")
    b = make_run_id("E00", "deadbeef" * 8, seed=20260827,
                    model_revision="main", dataset_split_hash="abc123")
    assert a == b


def test_run_id_changes_on_any_input_change():
    common = dict(seed=20260827, model_revision="main", dataset_split_hash="abc")
    variants = [
        make_run_id("E00", "h1", **common),
        make_run_id("E01", "h1", **common),
        make_run_id("E00", "h2", **common),
        make_run_id("E00", "h1", seed=999, **{k: v for k, v in common.items() if k != "seed"}),
        make_run_id("E00", "h1", **{**common, "dataset_split_hash": "zzz"}),
    ]
    assert len(set(variants)) == len(variants)


def test_allocate_run_dir_never_clobbers(tmp_path):
    d1 = allocate_run_dir(tmp_path, "E00", "runA")
    d1.mkdir(parents=True)
    d2 = allocate_run_dir(tmp_path, "E00", "runA")
    assert d2 != d1 and d2.name.endswith("-r2") or not d2.exists()
