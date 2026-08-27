import numpy as np
import pandas as pd
import pytest

from representation_reliability.data.splits import (
    ConfirmationSplitAccessError,
    build_discovery_label_map,
)
from representation_reliability.probes.linear import randomized_control_labels


def test_randomized_control_labels_independent_and_deterministic():
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 2, size=100)
    y_val = rng.integers(0, 2, size=50)

    a_tr, a_val, seeds_a = randomized_control_labels(y_train, y_val, seed=5)
    b_tr, b_val, seeds_b = randomized_control_labels(y_train, y_val, seed=5)
    c_tr, c_val, _ = randomized_control_labels(y_train, y_val, seed=6)

    # determinism for the same control seed
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_val, b_val)
    assert seeds_a == seeds_b
    # different control seed -> different permutations on BOTH splits
    assert not np.array_equal(a_tr, c_tr) or not np.array_equal(a_val, c_val)
    assert seeds_a != (0, 0) and len(set(seeds_a)) == 2

    # both splits are true permutations of their originals (multiset preserved)
    assert sorted(a_tr.tolist()) == sorted(y_train.tolist())
    assert sorted(a_val.tolist()) == sorted(y_val.tolist())
    # train and validation permutations are not the trivially identical stream
    tr_rng = np.random.default_rng(seeds_a[0])
    val_rng = np.random.default_rng(seeds_a[1])
    assert list(tr_rng.permutation(10)) != list(val_rng.permutation(10))


def test_discovery_label_map_refuses_confirmation_ids():
    df = pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "split": ["train", "discovery_test", "confirmation"],
        "target_label": [1, 0, 1],
    })
    # Building the map over a frame containing confirmation rows raises:
    with pytest.raises(ConfirmationSplitAccessError):
        build_discovery_label_map(df)
    # The guard fires even if only one confirmation id is requested.
    conf_only = pd.DataFrame({
        "sample_id": ["c"], "split": ["confirmation"], "target_label": [1],
    })
    with pytest.raises(ConfirmationSplitAccessError):
        build_discovery_label_map(conf_only)


def test_validate_splits_does_not_observe_confirmation_labels():
    df = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "pair_id": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
        "split": ["train", "train",
                  "validation", "validation",
                  "discovery_test", "discovery_test",
                  "confirmation", "confirmation"],
        # confirmation rows carry IMPOSSIBLE labels; validation must ignore them
        "target_label": [1, 0, 0, 1, 1, 0, 7, -5],
    })
    from representation_reliability.data.splits import validate_splits

    summary = validate_splits(df)
    assert summary["confirmation"]["labels_observed"] is False
    assert summary["confirmation"]["labels_present"] is None