import numpy as np
import pandas as pd
import pytest

from representation_reliability.data.splits import (
    ConfirmationSplitAccessError,
    apply_splits,
    assign_group_splits,
    discovery_view,
    require_confirmation_access,
    validate_splits,
)


def _make_df(n_groups=40):
    rows = []
    for g in range(n_groups):
        for label in (0, 1):
            rows.append({
                "sample_id": f"g{g}-{label}",
                "pair_id": f"group-{g:03d}",
                "target_label": label,
                "prompt": f"premise {g} label {label}",
            })
    return pd.DataFrame(rows)


def test_group_split_proportions_and_determinism():
    df = _make_df()
    a1 = assign_group_splits(df["pair_id"].tolist(), seed=5)
    a2 = assign_group_splits(df["pair_id"].tolist(), seed=5)
    assert a1 == a2
    split_df = apply_splits(df, a1)
    summary = validate_splits(split_df)   # raises on degenerate splits
    # each group stays whole
    cross = split_df.groupby("pair_id")["split"].nunique()
    assert (cross == 1).all()
    total = len(a1)
    fr = {k: v / total for k, v in
          pd.Series(list(a1.values())).value_counts().items()}
    assert abs(fr["train"] - 0.6) < 0.12
    assert fr["confirmation"] > 0


def test_confirmation_isolation():
    df = _make_df()
    assignment = assign_group_splits(df["pair_id"], seed=3)
    split_df = apply_splits(df, assignment)
    view = discovery_view(split_df)
    assert "confirmation" not in set(view["split"])
    assert len(view) < len(split_df)
    require_confirmation_access("confirmatory_evaluation")   # allowed explicitly
    with pytest.raises(ConfirmationSplitAccessError):
        require_confirmation_access("just curious")


def test_fraction_validation():
    bad = {"train": 0.6, "validation": 0.15, "discovery_test": 0.15,
           "confirmation": -0.1}
    with pytest.raises(ValueError):
        assign_group_splits(["a", "b", "c", "d", "e"], fractions=bad)


def test_classes_present_in_every_learning_split():
    df = _make_df()
    split_df = apply_splits(df, assign_group_splits(df["pair_id"], seed=11))
    summary = validate_splits(split_df)
    for name in ("train", "validation", "discovery_test"):
        assert 0 in summary[name]["labels_present"]
        assert 1 in summary[name]["labels_present"]
