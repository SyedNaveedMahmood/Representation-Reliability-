import numpy as np
import pandas as pd
import pytest

from representation_reliability.data.splits import build_discovery_label_map
from representation_reliability.runners.e00c_followup import (
    label_lookup_from_mapping,
    signed_cosine,
)
from representation_reliability.runners.probe import design_matrices_for


def test_signed_cosine_preserves_orientation():
    a = np.array([1.0, 0.0, 0.0])
    assert signed_cosine(a, a) == 1.0
    assert signed_cosine(a, -a) == -1.0


def test_rms_positive_scaling_does_not_preserve_cross_example_ranking():
    # A positive, sample-specific denominator preserves each score's sign but
    # can reverse ranking across examples. This is why raw-residual cosine is
    # not an exact readout-geometry diagnostic under RMSNorm.
    numerators = np.array([2.0, 3.0])
    rms = np.array([1.0, 3.0])
    normalized_scores = numerators / rms
    assert numerators[1] > numerators[0]
    assert normalized_scores[1] < normalized_scores[0]


def test_signed_cosine_rejects_zero_direction():
    try:
        signed_cosine(np.zeros(3), np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("zero direction should be rejected")


def test_discovery_label_map_adapter_satisfies_probe_callable_contract():
    rows = []
    expected = {}
    labels = {}
    for split_index, split in enumerate(
        ("train", "validation", "discovery_test")
    ):
        expected[split] = []
        for label in (0, 1):
            sample_id = f"{split}-{label}"
            expected[split].append(sample_id)
            labels[sample_id] = label
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "target_label": label,
                    "token_selector": "last_prompt",
                    "layer": 17,
                    "site": "resid_post",
                    "feature": float(10 * split_index + label),
                }
            )
    frame = pd.DataFrame(rows)
    label_map = build_discovery_label_map(frame)
    label_of = label_lookup_from_mapping(label_map)

    class CpuLoader:
        def rows_for(self, mask):
            meta = frame[mask].reset_index(drop=True)
            return meta[["feature"]].to_numpy(dtype=np.float64), meta

    mats = design_matrices_for(
        CpuLoader(),
        frame,
        "last_prompt",
        17,
        "resid_post",
        label_of=label_of,
        expected_ids_by_split=expected,
    )
    for split in ("train", "validation", "discovery_test"):
        assert mats[split]["sample_ids"] == expected[split]
        np.testing.assert_array_equal(
            mats[split]["y"], [labels[sid] for sid in expected[split]]
        )

    with pytest.raises(KeyError):
        label_of("unknown-sample")
