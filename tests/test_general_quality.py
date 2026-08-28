from representation_reliability.metrics.general_quality import (
    deterministic_hellaswag_indices,
)


def test_hellaswag_subset_is_deterministic_and_identity_based():
    rows = [{"ind": index} for index in range(20)]
    first = deterministic_hellaswag_indices(rows, 5)
    second = deterministic_hellaswag_indices(rows, 5)
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert first != list(range(5))
