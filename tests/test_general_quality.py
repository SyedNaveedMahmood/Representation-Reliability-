from representation_reliability.metrics.general_quality import (
    block_scored_token_count,
    deterministic_hellaswag_indices,
)


def test_hellaswag_subset_is_deterministic_and_identity_based():
    rows = [{"ind": index} for index in range(20)]
    first = deterministic_hellaswag_indices(rows, 5)
    second = deterministic_hellaswag_indices(rows, 5)
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert first != list(range(5))


def test_wikitext_block_scored_token_accounting_meets_e13_floor():
    assert block_scored_token_count(10_000, 512) == 9_980
    assert block_scored_token_count(10_021, 512) == 10_001
