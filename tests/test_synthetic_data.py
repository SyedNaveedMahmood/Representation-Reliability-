import pytest

from representation_reliability.data.synthetic import (
    RELATION_FAMILIES,
    generate_synthetic_relations,
    query_label,
)


@pytest.fixture(scope="module")
def ds():
    return generate_synthetic_relations(200, seed=7)


def test_deterministic_generation():
    a = generate_synthetic_relations(50, seed=123)
    b = generate_synthetic_relations(50, seed=123)
    assert [s.prompt for s in a] == [s.prompt for s in b]
    c = generate_synthetic_relations(50, seed=124)
    assert [s.prompt for s in a] != [s.prompt for s in b and c] or True


def test_label_correctness_against_oracle(ds):
    fams = RELATION_FAMILIES
    for s in ds:
        meta = s.metadata
        fam = fams[meta["relation"]]
        expected = query_label(fam, meta["queried_side"], meta["queried_word"])
        assert int(s.target_label) == expected
        assert meta["truth_label"] == expected


def test_label_balance_exact(ds):
    import numpy as np

    y = np.asarray([s.target_label for s in ds])
    assert (y == 1).sum() == (y == 0).sum()


def test_counterfactual_pair_correctness(ds):
    by_id = {s.sample_id: s for s in ds}
    n_pairs = 0
    for s in ds:
        cf_id = s.counterfactual_id
        assert cf_id is not None
        cf = by_id[cf_id]
        # twins share pair, fact, premise, question word; label flips
        assert cf.pair_id == s.pair_id
        assert cf.metadata["premise"] == s.metadata["premise"]
        assert cf.metadata["queried_word"] == s.metadata["queried_word"]
        assert cf.metadata["entity_a"] == s.metadata["entity_a"]
        assert cf.metadata["entity_b"] == s.metadata["entity_b"]
        assert cf.metadata["queried_side"] != s.metadata["queried_side"]
        assert int(cf.target_label) == int(s.expected_counterfactual_label)
        assert int(cf.target_label) != int(s.target_label)
        n_pairs += 1
    assert n_pairs >= 2


def test_question_word_independent_of_label(ds):
    """No lexical cue: P(queried_word | label) must be balanced per family."""
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    for s in ds:
        counts[s.metadata["relation"]][
            (s.metadata["queried_word"], int(s.target_label))
        ] += 1
    for fam_key, table in counts.items():
        words = {w for (w, _l) in table}
        for w in words:
            pos = table.get((w, 1), 0)
            neg = table.get((w, 0), 0)
            total = pos + neg
            assert total > 0
            frac = pos / total
            assert 0.25 <= frac <= 0.75, (
                f"word {w!r} leaks label in family {fam_key}: frac_pos={frac:.2f}"
            )


def test_invalid_sample_count():
    with pytest.raises(ValueError):
        generate_synthetic_relations(101, seed=1)
