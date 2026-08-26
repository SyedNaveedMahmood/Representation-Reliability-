import pytest

from representation_reliability.data.synthetic import generate_synthetic_relations
from representation_reliability.data.transforms import (
    TRANSFORM_REGISTRY,
    transform_paraphrase,
    transform_query_entity_swap,
)


@pytest.fixture(scope="module")
def ds():
    return generate_synthetic_relations(40, seed=42)


def test_transform_registry_classes():
    assert TRANSFORM_REGISTRY["paraphrase"][1] == "invariant"
    assert TRANSFORM_REGISTRY["query_entity_swap"][1] == "controlled_change"


def test_paraphrase_is_invariant(ds):
    s = ds[0]
    t = transform_paraphrase(s, seed=1)
    assert int(t.target_label) == int(s.target_label)
    assert t.metadata["transform_class"] == "invariant"
    assert t.metadata["expected_label_relation"] == "same"
    assert t.metadata["parent_sample_id"] == s.sample_id
    assert t.prompt != s.prompt          # wording actually changed


def test_query_entity_swap_is_controlled_change(ds):
    for s in ds:
        if s.target_label == 1:
            break
    t = transform_query_entity_swap(s)
    assert int(t.target_label) != int(s.target_label)
    assert t.metadata["transform_class"] == "controlled_change"
    assert t.metadata["parent_sample_id"] == s.sample_id
    assert t.metadata["queried_word"] == s.metadata["queried_word"]
    assert t.metadata["queried_side"] != s.metadata["queried_side"]
