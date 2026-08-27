import pandas as pd

from representation_reliability.runners.e01a import (
    alpha_values,
    parse_trace_layers,
    select_same_label_sources,
    select_shuffled_opposite_sources,
)


def _rows():
    data = []
    for i, relation in enumerate(("north_south", "north_south", "east_west")):
        pair = f"p{i}"
        for label in (1, 0):
            data.append(
                {
                    "sample_id": f"{pair}-{label}",
                    "pair_id": pair,
                    "target_label": label,
                    "relation": relation,
                    "queried_word": "north" if relation == "north_south" else "east",
                    "template_id": f"{relation}.p0",
                }
            )
    return pd.DataFrame(data)


def test_same_label_sources_never_use_same_pair_and_preserve_label():
    df = _rows()
    mapping = select_same_label_sources(df)
    by_id = df.set_index("sample_id")
    for base, source in mapping.items():
        assert by_id.loc[base, "pair_id"] != by_id.loc[source, "pair_id"]
        assert int(by_id.loc[base, "target_label"]) == int(
            by_id.loc[source, "target_label"]
        )


def test_shuffled_sources_flip_label_and_never_use_matched_pair():
    df = _rows()
    mapping = select_shuffled_opposite_sources(df, seed=11)
    by_id = df.set_index("sample_id")
    for base, source in mapping.items():
        assert by_id.loc[base, "pair_id"] != by_id.loc[source, "pair_id"]
        assert int(by_id.loc[base, "target_label"]) != int(
            by_id.loc[source, "target_label"]
        )


def test_alpha_profiles_and_trace_layer_validation():
    assert alpha_values("smoke") == (0.0, 1.0)
    assert 1.0 in alpha_values("full")
    assert parse_trace_layers(
        "20,23,27", intervention_layer=17, num_layers=28
    ) == [17, 20, 23, 27]
