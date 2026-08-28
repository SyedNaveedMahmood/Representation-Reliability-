from dataclasses import replace

import pandas as pd
import pytest

from representation_reliability.data.base import samples_to_dataframe
from representation_reliability.data.synthetic import generate_synthetic_relations
from representation_reliability.runners.e01a import (
    _allocate_or_resume_run_dir,
    alpha_values,
    parse_trace_layers,
    select_same_label_sources,
    select_shuffled_opposite_sources,
)
from representation_reliability.runners.e01a_support import (
    build_source_plans,
    intervention_base_activations,
    load_activation_snapshot,
    save_activation_snapshot,
)
from representation_reliability.runtime.status import StatusFile


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
        assert int(by_id.loc[base, "target_label"]) == int(by_id.loc[source, "target_label"])


def test_shuffled_sources_flip_label_and_never_use_matched_pair():
    df = _rows()
    mapping = select_shuffled_opposite_sources(df, seed=11)
    by_id = df.set_index("sample_id")
    for base, source in mapping.items():
        assert by_id.loc[base, "pair_id"] != by_id.loc[source, "pair_id"]
        assert int(by_id.loc[base, "target_label"]) != int(by_id.loc[source, "target_label"])


def test_alpha_profiles_and_trace_layer_validation():
    assert alpha_values("smoke") == (0.0, 1.0)
    assert 1.0 in alpha_values("full")
    assert parse_trace_layers("20,23,27", intervention_layer=17, num_layers=28) == [17, 20, 23, 27]
    with pytest.raises(ValueError, match="within the model"):
        parse_trace_layers("20,28", intervention_layer=17, num_layers=28)


def test_source_plan_rejects_a_counterfactual_nuisance_mismatch():
    samples = generate_synthetic_relations(n_samples=12, seed=9)
    frame = samples_to_dataframe(samples)
    frame["split"] = "discovery_test"
    by_id = {sample.sample_id: sample for sample in samples}
    base = samples[0]
    matched = by_id[str(base.counterfactual_id)]
    corrupted = replace(
        matched,
        metadata={**matched.metadata, "queried_word": "corrupted"},
    )
    by_id[matched.sample_id] = corrupted

    with pytest.raises(RuntimeError, match="counterfactual nuisance mismatch"):
        build_source_plans(
            frame,
            by_id,
            base_sample_ids=[base.sample_id],
            seed=4,
        )


def test_activation_snapshot_roundtrip_and_identity_guard(tmp_path):
    sample_ids = ["a", "b"]
    activations = {
        17: {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        },
        20: {
            "a": [5.0, 6.0],
            "b": [7.0, 8.0],
        },
    }
    token_indices = {"a": 3, "b": 4}
    token_sites = {
        "a": {"token_index": 3, "token_id": 8},
        "b": {"token_index": 4, "token_id": 9},
    }
    save_activation_snapshot(
        tmp_path,
        activations,
        sample_ids=sample_ids,
        token_indices=token_indices,
        token_sites=token_sites,
    )
    loaded = load_activation_snapshot(
        tmp_path,
        expected_sample_ids=sample_ids,
        expected_layers=[17, 20],
    )
    assert loaded is not None
    loaded_activations, loaded_indices, loaded_sites = loaded
    assert loaded_indices == token_indices
    assert loaded_sites == token_sites
    assert loaded_activations[20]["b"].tolist() == [7.0, 8.0]

    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_activation_snapshot(
            tmp_path,
            expected_sample_ids=list(reversed(sample_ids)),
            expected_layers=[17, 20],
        )


def test_e01a_run_directory_resumes_only_incomplete_identical_run(tmp_path):
    canonical = tmp_path / "E01A" / "E01A_abc"
    first = StatusFile.create(canonical, "E01A_abc", "E01A")
    first.complete()
    rerun = tmp_path / "E01A" / "E01A_abc-r2"
    second = StatusFile.create(rerun, "E01A_abc", "E01A")
    second.fail("simulated interruption")

    selected, resumed = _allocate_or_resume_run_dir(
        tmp_path, "E01A_abc", resume=True
    )
    assert selected == rerun
    assert resumed is True

    fresh, resumed = _allocate_or_resume_run_dir(
        tmp_path, "E01A_abc", resume=False
    )
    assert fresh == tmp_path / "E01A" / "E01A_abc-r3"
    assert resumed is False


def test_intervention_uses_base_captured_with_exact_forward_batch():
    clean = {
        "a": {"captured": {17: [1.0, 2.0]}},
        "b": {"captured": {17: [3.0, 4.0]}},
    }
    stale_extraction = {"a": [10.0, 20.0], "b": [30.0, 40.0]}
    bases = intervention_base_activations(clean, ["a", "b"], layer=17)
    assert bases["a"].tolist() == [1.0, 2.0]
    assert bases["b"].tolist() == [3.0, 4.0]
    assert bases["a"].tolist() != stale_extraction["a"]
