import pytest

from representation_reliability.config import (
    CONFIG_ROOT,
    ConfigError,
    apply_override,
    build_merged_doc,
    config_hash,
    deep_merge,
    parse_override,
    resolve_config,
)


def _resolve(overrides=()):
    return resolve_config(
        base_path=CONFIG_ROOT / "base.yaml",
        model_path=CONFIG_ROOT / "models" / "qwen3_0.6b.yaml",
        experiment_path=CONFIG_ROOT / "experiments" / "E00_cartography.yaml",
        overrides=list(overrides),
    )


def test_default_config_resolves():
    cfg, prov = _resolve()
    assert cfg.experiment.id == "E00"
    assert cfg.model.id == "Qwen/Qwen3-0.6B"
    # experiment yaml overrides the model yaml's singular selector list
    assert cfg.representation.token_selectors == ["last_prompt", "target_span_last"]
    assert cfg.representation.sites == ["resid_post"]
    assert cfg.dataset.n_samples == 2000


def test_merge_order_experiment_beats_model_beats_base():
    cfg, _ = _resolve(["runtime.batch_size=2", "dataset.n_samples=64"])
    assert cfg.runtime.batch_size == 2          # base value 8 overridden
    assert cfg.dataset.n_samples == 64          # experiment value overridden


def test_same_resolved_config_same_hash():
    h1 = config_hash(_resolve()[0])
    h2 = config_hash(_resolve()[0])
    assert h1 == h2


def test_changed_scientific_parameter_changes_hash():
    base = config_hash(_resolve()[0])
    changed = config_hash(_resolve(["dataset.n_samples=100"])[0])
    other = config_hash(_resolve(["probe.C_grid=[0.5,1.0]"])[0])
    assert base != changed and base != other


def test_unknown_key_is_rejected_not_silently_dropped(tmp_path):
    import yaml as _yaml
    doc = _yaml.safe_load(
        (CONFIG_ROOT / "experiments" / "E00_cartography.yaml").read_text()
    )
    doc["dataset"]["mystery_scientific_knob"] = 42
    p = tmp_path / "bad_experiment.yaml"
    p.write_text(_yaml.safe_dump(doc))
    with pytest.raises(ConfigError):
        resolve_config(
            base_path=CONFIG_ROOT / "base.yaml",
            model_path=CONFIG_ROOT / "models" / "qwen3_0.6b.yaml",
            experiment_path=p,
        )


@pytest.mark.parametrize("override", [
    "representation.sites=['resid_banana']",
    "representation.layers=[-1]",
    "runtime.dtype=float128",
    "experiment.id=''",
])
def test_invalid_configs_are_rejected(override):
    with pytest.raises(ConfigError):
        _resolve([override])


def test_parse_and_apply_override():
    path, value = parse_override("storage.shard_size=8")
    assert path == "storage.shard_size"
    assert value == 8
    doc = {"a": {"b": {"c": 1}}}
    apply_override(doc, "a.b.c", 7)
    assert doc["a"]["b"]["c"] == 7


def test_deep_merge_scalars_override_lists_replace():
    out = deep_merge({"a": [1, 2], "x": {"y": 1}}, {"a": [3], "x": {"z": 2}})
    assert out["a"] == [3] and out["x"] == {"y": 1, "z": 2}
