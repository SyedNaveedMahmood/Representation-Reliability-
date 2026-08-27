from representation_reliability.adapters.hf import HFAdapter
from representation_reliability.config.schema import (
    GenerationConfig,
    ModelConfig,
    RuntimeConfig,
)


def test_generation_kwargs_come_from_model_config_not_runtime():
    mcfg = ModelConfig(
        id="Qwen/Qwen3-0.6B", family="qwen3",
        generation=GenerationConfig(max_new_tokens=17, do_sample=True),
    )
    rcfg = RuntimeConfig(dtype="float32", batch_size=4)
    adapter = HFAdapter(mcfg, rcfg)          # no load() needed for kwargs
    kwargs = adapter.default_generation_kwargs()
    assert kwargs == {"max_new_tokens": 17, "do_sample": True}


def test_generation_kwargs_change_with_config_values():
    base = dict(id="Qwen/Qwen3-0.6B", family="qwen3")
    lo = HFAdapter(ModelConfig(**base,
                               generation=GenerationConfig(max_new_tokens=4)),
                   None).default_generation_kwargs()
    hi = HFAdapter(ModelConfig(**base,
                               generation=GenerationConfig(max_new_tokens=400)),
                   None).default_generation_kwargs()
    assert lo["max_new_tokens"] == 4 and hi["max_new_tokens"] == 400

    deterministic = HFAdapter(
        ModelConfig(**base, generation=GenerationConfig(do_sample=False)), None
    ).default_generation_kwargs()
    sampling = HFAdapter(
        ModelConfig(**base, generation=GenerationConfig(do_sample=True)), None
    ).default_generation_kwargs()
    assert deterministic["do_sample"] is False and sampling["do_sample"] is True


def test_padding_side_policy_is_explicit_right():
    mcfg = ModelConfig(id="Qwen/Qwen3-0.6B", family="qwen3")
    rcfg = RuntimeConfig(dtype="float32", batch_size=2)
    adapter = HFAdapter(mcfg, rcfg).load()
    assert adapter.tokenizer.padding_side == "right"
