"""GPU/model contract tests (skipped when the model is not available locally).

Covers:
  - site resolution maps to real Qwen3 modules;
  - target_span_last / last_prompt land on expected tokens;
  - Test C: hook-path and output_hidden_states extraction agree;
  - cache round-trip through the real adapter;
  - Test D (optional): agreement with an independent NNsight extraction.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

MODEL_ID = "Qwen/Qwen3-0.6B"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture(scope="module")
def adapter():
    from representation_reliability.config.loader import CONFIG_ROOT
    from representation_reliability.config.schema import (
        ModelConfig,
        RuntimeConfig,
    )
    from representation_reliability.adapters.hf import HFAdapter

    mcfg = ModelConfig(id=MODEL_ID, family="qwen3", dtype="float32",
                       device_map="cuda")
    rcfg = RuntimeConfig(dtype="float32", batch_size=4)
    return HFAdapter(mcfg, rcfg).load()


PROMPTS = [
    ("Premise: Luma is north of Reko.\nQuestion: Is Reko south of Luma?\nAnswer:",
     "Is Reko south of Luma?", True),
    ("Premise: Vanta happens before event Sidra.\nQuestion: Does event Sidra "
     "happen after event Vanta?\nAnswer:", "Does event Sidra happen after event Vanta?",
     True),
]


def _resolve_token(adapter, strategy, prompt, target):
    from representation_reliability.extraction.token_selection import (
        resolve_token_selection,
    )
    return resolve_token_selection(
        strategy=strategy, tokenizer=adapter.tokenizer, prompt_text=prompt,
        chat_template_used=False, target_text=target,
    )


def test_resolve_sites_map_to_native_modules(adapter):
    r1 = adapter.resolve_site("resid_post", 5)
    assert r1.native_module_name == f"{adapter._prefix}layers[5]"
    r2 = adapter.resolve_site("attn_out", 0)
    assert ".self_attn" in r2.native_module_name
    with pytest.raises(ValueError):
        adapter.resolve_site("resid_post", adapter.num_layers)


def test_token_selection_lands_on_expected_text(adapter):
    prompt, target, _ = PROMPTS[0]
    enc = adapter.encode_with_offsets(prompt)
    # last_prompt must be the final token
    tok = adapter.tokenize([prompt])
    seq_len = tok["input_ids"].shape[1]
    sel = _resolve_token(adapter, "last_prompt", prompt, None)
    assert sel.index == seq_len - 1
    # target_span_last must START inside the question sentence (its last
    # token may merge into trailing whitespace, e.g. '?\n')
    span_start = prompt.index(target)
    tsel = _resolve_token(adapter, "target_span_last", prompt, target)
    s, e = enc["offset_mapping"][tsel.index]
    assert s >= span_start and s < span_start + len(target)
    # decode the selected token neighborhood -> must contain the '?' glyph
    q_tail = adapter.decode(enc["input_ids"][max(0, tsel.index - 1): tsel.index + 1])
    assert "?" in q_tail


def test_hook_vs_hidden_states_agree(adapter):
    """Test C: two extraction mechanisms agree on INTERIOR resid_post layers.

    Under transformers >=4.5x the final ``hidden_states`` element is post-
    final-norm, so the comparison deliberately excludes the last layer; the
    hook path is authoritative and serves every layer (see load().calibrate()).
    """
    prompts = [p for p, _, _ in PROMPTS]
    idxs = [_resolve_token(adapter, "last_prompt", p, None).index for p in prompts]
    interior = [0, adapter.num_layers // 2]     # exclude final layer
    requests = [("resid_post", l) for l in interior]
    hooks_out = adapter.extract(prompts, requests, idxs, use_hooks_for_resid=True)
    hs_out = adapter.extract(prompts, requests, idxs, use_hooks_for_resid=False)
    for key in requests:
        dev = float(np.max(np.abs(hooks_out[key] - hs_out[key])))
        scale = max(float(np.max(np.abs(hooks_out[key]))), 1e-6)
        assert dev / scale < 2e-2, (key, dev / scale)
    # and the hook path handles the FINAL layer where hidden_states cannot:
    final_key = [("resid_post", adapter.num_layers - 1)]
    fh = adapter.extract(prompts, final_key, idxs, use_hooks_for_resid=True)
    assert np.isfinite(fh[final_key[0]]).all()


def test_nnsight_crosscheck(adapter):
    """Optional Test D: NNsight's extraction of resid_post agrees."""
    import torch as _torch

    nnsight = pytest.importorskip("nnsight")
    prompt, target, _ = PROMPTS[0]
    sel = _resolve_token(adapter, "target_span_last", prompt, target)
    layer = adapter.num_layers // 2

    ours = adapter.extract([prompt], [("resid_post", layer)], [sel.index])[
        ("resid_post", layer)
    ][0]

    lm = nnsight.LanguageModel(
        MODEL_ID, device_map="cuda", dispatch=True, dtype=_torch.float32,
    )
    assert str(lm.dtype).endswith("float32"), lm.dtype
    with lm.trace(prompt):
        out = lm.model.layers[layer].output.save()
    tensor = out[0] if isinstance(out, tuple) else out
    print("\nnnsight dtype:", tensor.dtype, "shape:", tuple(tensor.shape))
    theirs = tensor[0][sel.index].float().cpu().detach().numpy()

    denom = max(float(np.max(np.abs(theirs))), 1e-6)
    rel = float(np.max(np.abs(ours - theirs))) / denom
    print("ours|max", float(np.max(np.abs(ours))), "theirs|max", denom)
    assert rel < 5e-2, f"HF-hook vs NNsight relative deviation {rel:.4f}"


def test_cache_roundtrip_with_real_adapter(adapter, tmp_path):
    from representation_reliability.extraction.activations import (
        build_token_selections,
    )
    from representation_reliability.extraction.cache import (
        ActivationCacheReader,
        ActivationShardWriter,
    )

    prompt, target, _ = PROMPTS[0]
    sel = _resolve_token(adapter, "target_span_last", prompt, target)
    layer = adapter.num_layers // 2
    got = adapter.extract([prompt], [("resid_post", layer)], [sel.index])
    vec = got[("resid_post", layer)][0]

    w = ActivationShardWriter(tmp_path, shard_size=1)
    w.add(vec, {"sample_id": "x", "site": "resid_post", "layer": layer})
    w.flush()
    reader = ActivationCacheReader(tmp_path)
    df = reader.index()
    loaded = reader.load_rows(df)[0]
    scale = max(float(np.max(np.abs(vec))), 1e-9)
    assert float(np.max(np.abs(loaded - vec))) / scale < 1e-5
    diag = reader.verify_alignment()
    assert all(d["aligned"] for d in diag.values())


