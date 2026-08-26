# External Module Integration

Clone into `external/repos/`. Do not merge all dependency stacks into one environment.

## NNsight
Repo: `https://github.com/ndif-team/nnsight.git`

Role:
- extraction;
- modification;
- gradients;
- generation-time interventions.

Allowed in main env:
```bash
pip install nnsight
```

Keep trace/save details inside `adapters/nnsight.py`.

## pyvene
Repo: `https://github.com/frankaging/pyvene.git`

Role:
- structured/interchange/path interventions;
- later trainable interventions.

Clone reference repo. Test installation separately before adding to main env.

## AxBench
Repo: `https://github.com/stanfordnlp/axbench.git`

Role:
- detection/steering references;
- concept evaluation.

Use isolated environment; upstream recommends `uv sync`.

## internal_probing
Repo: `https://github.com/zazamrykh/internal_probing.git`

Role:
- linear/PEP correctness-hallucination probes;
- activation enrichment;
- TriviaQA/GSM8K/etc.

Use isolated env:
```bash
pip install -e ".[dev]"
```

Preserve the useful design idea: activation enrichment is cached separately from probe training.

## Reasoning-Flow
Repo: `https://github.com/MasterZhou1/Reasoning-Flow.git`

Role:
- carrier-invariant logic data;
- step hidden states;
- velocity/curvature.

Use included data first. Our extension must add causal tests, not just more geometry plots.

## Multi-component Causal Tracing
Repo: `https://github.com/ZiruiYan/multi-component-causal-tracing.git`

Role:
- PGB-CT;
- sparse joint causal heads/MLPs.

**Strict isolation.**
Its documented environment intentionally uses Transformers 4.51.x because later cache/model internals can differ.

Do not downgrade the main project environment.

Preferred flow:
1. run PGB-CT in its own env;
2. export selected components/masks;
3. ingest those results locally.

## IBM activation-steering
Repo: `https://github.com/IBM/activation-steering.git`

Role:
- steering-vector extraction;
- PCA/diff steering;
- conditional steering reference.

Isolated env first:
```bash
pip install -e .
```

## Optional
- SAELens: `https://github.com/decoderesearch/SAELens.git`
- SAE consistency: `https://github.com/xiangchensong/sae-feature-consistency.git`
- ICR Probe: `https://github.com/XavierZhang2002/ICR_Probe.git`
- Patchscopes/PAIR: `https://github.com/PAIR-code/interpretability.git`
- Honest LLaMA/ITI: `https://github.com/likenneth/honest_llama.git`
- pyReFT: `https://github.com/stanfordnlp/pyreft.git`
- StateBridge: `https://github.com/YanwenPneg/StateBridge.git`
- Selective steering: `https://github.com/knoveleng/steering.git`

Do not train large SAEs for preliminary work; use pretrained SAEs later.

## SHA logging
At each run:
```bash
git -C external/repos/<repo> rev-parse HEAD
```
Store the exact SHA in run manifest.
