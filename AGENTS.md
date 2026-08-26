# Coding Agent Contract

This file is authoritative.

## Mission

Build a **modular causal-representation experimentation harness** to discover open phenomena in representation reliability.

Do not optimize for reproducing one paper or leaderboard performance. Do not change the scientific question to fit whichever external library is easiest.

## Non-negotiable engineering rules

### Stable local APIs
External repos must sit behind local adapters. Project runners must not depend directly on AxBench internals, NNsight trace syntax, pyvene config objects, or model-specific module names.

### Token sites must be explicit
Every activation record stores:
- raw text;
- token IDs and decoded strings;
- selector strategy;
- resolved token index;
- character span when available;
- chat-template use.

`-1` is not a scientific site description.

### Layer/site convention
Canonical sites:
- `resid_pre`
- `attn_out`
- `mlp_out`
- `resid_post`

Layers are 0-indexed. Store the model-native module name alongside the canonical site.

### Raw evidence
Every run writes per-example outputs before aggregates. Never save only plots/means.

### Intervention controls
Every causal/steering study includes:
- no intervention;
- magnitude-matched random direction;
- shuffled source/counterfactual where applicable;
- multiple magnitudes;
- `||delta h||`, `||h||`, and their ratio.

Where useful also include adjacent-layer and irrelevant-concept controls.

### No causal language from probes
Probe-only results establish decodability/correlation, not model use.

### Full run manifests
Record:
- project git SHA;
- external repo SHAs;
- Python, torch, transformers, NNsight/pyvene versions;
- CUDA/GPU;
- model/tokenizer ID and revision;
- prompt hash;
- dataset split hash;
- seeds;
- config/overrides;
- wall time and peak VRAM.

### Cache expensive forwards
Use portable artifacts:
- metadata: parquet/jsonl;
- tensors: safetensors or tensor shards.
Avoid arbitrary-object pickle as primary evidence.

### Safe resume
Long jobs must resume by shard. Partial output must never be marked complete.

### Tests before sweeps
Before multi-hour GPU work:
- unit tests;
- tiny-model/CPU contract where feasible;
- <=50-example GPU smoke;
- expected files validated.

## Probe rules

Always include:
- majority baseline;
- text/surface baseline when meaningful;
- random-label control;
- train/val/test separation;
- validation-only hyperparameter selection;
- AUROC + AUPRC;
- calibration when probe is used as a monitor.

## Causal patching rules

Source/base pairs should match nuisance attributes. Save:
- base/source/patched outputs;
- expected counterfactual;
- probability/logit deltas;
- intervention norm.

If there is no well-defined counterfactual target, call the outcome `behavioral_effect`, not `recovery`.

## Steering rules

Never evaluate one alpha. Default normalized-direction sweep:

`[-4,-2,-1,-0.5,0,0.5,1,2,4]`

Include norm-matched random directions.

## Robustness rules

Separate:
- **invariant transformations**: target stays same;
- **controlled-change transformations**: target changes predictably.

Negation is not semantics-preserving if the target changes.

## Temporal reasoning rule

Do not assume visible CoT is faithful internal thought. Claims concern model states conditioned on generated tokens unless faithfulness is separately established.

## Multiple comparisons

Layer/head/token scans are exploratory. Select peaks on discovery data and confirm on untouched data.

## Architecture to implement

```text
src/representation_reliability/
    adapters/
        hf.py
        nnsight.py
        pyvene.py
    data/
        base.py
        synthetic.py
        transforms.py
    extraction/
        activations.py
        cache.py
    probes/
        linear.py
    interventions/
        patch.py
        add.py
        ablate.py
        rotate.py          # later
        kv.py              # later
    trajectories/
        geometry.py
        changepoints.py
    metrics/
        decoding.py
        causal.py
        steering.py
        monitoring.py
        robustness.py
        safety.py
    runners/
        extract.py
        probe.py
        intervene.py
        sweep.py
    reporting/
        tables.py
        plots.py
        discovery_report.py
```

Do not create empty modules just to match the tree. Implement in phase order.

## Phase order

### Phase 0A
Config loader -> manifest -> HF adapter -> extraction/cache -> linear probe -> metrics -> CLI.

### Phase 0B
NNsight adapter -> replacement/addition -> counterfactual patching -> alpha sweeps -> random controls.

### Phase 0C
Transforms -> robustness -> trajectory metrics -> bootstrap reporting.

### Phase 1
E00–E07.

### Phase 2
Only after discovery signals:
SAEs, PGB-CT integration, KV interventions, ReFT/low-rank, latent communication.

## Forbidden shortcuts

Do not:
- cherry-pick layers after test inspection;
- discard failed seeds;
- compare unmatched intervention norms;
- silently alter prompt semantics across models;
- claim "feature" when only a probe direction exists;
- silently truncate contexts differently;
- use sampling in one condition and greedy decoding in another;
- use an API LLM judge as the sole preliminary outcome.

## What counts as interesting

Examples:
- high D but negligible C;
- C succeeds while safe additive S fails;
- monitor detects corruption but final error remains unpredictable;
- steering direction reverses under an innocuous transformation;
- concept migrates across token/layer sites;
- SAE activation is stable but causal meaning is not;
- a correction removes visible evidence while KV state still drives the obsolete answer.

When an unexpected result appears, add a new hypothesis to `research/EXPERIMENT_REGISTRY.yaml` **before scaling it**.
