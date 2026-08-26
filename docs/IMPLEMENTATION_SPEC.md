# Implementation Specification

## Canonical representation site

A site is:
```text
(model, module_site, layer, token_selector, optional_component)
```

Canonical sites:
- `resid_pre`
- `attn_out`
- `mlp_out`
- `resid_post`

## Sample schema
Required:
```text
sample_id
prompt
target_label
task_name
metadata
```
Optional:
```text
pair_id
counterfactual_id
expected_counterfactual_label
```

## Token selection
Store:
- strategy;
- resolved index;
- token ID;
- decoded token;
- character span when available.

## Activation metadata
```text
sample_id
model_id
model_revision
tokenizer_revision
site
layer
token_index
token_id
token_text
tensor_file
tensor_key
dtype
shape
prompt_hash
```

## Model adapter
Must support:
```python
load()
tokenize()
generate()
forward_logits()
resolve_site()
extract()
intervene()
```

## Instrumentation backends

### HF hooks
Implement basic extraction/simple residual interventions first for debugging and independence.

### NNsight
Primary flexible backend for extraction, editing, gradients, generation-time interventions.

### pyvene
Use for structured counterfactual/path/trainable interventions. Keep behind adapter.

## Probe API
Phase 0:
- logistic regression;
- standardization;
- validation-based regularization;
- random-label control;
- coefficient save;
- untouched confirmation split.

## Causal operators

Replace:
\[
h'_a=h_b
\]

Add:
\[
h'=h+\alpha v
\]

Ablate direction:
\[
h'=h-\mathrm{proj}_v(h)
\]

Every result stores target metric before/after, delta norm, activation norm, and optional output KL.

## Steering constructors
Phase 0:
- diff mean;
- probe normal.

Later:
- pairwise PCA;
- rotations;
- ReFT;
- SAE features.

## Trajectory representation
Per generated step:
```text
step_id
token_start
token_end
pooled_state[layer]
```
Initial geometry:
- position;
- velocity;
- acceleration;
- turn angle;
- curvature;
- displacement;
- probe margin.

## Transform API
Two classes:
- invariant;
- controlled_change.

Every transformed sample stores parent ID and expected label relation.

## Run outputs
```text
runs/<experiment>/<run>/
    config.resolved.yaml
    manifest.json
    status.json
    samples.parquet
    predictions.parquet
    interventions.parquet
    metrics.json
    bootstrap.json
    figures/
    tensors/
    logs/
```

## CLI
```bash
rr validate-config CONFIG
rr extract CONFIG
rr probe CONFIG
rr intervene CONFIG
rr run CONFIG
rr summarize RUN_DIR
```

## Run identity
Hash:
- experiment ID;
- resolved config;
- seed;
- model revision;
- dataset split hash.

Never overwrite completed runs without explicit force.

## Precision
Default:
- bf16 inference where supported;
- fp32 probe fitting;
- store activation dtype explicitly.

## Randomness
Separate recorded seeds for:
- data split;
- generation;
- probe;
- controls;
- bootstrap.
