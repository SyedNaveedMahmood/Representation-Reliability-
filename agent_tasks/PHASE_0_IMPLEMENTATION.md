# Phase 0 Implementation Task List

Execute in order.

## 0.1 Config + manifests
Implement:
- YAML load/merge;
- Pydantic validation;
- resolved config save;
- config hash;
- run status;
- environment/GPU/git metadata.

Tests:
- invalid site fails;
- same resolved config -> same hash;
- unknown scientific keys do not silently disappear.

## 0.2 HF adapter
Support first:
- Qwen3-0.6B;
- Qwen3-1.7B.

Implement:
- tokenizer;
- optional chat template;
- generation;
- forward logits;
- canonical site resolution.

## 0.3 Activation extraction/cache
Implement:
- selected layers/sites/tokens;
- safetensor/tensor shards;
- metadata parquet;
- resume by shard;
- activation dtype logging.

Validate reload == written tensor within dtype tolerance.

## 0.4 Synthetic relational generator
Requirements:
- deterministic;
- matched counterfactual pairs;
- exact labels;
- invariant and controlled-change transforms;
- split isolation.

## 0.5 Linear probe
Implement:
- standardization;
- logistic regression;
- validation C search;
- class weighting;
- random-label control;
- coefficient save;
- AUROC/AUPRC/balanced accuracy.

Never use confirmation split for layer selection.

## 0.6 E00
Run:
1. <=50 example smoke Qwen3-0.6B;
2. 500 example pilot;
3. Qwen3-1.7B discovery.

Required artifact:
D-vs-layer with random-label and text baselines.

## 0.7 NNsight adapter
Install NNsight in main env.

Implement:
- read activation;
- replace activation;
- add delta;
- save output logits.

Contract:
alpha=0 == baseline.

## 0.8 E01 counterfactual patching
Implement:
- matched base/source;
- replacement;
- shuffled source;
- norm-matched random;
- normalized recovery.

Scan every 4th layer first.

## 0.9 E02 steering
Implement:
- diff mean;
- probe normal;
- direction normalization;
- alpha sweep;
- random directions.

Save raw output at every alpha.

## 0.10 Reporting
Produce:
```text
site
D
C
S
K
G_DC
G_CS
CI
control separation
```

## Stop condition
Do not implement SAEs, KV cache, PGB-CT, PEP, or ReFT until E00-E03 work end-to-end and resumably.
