# E01 Actionability Confirmation Protocol

Status: `confirmation_preregistered_authorized_once`

This document prospectively freezes the single confirmation analysis for E01A/E01B. It was written from discovery evidence only. No confirmation examples, labels, activations, metrics, caches, or split counts were accessed while preparing it.

## Scope and claim boundary

The confirmation concerns only the synthetic relation task, the two named Qwen3 checkpoints, and the frozen layer-17 `resid_post` / `last_prompt` site. It does not establish a universal mechanism across tasks, sites, architectures, or model families.

If all four primary hypotheses pass, the frozen claim template is:

> Under the tested relation task, site, and checkpoints, semantic actionability is distributed rather than reducible to a single linearly decoded feature. A probe-defined scalar semantic coordinate is causally effective, while structured orthogonal state carries additional causal information. The larger checkpoint additionally exhibits a structured interaction in which orthogonal state changes the efficacy of the same scalar semantic intervention.

For Qwen3-0.6B, absence language is restricted to “no structured q-by-context interaction was detected” unless a future, separately preregistered equivalence study supplies an independently justified margin. This confirmation does not test equivalence.

## Locked scientific identity

### Models and tokenizer

| Model | Config | Frozen resolved model/tokenizer revision | Hidden size |
|---|---|---|---:|
| Qwen3-0.6B | `configs/models/qwen3_0.6b.yaml` | `c1899de289a04d12100db370d81485cdf75e47ca` | 1024 |
| Qwen3-1.7B | `configs/models/qwen3_1.7b.yaml` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 2048 |

Both use BF16. The candidates are exactly `" Yes"` and `" No"`, each one token under the frozen tokenizer, with IDs `[7414, 2308]` in Yes/No order. A mismatch is a hard failure.

### Site and probe

- canonical site: `resid_post`
- zero-indexed layer: `17`
- token selector: `last_prompt`
- native module: `model.layers[17]`
- trace layers: `[17, 20, 23, 27]`
- direction: the unit coefficient direction of the frozen train/validation linear truth probe
- scaler fitting: train only
- regularization selection: validation only
- final probe fitting: train only after validation-only C selection, exactly matching the established discovery probe contract
- confirmation labels: never used to fit the scaler, choose C, fit the probe, choose the direction, or construct targets

Frozen probe/scaler digests:

| Model | Probe/scaler digest |
|---|---|
| Qwen3-0.6B | `6177a52089623422091c3f725aaffb18db584063485c91a9c16a7492694d5a2e` |
| Qwen3-1.7B | `f368240514b0ae5fc9fabd14401656456b16b3a0c9bbc417f8a2dd4982b606b7` |

The runner must reconstruct the probe only from the frozen train/validation data and reject any digest mismatch. It may not select or modify the probe using confirmation.

### Validation-derived scalar targets and scales

These values are immutable outputs of the completed E01B-1 discovery runs.

| Quantity | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| `q0_star` | -3.1458192552878472 | -3.1950168980032565 |
| `q1_star` | -1.5404356794123624 | 14.815493967122066 |
| Q05 | -3.5167752670741588 | -8.341544216220596 |
| Q25 | -3.145577740246652 | -3.1946762711727907 |
| Q50 | -2.213101394332157 | 5.353254935615931 |
| Q75 | -1.5407368278597313 | 14.813350379272423 |
| Q95 | -0.983605862111277 | 18.201303712784043 |
| `sigma_q_validation` | 0.8701864873690688 | 9.709164114642144 |
| `sigma_margin_validation` | 1.38642706615052 | 3.11930693101263 |
| validation matched-context fallback norm | 9.397263646574874 | 55.86045287059804 |

Frozen target artifact SHA-256 digests:

| Model | Artifact | SHA-256 |
|---|---|---|
| Qwen3-0.6B | `runs/E01B2/E01B2_f2d75dab1eba/setpoint_targets.json` | `709da161845d33e11275b7e66ca0686e652d45cbb86468f101ba7220eae7c7bb` |
| Qwen3-1.7B | `runs/E01B2/E01B2_e2b4b02cb3a4/setpoint_targets.json` | `2c17d27e50869280f64b73624deb31c14302ef729edf263bdd4d0a524a72ccdc` |

Layer-specific validation scales in those frozen artifacts are used for secondary trace standardization. Confirmation must not re-estimate any target or standardization scale.

## Frozen interventions

Let `u` be the frozen unit truth-probe direction, `q_b = u^T h_b`, and `q_target` the validation-only median for the base example's opposite label.

- `Y00`: clean state, `h_b`
- `Y10`: scalar setpoint only, `h_b + (q_target - q_b)u`
- `Y01,c`: orthogonal context only, `h_b + lambda v_perp,c`
- `Y11,c`: scalar setpoint plus context, `h_b + (q_target - q_b)u + lambda v_perp,c`

The primary lambda is `1.0`; `0.5` is secondary. Context families are matched, same-family shuffled, different-family shuffled, same-label, and random orthogonal. Context vectors are explicitly orthogonalized against `u` and per-example norm-matched to the matched-twin orthogonal reference norm. If that norm is below `1e-12`, use the corresponding frozen validation fallback norm above and record the fallback.

The semantic target for every oriented outcome is the base example's frozen opposite class. It is never derived from a context source.

### Source construction and randomness

- matched context: the authoritative `counterfactual_id`
- same-family shuffled: opposite-label sample from a different pair and the same relation family
- different-family shuffled: opposite-label sample from a different pair and a different relation family
- same-label: same-label sample from a different pair
- selections: deterministic stable-hash choice from the eligible confirmation pool using base seed `20260830`
- scalar random direction seeds: `20270830` through `20270839`
- scalar orthogonal-random seeds: `20280830` through `20280839`
- random orthogonal context seeds: `20300830` through `20300839`

The source plan is created once after authorized confirmation access, persisted before model interventions, shared by semantic sample identity across both checkpoints, and never regenerated or selected by outcome. The runner rejects self-source, base-pair leakage, label/family violations, model disagreement in source identity, or a changed source-plan digest.

## Outcomes and factorial estimands

The raw native quantity is `logit(" Yes") - logit(" No")`, oriented toward the frozen opposite-class target. Raw evidence is written before aggregates.

- scalar causal actionability: `Q0 = Y10 - Y00`
- additive structured signal: `A_c = Y01,c - Y00`
- q effect in context: `Q_c = Y11,c - Y01,c`
- interaction: `G_c = (Y11,c - Y10) - (Y01,c - Y00) = Q_c - Q0`

Scalar controls are magnitude-matched random and orthogonal-random directions. Random-context rows are averaged over the ten frozen seeds within base example before primary matched contrasts. The cluster is always `pair_id`; directed twins are never resampled independently.

## Primary family: exactly four hypotheses

All tests are directional and use lambda `1.0`.

### H1 — scalar causal actionability in both checkpoints

For each checkpoint all three requirements must hold: `Q0 > 0`, `Q0 - Q_random > 0`, and `Q0 - Q_orthogonal_random > 0`. H1 is an intersection-union conjunction across all six checkpoint-specific component tests. Its raw p-value is the maximum component p-value; it passes only when the Holm-adjusted H1 p-value is below 0.05 and every component estimate is positive.

### H2 — causal information outside scalar q in both checkpoints

For each checkpoint, `A_matched - A_random > 0`. H2 is a conjunction across the two checkpoint-specific component tests. Its raw p-value is their maximum; it passes only when the Holm-adjusted H2 p-value is below 0.05 and both component estimates are positive.

### H3 — structured interaction in Qwen3-1.7B

`G_matched - G_random > 0` in Qwen3-1.7B.

### H4 — checkpoint difference in structured interaction

`(G_matched - G_random)_1.7B - (G_matched - G_random)_0.6B > 0`.

The runner must prove the two model evaluations use identical semantic sample and pair identities before using a paired cross-checkpoint contrast. Any mismatch is a hard failure, not permission to change the test.

## Locked inference

- family-wise alpha: `0.05`
- multiple comparisons: Holm step-down correction across exactly H1–H4
- effect confidence intervals: deterministic 95% percentile pair-cluster bootstrap, 10,000 draws, seed `20260831`
- directional p-values: pair-cluster sign-flip randomization, 100,000 Monte Carlo draws with deterministic hypothesis-specific seed offsets and the plus-one correction
- H1/H2 conjunction p-values: intersection-union maximum of component one-sided p-values
- H1/H2 display estimate: minimum component estimate, accompanied by all component estimates; its interval is the bootstrap distribution of the same minimum statistic
- H4: paired semantic-pair cluster resampling/sign flipping after exact cross-model identity validation
- confidence intervals are descriptive marginal 95% intervals; multiplicity control is applied to the four primary p-values

Random seeds are averaged within base before pair-level inference. All directed rows from a sampled pair travel together. Missing or non-finite primary rows are an integrity failure rather than silently dropped observations.

## Decision rule

- **Strong confirmation:** H1, H2, H3, and H4 pass.
- **Partial confirmation:** the distributed-actionability core is replicated (H1, H2, H3 pass without a material sign reversal) but H4 is inconclusive, or another scale-specific extension is inconclusive without contradicting the core.
- **Failed confirmation:** H1 or H2 materially fails/reverses, the Qwen3-1.7B structured interaction materially reverses, or a major core-mechanism contradiction appears.

E14 may proceed only if H1, H2, and H3 pass and there is no material sign reversal. H4 may be inconclusive only if its point direction remains consistent.

## Secondary confirmation only

The following are frozen secondary outcomes and are not promoted into H1–H4: same-family, different-family, and same-label contexts; lambda `0.5`; the validation-derived continuous q grid; relation-family heterogeneity; traces at L17/L20/L23/L27; layerwise A/G decomposition; behavioral flips; and raw plus validation-standardized effect sizes.

No family is selected or dropped. No new metric, threshold, seed, context, lambda, layer, or favorable subgroup may be added after confirmation access.

## Integrity and single-use execution

Confirmation is one joint campaign containing both models with both configurations frozen before the first split load. The runner creates an append-only access record immediately before the first confirmation read, including UTC timestamp, current git commit, protocol commit/digest, both model revisions, and environment versions. It refuses a second campaign. A documented engineering restart may resume only the same campaign and scientific identity, and must rerun every affected condition consistently.

Every run must verify probe, target, model/tokenizer, candidate-token, site, layer, source-plan, context-norm, hypothesis-registry, and analysis-code identities. It must also verify no-op equality, hook removal, finite values, setpoint identity, context orthogonality, norm matching, factorial compatibility, and complete traces.

Confirmation artifacts are isolated under `runs/CONFIRMATION/` and never overwrite discovery. Required artifacts are `manifest.json`, `status.json`, `scalar_rows.parquet`, `factorial_rows.parquet`, `trace_rows.parquet`, `primary_hypotheses.parquet`, `secondary_metrics.parquet`, and `CONFIRMATION_SUMMARY.md`; raw rows precede aggregates.

Locked scientific implementation SHA-256 identities at preregistration:

| File | SHA-256 |
|---|---|
| `src/representation_reliability/runners/e01b2_support.py` | `9d83e5f731507d7a1a1d0ea9b1d2f93294cf39d98428f73139b4ce5a3235b854` |
| `src/representation_reliability/metrics/factorial.py` | `07d061aba02760026be7d0cc0583667ea9eda9cfdb63b64cbb02271d3a95d902` |
| `src/representation_reliability/interventions/orthogonal_context.py` | `d8d64e5f12879066c701908b454ac18dcd72e257a7843923b8764dfd8995764e` |
| `src/representation_reliability/interventions/setpoint.py` | `f74dbe927c3a5177c478f7907c9889410627d1eef6773c40631970e447109d61` |

The dedicated confirmation implementation may call these frozen scientific primitives. Any later code change must preserve their numerical definitions and be recorded in the confirmation manifest.

## Access prohibition before remote preregistration

No confirmation data may be accessed until this protocol and registry authorization are committed and the commit is present on `origin/main`. This includes examples, labels, activations, metrics, caches, and counts that require opening the split.
