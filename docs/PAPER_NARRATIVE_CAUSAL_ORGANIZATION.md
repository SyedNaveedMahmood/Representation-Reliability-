# Paper Narrative — Causal Organization of an Actionable Representation

Status date: 2026-08-29. Written after the E13 diagnostic confirmation (strong)
and the E17 cross-family replication (replicated).

## 1. Final supported thesis

> A semantic variable can be equally and perfectly decodable in two models that
> nevertheless implement different **causal organizations** of it. Distillation
> reproduces a teacher's behavior — here, behavior that matches or exceeds the
> teacher — and can raise representation similarity, without reproducing that
> causal organization. The specific thing that fails to transfer is the scalar
> semantic conversion pathway, and it fails by almost exactly the same amount in
> two unrelated model families, while the compensating distortion is
> family-specific.

The unit of analysis is the causal-organization profile

```text
C = (D, Q, A, G, B)
```

where `D` is probe decodability, `B` is native target-margin AUROC, and `Q`, `A`,
`G` are the scalar, additive-context, and interaction terms of a four-arm
factorial intervention. These are **components of a profile, not a hierarchy**.

## 2. Prior-work boundary

Prior work establishes that a variable can be decodable without being used —
the decoding/causal-use divergence. This project does not re-litigate that. Its
contribution starts one level in: **causal use is itself structured**, and that
structure can differ between models that agree on both the decodable content and
the observable behavior. The novel objects are the componentwise `Q`/`A`/`G`
profile, the behavior-matched checkpoint comparison, and the finding that the
components dissociate from one another under distillation.

## 3. Confirmed results

**E01 / E01B (confirmed).** Two Qwen3 checkpoints with near-saturated
decodability implement different causal organizations. The scalar q-coordinate is
causal; substantial causal information lies outside it; Qwen3-1.7B additionally
shows a structured q-by-context interaction not detected in 0.6B. Mechanism
frozen against further tuning.

**E14 (confirmed).** INT4 preserves precision-native decodability far better than
`A`/`G` actionability, with the honest boundary that generic LM quality also
degrades — so this is mixed actionability/general degradation, not selective
semantic damage.

**E13 diagnostic (confirmed, this campaign).** One-shot access to an untouched
200-row / 100-pair holdout, access count 1. Hierarchical gatekeeping:

* Stage A, behavioral non-inferiority against the frozen `δ_B = 0.03`: both logit
  KD and hidden-state KD are non-inferior — in fact superior — to the teacher in
  3/3 seeds (`ΔB = +0.044`, `+0.055`).
* Stage B, componentwise mismatch against the frozen `δ_C = 0.10` SESOI with Holm
  within regime: `Q` and `A` are systematically mismatched in both regimes in 3/3
  seeds (`ΔQz = -0.167`/`-0.173`, Holm `0.0003`; `ΔAz = +0.421`/`+0.508`, Holm
  `0.0004`/`0.0003`). `G` is **not** mismatched (Holm `1.0`) and is reported as a
  genuine null.
* `D_native = 1.000000` for all eight evaluated models, so the mismatch is not a
  decodability deficit.

**E13 method branch (closed, negative).** Conversion-response distillation failed
its frozen success criterion. The mechanism controls showed its apparent gain was
generic local-sensitivity regularization: family-shuffled teacher targets with no
sample-level semantic correspondence achieved the campaign's lowest COD. This is
retained as supporting evidence and as the reason COD is not used as a headline
statistic anywhere in the paper.

## 4. Cross-family status

**E17 (discovery, replicated).** Candidate pairs were screened on engineering,
`B` and `D` only — no `Q`/`A`/`G`, no COD, no intervened forward pass — so no pair
could be chosen for reproducing the desired mechanism. SmolLM2 was rejected on
teacher behavior (`B = 0.687 < 0.85`); Llama-3.2 was excluded as a gated repo; the
frozen third candidate, OLMo-2 7B → 1B, was selected and locked.

The phenomenon replicated in all three trained regimes. `D = 1.000` throughout,
students exceed the teacher behaviorally in 3/3 seeds, and `ΔQz = -0.164`/`-0.165`
for logit and hidden-state KD, in the same direction, in 3/3 seeds.

The cross-family structure:

| component | Qwen3 | OLMo-2 | reading |
|---|---:|---:|---|
| `ΔQz` | -0.167 / -0.173 | -0.164 / -0.165 | **family-general deficit** |
| `ΔAz` | +0.421 / +0.508 | -0.303 / -0.347 | present in both, **opposite sign** |
| `ΔGz` | +0.013 / +0.012 | -0.023 / -0.022 | **null in both** |

## 5. Remaining experiments

None is required before writing. Optional strengtheners, in priority order:

1. A within-family R2-vs-R3 representation-similarity comparison inside E17
   (E17 recorded CKA only for R3, and its checkpoints were pruned for disk).
2. A non-synthetic task, to move beyond the five relation families.
3. A second site or relative depth, to test whether the `Q` deficit is site-specific.
4. E16 training-emergence, if a suitable checkpoint family is available — this
   would turn the static dissociation into a developmental one.

E15 remains high-upside and high-cost; it should not be used merely as an agent
benchmark.

## 6. Claim limitations

* One synthetic relation task, five families, one `resid_post` site at
  `last_prompt`, one relative depth, 100 optimizer steps, three seeds per regime.
* Probes establish decodability, not endogenous causal use; the model is not shown
  to use exactly the intervened axis unprompted.
* E17 is discovery, not confirmation. Its `0.03`/`0.10` thresholds are reused
  descriptively; no Holm-adjusted claim is made from it.
* The `A` result does not generalize in sign across families and must be reported
  as family-specific.
* Behavior-matched checkpoints in both experiments are early (steps 10-25), so
  statements about hidden-state KD describe that operating point, not the
  objective at convergence.
* The E13 holdout is consumed (access count 1) and may never be reused.
* Nothing here supports the claim that causal organization can be *transferred*.
