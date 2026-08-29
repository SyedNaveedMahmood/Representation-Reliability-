# Causal Organization Across Model Families — Synthesis

Status date: 2026-08-29. Combines confirmed Qwen evidence (E01, E13) with
cross-family discovery evidence (E17).

## 1. Main comparison

| Family | Teacher | Student | D relation | B relation after KD | Q mismatch | A mismatch | G mismatch |
|---|---|---|---|---|---|---|---|
| Qwen3 (E13, **confirmed**) | Qwen3-1.7B | Qwen3-0.6B | `D = 1.000` in both, and in the untrained student | student **exceeds** teacher, `ΔB = +0.044` (R2) / `+0.055` (R3), 3/3 seeds | **yes**, `-0.167` (R2) / `-0.173` (R3), Holm `0.0003` | **yes**, `+0.421` (R2) / `+0.508` (R3), Holm `0.0004`/`0.0003` | **no**, `+0.013`/`+0.012`, Holm `1.0` |
| OLMo-2 (E17, discovery) | OLMo-2-1124-7B | OLMo-2-0425-1B | `D = 1.000` trained, `0.9998` untrained | student **exceeds** teacher, `ΔB = +0.0097` (R2) / `+0.0097` (R3), 3/3 seeds | **yes**, `-0.164` (R2) / `-0.165` (R3), 3/3 seeds | **yes**, `-0.303` (R2) / `-0.347` (R3), 3/3 seeds | **no**, `-0.023`/`-0.022`, 0/3 seeds |
| Qwen3 (E01, confirmed) | — | Qwen3-1.7B vs 0.6B, no distillation | near-saturated in both | — | scalar q-coordinate is causal in both | distributed causal support beyond the scalar | structured q-by-context interaction present in 1.7B, not detected in 0.6B |

The two distillation systems differ in family, parameter ratio (2.8× vs 4.7×),
architecture shape (equal-depth/narrower vs shallower-and-narrower), tokenizer,
vocabulary, and absolute intervention layer (17 vs teacher-19/student-9). The
dissociation survives all of it.

## 2. What is family-general

**The scalar-conversion deficit is the invariant.** Both teachers convert the
scalar semantic coordinate into behavior at nearly the same standardized
strength, and every distilled student in both families collapses that conversion
to approximately zero:

| quantity | Qwen3 | OLMo-2 |
|---|---:|---:|
| teacher `Qz` | 0.1638 | 0.1748 |
| student `Qz`, logit KD | -0.0037 | +0.0107 |
| student `Qz`, hidden-state KD | -0.0093 | +0.0097 |
| `ΔQz`, logit KD | **-0.1675** | **-0.1641** |
| `ΔQz`, hidden-state KD | **-0.1731** | **-0.1651** |

Four independent `ΔQz` estimates from two families agree within `0.01`
standardized units, in the same direction, in 3/3 seeds each, at checkpoints
where the students match or beat the teacher behaviorally and where the variable
is perfectly decodable. This is the strongest quantitative result in the project.

**The `G` null is also family-general.** The structured interaction term sits
inside the `0.10` SESOI in every seed, every regime, both families. Distillation
does not systematically distort it — and, from the closed E13 method branch, it
is also the component no conversion-response objective could learn to transfer.

## 3. What is family-specific

**The sign of the `A` mismatch.** Qwen3 students *over*-rely on the matched-context
additive pathway (`ΔAz = +0.42`/`+0.51`); OLMo-2 students *under*-rely on it
(`ΔAz = -0.30`/`-0.35`). Comparable magnitude, opposite direction. Whatever
compensates for the lost scalar conversion is not the same mechanism in the two
families.

This is why the replication target was frozen as the broader principle rather
than a specific Q/A/G signature. Had the criterion demanded an identical pattern,
this would have read as a failure; it is instead the more interesting result —
the *deficit* is universal, the *compensation* is idiosyncratic.

**Whether SFT reproduces the A effect.** In OLMo-2, hard-label SFT reproduces the
Q deficit as strongly as KD (`-0.173`) but its A mismatch is inconsistent across
seeds (1/3). The Q deficit is objective-independent; A is objective-dependent.

## 4. Representation similarity does not rescue causal organization

Confirmed in Qwen3: at B-matched checkpoints, hidden-state KD's CKA (`0.704`
mean) was **below** logit KD's (`0.770`) and below the untrained student's
(`0.745`), while its causal mismatch was, if anything, larger. Observed in
OLMo-2: R3 reaches high CKA (`0.885` mean) and carries the **largest** A mismatch
of any regime (`-0.347`) with a Q mismatch indistinguishable from R2's.

So across both families, high or improved teacher/student representation
similarity coexists with large causal-organizational mismatch. Explicitly
optimizing hidden-state agreement does not buy causal-organizational agreement.

A caveat stated plainly: E17 recorded CKA only for R3, because projected
diagnostics require a projector, and its student checkpoints were pruned to fit
the disk budget. So the *within-E17* R2-vs-R3 similarity comparison is not
available. The E13 confirmation supplies that comparison; E17 supplies only the
"high similarity, still mismatched" half.

## 5. COD is not the right headline statistic anywhere

The closed E13 method branch established that Causal Organization Distance is a
magnitude-dominated norm which improves under generic response regularization:
family-shuffled teacher targets carrying no sample-level semantic correspondence
achieved the lowest COD of that entire campaign. COD was therefore excluded from
the E13 confirmation and from E17 primary comparison, and componentwise `Q`/`A`/`G`
gaps were used instead. That decision is what made the family-general `Q` result
and the family-specific `A` result visible at all; a single scalar distance would
have blurred them together.

## 6. Reassessed paper narrative

E13 confirmed and E17 replicated, so the strongest supported narrative is
available:

> Existing work shows that decodability and causal use can diverge. We show that
> causal use itself has internal organization: scalar semantic efficacy,
> distributed causal support, and contextual interaction can differ across models
> even when information is equally decodable. More importantly, distillation can
> reproduce behavior — here, behavior that matches or exceeds the teacher — and
> improve representation alignment without reproducing this causal organization,
> and the dissociation generalizes beyond one model family.

With one sharpening the data now supports and the earlier draft did not: the
*specific* thing distillation fails to transfer is the scalar-conversion pathway,
and it fails to transfer it by almost exactly the same amount in two unrelated
families, while the compensating distortion differs by family.

## 7. What is not claimed

* **Not** that causal organization can be transferred. The conversion-response
  method branch (E13 R4-R16) failed its frozen success criterion and is closed.
  Nothing in the confirmation or the replication revives it.
* **Not** that `D`, `Q`, `A`, `G`, `B` form a strict causal hierarchy. They are
  components of a causal-organization profile.
* **Not** that E17 is confirmatory. It is discovery, reusing E13's frozen
  thresholds descriptively, with no Holm-adjusted claim.
* **Not** that the probes establish endogenous causal use. They establish
  decodability.
* **Not** that this generalizes to real tasks, other sites, other layers, or
  longer training. Both experiments use one synthetic relation task, one site,
  one selector, 100 optimizer steps.

## 8. Status of each experiment

| Experiment | Status |
|---|---|
| E01 / E01B | discovery + strong confirmation complete; mechanism frozen |
| E14 | strong confirmation complete; mixed actionability/general degradation |
| E13 diagnostic | **strong confirmation complete**, holdout consumed (access count 1) |
| E13 method branch (R4-R16) | closed as a negative result; not to be reopened |
| E17 | cross-family discovery complete; phenomenon replicated |
| E15, E16 | proposed, not authorized, not started |

## 9. Recommendation

```text
paper-ready: begin writing
```

The core claim is confirmed in one family under preregistration and replicated in
a second family under a frozen no-peeking screen, with the mechanism localized to
a specific component that behaves identically across families. The remaining
uncertainties — the family-specificity of `A`, the missing within-E17 R2/R3
similarity comparison, generalization beyond the synthetic task — are honest
limitations to state in the paper, not blockers that another experiment must
clear first.
