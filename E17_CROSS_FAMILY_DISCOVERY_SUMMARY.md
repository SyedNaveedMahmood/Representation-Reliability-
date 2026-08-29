# E17 Cross-Family Causal-Organization Discovery

Status date: 2026-08-29. **Open discovery. The phenomenon replicated.**
E13 confirmation was not accessed, re-opened, or re-evaluated.

Protocol: `docs/E17_CROSS_FAMILY_CAUSAL_ORGANIZATION_PROTOCOL.md` (frozen at commit
`e3daa62`, before any candidate model was loaded).
Campaign: `runs/E17_CROSS_FAMILY`.

## Selected pair, and how it was selected

| Candidate | Loads | Teacher B | Teacher D | Student D | Eligible |
|---|---|---:|---:|---:|---|
| SmolLM2 (1.7B → 360M) | yes | 0.687416 | 1.000000 | 0.996080 | **no** — teacher B below the frozen 0.85 floor |
| Llama-3.2 (3B → 1B) | no | — | — | — | **no** — gated repo, `GatedRepoError 401` |
| OLMo-2 (7B → 1B) | yes | 0.986296 | 1.000000 | 0.999776 | **yes** |

```text
Q/A/G inspected during candidate selection: NO
```

Only engineering, `B` and `D` were computed during screening. No intervened
forward pass, no factorial arm, no COD, no steering. SmolLM2 was rejected on
behavior alone even though its decodability was excellent, and Llama-3.2 was
excluded because its repositories are gated and the protocol forbids changing
account permissions mid-experiment. That evidence is recorded in
`runs/E17_CROSS_FAMILY/screen/selection.json`, including an amendment giving the
true gating reason rather than the bare local-config error the screen first
logged.

**Selected: `allenai/OLMo-2-1124-7B-Instruct` → `allenai/OLMo-2-0425-1B-Instruct`.**

This pair stresses the design harder than Qwen3 did. The Qwen3 teacher and
student had equal depth and differed only in width (28×2048 → 28×1024). OLMo-2
differs in **both**: 32 blocks × 4096 → 16 blocks × 2048. The frozen relative-depth
rule `layer* = round(0.60 × (L − 1))` therefore places the teacher at layer 19 and
the student at layer 9 — different absolute sites, same relative depth, no causal
layer search.

## Frozen references

| Model | B | D native | Qz | Az | Gz | PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher OLMo-2 7B | 0.990044 | 1.000000 | 0.174767 | 1.457289 | 0.026749 | 12.782 | 0.830 |
| R0 OLMo-2 1B | 0.678044 | 0.999778 | 0.011045 | 0.388005 | -0.000365 | 18.790 | 0.700 |

Integrity on every model: no-op hook deviation `0.0`, context/direction dot
product `≤ 2.6e-16`, all factorial evidence finite, 300 evaluated rows.

## Behavior-matched results

| regime | seed | step | B | ΔB vs teacher | D native | Qz | ΔQz | Az | ΔAz | Gz | ΔGz | CKA | PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 20261705 | 25 | 1.000000 | +0.009956 | 1.000000 | 0.001575 | **-0.173192** | 1.687730 | +0.230441 | 0.012876 | -0.013874 | — | 19.192 | 0.706 |
| R1 | 20261715 | 25 | 1.000000 | +0.009956 | 1.000000 | 0.001639 | **-0.173128** | 1.406122 | -0.051167 | 0.007599 | -0.019150 | — | 19.123 | 0.700 |
| R1 | 20261725 | 25 | 1.000000 | +0.009956 | 1.000000 | 0.003202 | **-0.171565** | 1.094346 | -0.362943 | 0.013438 | -0.013311 | — | 19.053 | 0.700 |
| R2 | 20261705 | 25 | 1.000000 | +0.009956 | 1.000000 | 0.011901 | **-0.162866** | 1.202562 | **-0.254727** | 0.004553 | -0.022196 | — | 18.835 | 0.698 |
| R2 | 20261715 | 25 | 0.999222 | +0.009178 | 1.000000 | 0.011033 | **-0.163734** | 1.042113 | **-0.415176** | 0.003347 | -0.023402 | — | 18.778 | 0.700 |
| R2 | 20261725 | 25 | 1.000000 | +0.009956 | 1.000000 | 0.009175 | **-0.165592** | 1.219395 | **-0.237894** | 0.004003 | -0.022746 | — | 18.880 | 0.698 |
| R3 | 20261705 | 25 | 0.999867 | +0.009822 | 1.000000 | 0.010870 | **-0.163897** | 1.173668 | **-0.283621** | 0.006114 | -0.020635 | 0.902886 | 18.828 | 0.698 |
| R3 | 20261715 | 25 | 0.999267 | +0.009222 | 1.000000 | 0.010421 | **-0.164346** | 0.979196 | **-0.478093** | 0.003880 | -0.022869 | 0.871215 | 18.785 | 0.698 |
| R3 | 20261725 | 25 | 1.000000 | +0.009956 | 1.000000 | 0.007738 | **-0.167029** | 1.176996 | **-0.280293** | 0.004399 | -0.022350 | 0.879895 | 18.891 | 0.694 |

Regime means:

| regime | ΔB | Qz | ΔQz | Az | ΔAz | Gz | ΔGz | CKA | PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | +0.009956 | 0.002139 | -0.172628 | 1.396066 | -0.061223 | 0.011304 | -0.015445 | — | 19.123 | 0.702 |
| R2 | +0.009696 | 0.010703 | -0.164064 | 1.154690 | -0.302599 | 0.003968 | -0.022781 | — | 18.831 | 0.699 |
| R3 | +0.009667 | 0.009676 | -0.165091 | 1.109953 | -0.347336 | 0.004798 | -0.021951 | 0.884666 | 18.835 | 0.697 |

## Replication criterion

| regime | A: D ≥ 0.95 | B: ΔB > -0.03 in 2/3 | C: same-direction mismatch in 2/3 | D: no quality collapse | **Replicated** | Components |
|---|---|---|---|---|---|---|
| R1 | yes | yes (3/3) | yes | yes | **yes** | Q |
| R2 | yes | yes (3/3) | yes | yes | **yes** | Q, A |
| R3 | yes | yes (3/3) | yes | yes | **yes** | Q, A |

Component detail (seeds beyond the 0.10 SESOI in the same direction, out of 3):

| regime | Q | A | G |
|---|---|---|---|
| R1 | 3/3, mean -0.1726 | 1/3, mean -0.0612 | 0/3, mean -0.0154 |
| R2 | 3/3, mean -0.1641 | 3/3, mean -0.3026 | 0/3, mean -0.0228 |
| R3 | 3/3, mean -0.1651 | 3/3, mean -0.3473 | 0/3, mean -0.0220 |

## Replication questions

**RQ-E17.1 — Is `D_S0 ≈ D_T` before training?** Yes, and emphatically. The
untrained OLMo-2 1B student reaches `D = 0.999778` against the teacher's
`1.000000`. The semantic variable is essentially perfectly decodable in a model
that converts almost none of it into behavior (`B = 0.678`).

**RQ-E17.2 — Does SFT reach teacher-like B?** Yes. R1 reaches `B = 1.000000`
against the teacher's `0.990044` in 3/3 seeds.

**RQ-E17.3 — Does logit KD reach teacher-like B?** Yes. R2 reaches `0.999`-`1.000`,
`ΔB = +0.0097` on average, in 3/3 seeds.

**RQ-E17.4 — At B-matched checkpoints, does R2 retain a componentwise Q/A/G
mismatch?** Yes, decisively. `ΔQz = -0.164` and `ΔAz = -0.303` on average, both
beyond the SESOI in the same direction in 3/3 seeds. `ΔGz = -0.023` is inside the
SESOI in 3/3 seeds and is a null.

**RQ-E17.5 — Does hidden-state KD increase representation similarity?**
**Not answerable from E17 as run, and this is a limitation of the run, not a
finding.** Projected representation diagnostics are only computed when a
projector exists, so CKA was recorded for R3 (mean `0.885`) but not for R2 or R0,
and the student checkpoints were pruned afterwards to keep the wave inside the
device's disk budget, so the comparison cannot now be recovered without retraining.
The E13 confirmation did answer the analogous question, and answered it against
R3.

**RQ-E17.6 — Does improved representation similarity eliminate the causal
mismatch?** No. R3 achieves high teacher/student CKA (`0.871`-`0.903`) at its
B-matched checkpoints and nevertheless carries the **largest** A mismatch of any
regime (`-0.347` mean) and a Q mismatch indistinguishable from R2's (`-0.165` vs
`-0.164`). High representation similarity and large causal-organizational
mismatch coexist comfortably.

## What replicated, and what did not

The headline is that the dissociation replicates in a family with a different
architecture shape, a different depth ratio, a different tokenizer, and a
different absolute intervention site.

**Q is the family-general component.** The teacher converts the scalar semantic
coordinate into behavior at almost identical strength in both families — Qwen3
`Qz = 0.1638`, OLMo-2 `Qz = 0.1748` — and in both families every distilled student
collapses to approximately zero, giving `ΔQz` of `-0.167`/`-0.173` in Qwen3 and
`-0.164`/`-0.165` in OLMo-2. Four independent numbers across two families agree to
within `0.01` standardized units. Distillation reliably transfers the behavior and
reliably fails to transfer the scalar conversion pathway.

**A is present in both families but family-specific in direction.** Qwen3 students
*over*-use the matched-context additive pathway (`ΔAz = +0.42`/`+0.51`); OLMo-2
students *under*-use it (`ΔAz = -0.30`/`-0.35`). The magnitude is comparable, the
sign is opposite. The protocol anticipated exactly this: the replication target is
the principle, not an identical Q/A/G pattern.

**G is a consistent null in both families.** Qwen3 `ΔGz ≈ +0.012` and OLMo-2
`ΔGz ≈ -0.022`, inside the SESOI in every seed of every regime in both families.
The structured interaction term is the one component distillation does not
systematically distort — which is also the component the E13 method branch could
never learn to transfer.

**SFT is not the same as KD on A.** R1 replicates the Q deficit as strongly as the
KD regimes (`-0.173`) but its A mismatch is inconsistent across seeds (1/3 same
direction, and per-seed values spanning `+0.230` to `-0.363`). The Q deficit is the
robust part of the phenomenon; A depends on the objective.

## Quality controls

WikiText-2 perplexity moved from R0's `18.790` to `18.78`-`19.19` and HellaSwag
from `0.700` to `0.694`-`0.706` across all nine runs. No regime shows catastrophic
specialization, so the behavioral comparison is interpretable everywhere.

## Claim boundary

One non-Qwen family, one synthetic relation task with five families, one
relative-depth site, `resid_post` at `last_prompt`, 100 optimizer steps, three
seeds, 300 discovery rows. E17 is **discovery, not confirmation**: the `0.03` and
`0.10` thresholds are reused from E13 as descriptive values and no Holm-adjusted
or preregistered-confirmatory claim is made from E17. Probe results establish
decodability, not endogenous causal use.

Per the protocol, candidates 2 and 3 were not revisited after any causal outcome
was seen; selection ended before the first intervened forward pass.
