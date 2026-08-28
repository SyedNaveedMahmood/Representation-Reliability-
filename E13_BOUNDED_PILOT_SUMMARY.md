# E13 Bounded Distillation Reliability Pilot

Status date: 2026-08-28. One frozen seed completed for R0 frozen student, R1
hard-label SFT, and R2 full-vocabulary logit KD. This is pilot-only evidence.
E13 confirmation was not materialized or accessed.

Run: `runs/E13/E13_432643013985-r2`  
Teacher: `Qwen/Qwen3-1.7B`  
Student: `Qwen/Qwen3-0.6B`  
Open corpus: 4,000 train / 500 validation / 300 discovery-test directed rows  
Runtime: 442.8 seconds  
Peak allocated VRAM: 9.63 GiB

## Frozen checkpoint trajectory

All Q/A/G values are raw target-oriented Yes-minus-No margin effects. D frozen
uses the initial student axis for student checkpoints. Teacher uses its own
train/validation-only axis and targets.

| Regime | Step | B AUROC | D native | D frozen | Q | A matched-random | G matched-random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 0 | 0.964933 | 1.000000 | 1.000000 | 0.517917 | 3.624861 | 0.163194 |
| R0 | 0 | 0.747000 | 1.000000 | 1.000000 | -0.000208 | 1.113819 | -0.000486 |
| R1 | 0 | 0.747000 | 1.000000 | 1.000000 | -0.000208 | 1.113819 | -0.000486 |
| R1 | 10 | 0.987267 | 1.000000 | 0.989422 | 0.054167 | 3.784028 | 0.034583 |
| R1 | 25 | 1.000000 | 1.000000 | 1.000000 | 0.078750 | 24.034653 | 0.416250 |
| R1 | 50 | 1.000000 | 1.000000 | 1.000000 | 0.113333 | 28.259375 | 0.475417 |
| R1 | 100 | 1.000000 | 1.000000 | 1.000000 | 0.108750 | 28.300972 | 0.487014 |
| R2 | 0 | 0.747000 | 1.000000 | 1.000000 | -0.000208 | 1.113819 | -0.000486 |
| R2 | 10 | 0.957511 | 1.000000 | 0.980844 | 0.030000 | 3.460000 | 0.105972 |
| R2 | 25 | 1.000000 | 1.000000 | 1.000000 | -0.027083 | 8.983194 | 0.235417 |
| R2 | 50 | 1.000000 | 1.000000 | 1.000000 | 0.002500 | 9.341111 | 0.200208 |
| R2 | 100 | 1.000000 | 1.000000 | 1.000000 | -0.002083 | 9.186111 | 0.191944 |

At step 100, R1 A has CI `[28.1083, 28.4818]` and G has CI
`[0.4530, 0.5213]`. R2 A has CI `[8.9359, 9.4213]` and G has CI
`[0.1783, 0.2047]`. R0 G includes zero. All intervals are 500-draw
pair-cluster bootstrap intervals.

## Training and integrity

R1 loss fell from 0.7318 to 0.000002; R2 combined loss fell from 3.9165 to
1.1544. Pre-clip gradient norms were large but finite; every update was clipped
to 1.0. Every evaluation had exact no-op equivalence, finite Y arms, and maximum
context dot truth direction below `1.8e-15`. No checkpoint failed a numerical
gate.

Corpus generation removed 103 train, 32 validation, and 23 discovery candidate
pair collisions before reaching the frozen quotas. All retained prompts are
unique and all counterfactual pairs remain within one split.

## Pilot-only interpretation

1. **D is already high.** Native and frozen-axis D are 1.0 at R0 and return to
   1.0 after a small early rotation under both objectives.
2. **SFT changes causal organization strongly.** R1 improves B and moves Q
   modestly, but drives A and G far beyond the teacher rather than reproducing
   teacher magnitudes.
3. **Logit KD changes A/G but not Q.** R2 reaches perfect task-margin ranking,
   A rises substantially, and G approaches/slightly exceeds the teacher, while
   Q remains essentially at the R0 value.
4. **KD differs from SFT.** R1 produces much larger A/G than R2 under the same
   update budget. Neither objective transfers the teacher's full Q/A/G vector.
5. **A/G change before Q.** This bounded result is a selective hierarchy, not
   joint transfer. Calling the oversized effects teacher-like transfer would be
   misleading; they are changes in causal organization.
6. **Behavior can improve without equivalent causal-structure transfer.** Both
   regimes reach B=1.0 while Q remains far below the teacher, and A/G do not
   numerically match all teacher components.
7. **Full E13 discovery is justified but not authorized here.** The next
   standard experiment should replicate R0/R1/R2 over at least three seeds and
   add frozen general-quality controls before making a transfer claim.
8. **Conversion-response distillation is scientifically triggered.** R2
   improves B with D already saturated while the final A gap is large (R2 9.19
   versus teacher 3.62) and Q remains untransferred. The method is proposed only;
   it was not implemented or run.

## Claim boundary

This one seed does not establish that KD generally transfers G, that SFT is
inferior, or that larger raw A/G is better. Fine-tuning may alter output scale
and task specialization. Multi-seed replication and general-quality controls
are required before a scientific conclusion.
