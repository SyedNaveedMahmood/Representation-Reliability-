# Falsification and Kill Gates

Early experiments should kill weak hypotheses cheaply.

## Gate 0 — pipeline validity
Require:
- non-trivial model task performance;
- validated evaluator;
- correct token resolution;
- no duplicate leakage;
- cache/sample identity integrity;
- alpha=0 matches baseline within tolerance.

Fail -> engineering issue, no scientific interpretation.

## Gate 1 — representation signal
Probe must beat:
- random labels;
- majority;
- meaningful surface/text baseline.

Fail -> do not spend heavily on causal scanning unless independently motivated.

## Gate 2 — intervention sanity
Target effect must exceed:
- norm-matched random vector;
- shuffled source;
and must not require absurd `||delta||/||h||`.

Fail -> intervention artifact candidate.

## Gate 3 — held-out confirmation
Select sites on discovery data, confirm direction/magnitude/control separation on untouched data.

Fail -> exploratory overfit.

## Gate 4 — transformation test
At least one innocuous transform family.

Possible good outcomes:
- effect persists;
- D persists but C disappears;
- site migrates systematically.

Chaotic collapse -> no general mechanism claim.

## Gate 5 — second context
Before broad claim: second task or second model.

## E00
Continue if D is reproducible above controls.
Kill/redirect if text baseline explains it.

## E01
Candidate D/C gap if rankings/effects diverge beyond controls.
Reject gap if D and C strongly align everywhere tested.

## E02
Candidate if high-D sites remain weakly steerable under safety budget.
Do not call "unsteerable" until a second operator is tested later.

## E03
Candidate if replacement works but additive steering does not.
Escalate to rotation/ReFT to distinguish operator failure.

## E04
Candidate if corruption-state monitoring is strong but final-error monitoring is much weaker.
Strong version: manipulating monitor-associated state does not proportionally fix output.

## E05
Candidate if change-point-ranked reasoning steps are more causally sensitive than position/length-matched controls.

## E06
Candidate if probe-attributed and causal component sets have low overlap but each method is self-stable.

## E07
Especially strong finding: activation/probe invariance stays high while causal effect changes.

## E08
Candidate if SAE activation consistency fails to predict causal consistency.

## E09
Candidate cache scar if targeted K/V intervention recovers obsolete influence after correction and localizes to reproducible windows.

## E10
Candidate monitor spoofing if monitor score moves strongly under small intervention while output KL/accuracy stay nearly fixed.

## E11
Candidate if required control rank differs systematically by behavior/layer.
