# Metrics

No single scalar is authoritative. Report the reliability vector and gaps.

## D — Decodability
Primary:
- AUROC;
- AUPRC;
- balanced accuracy.

If used probabilistically:
- Brier score;
- ECE.

Controls:
- majority;
- random label;
- text/surface baseline.

## C — Causal use

Normalized counterfactual logit recovery:
\[
C_{\text{norm}}=
\frac{m(F(h_a\leftarrow h_b))-m(F(a))}
{m(F(b))-m(F(a))+\epsilon}.
\]

Also:
- counterfactual accuracy;
- paired ATE;
- necessity/sufficiency-style tests where well-defined.

## S — Steerability
For \(B(\alpha)\), report:
- maximum target effect within safety budget;
- area under response curve;
- Spearman \(\rho(\alpha,B)\);
- monotonicity violations;
- saturation;
- sign reversals;
- bidirectional control where relevant.

Efficiency:
\[
S_{\text{eff}}=\frac{\Delta B}{\|\Delta h\|+\epsilon}.
\]

## M — Monitorability
- AUROC;
- AUPRC;
- Brier;
- ECE;
- risk-coverage;
- selective accuracy.

Do not merge distinct labels like corruption and final answer error.

## R — Robustness
For transform family:
- raw metric delta;
- normalized retention;
- layer-rank Spearman;
- top-k Jaccard;
- direction cosine.

Direction similarity is not causal stability.

## K — Collateral safety
Report:
- target delta;
- each utility delta;
- output KL/JS on controls;
- refusal drift;
- length/verbosity drift where relevant.

Convenience:
\[
K=\frac{|\Delta B|}{\epsilon+\|\Delta U\|_2}.
\]

## Gap metrics
\[
G_{DC}=D^*-C^*
\]
\[
G_{CS}=C^*-S^*
\]
\[
G_{MO}=M_{\mathrm{corruption}}-M_{\mathrm{output-error}}
\]
\[
G_{SR}=S^*-R_S^*
\]

Use normalized versions only for visualization. Retain raw metrics.

## Statistics
Minimum for escalated findings:
- 3 seeds;
- 95% bootstrap CI;
- paired bootstrap for interventions;
- explicit n.

Discovery scan picks sites on discovery split. Confirmation uses untouched data.

Correct multiple comparisons for confirmatory head/layer inference.
