# Research Charter: Representation Reliability

## Unifying problem

Several properties are often treated as interchangeable:
1. a variable is decodable;
2. the model can access it;
3. the model causally uses it;
4. it can be externally manipulated;
5. a monitor built on it is trustworthy;
6. it is robust across perturbations.

They are not logically equivalent.

Let \(h_{\ell,t}(x)\) be an internal state, \(z(x)\) a target variable, and \(F(x)\) model behavior.

### Decodability
\[
\hat z=g(h_{\ell,t}(x)).
\]
High held-out performance means a decoder can access information correlated with \(z\). It does not prove the model uses it.

### Causal use
For matched examples:
\[
h'_{\ell,t}(x_a)\leftarrow h_{\ell,t}(x_b).
\]
Behavior changing toward the predeclared counterfactual supports causal relevance.

### Steerability
\[
h'=h+\alpha v
\]
or a general \(I_\theta(h)\). Reliability requires efficacy, controllable range, dose-response structure, and limited collateral drift.

### Monitorability
\[
m(h)\approx P(E=1\mid h)
\]
where \(E\) is a precisely defined error/failure label. Monitor reliability includes calibration, OOD transfer, and attack resistance.

### Robustness
For invariant \(T\), compare reliability profiles on \(x\) and \(T(x)\). D, C, S, and M may fail independently.

### Collateral safety
Report target change and unrelated utility drift separately. A convenience ratio may be:
\[
K=\frac{|\Delta B|}{\epsilon+\|\Delta U\|_2}.
\]

## Gap taxonomy

\[
G_{DC}=D-C
\]
Decode-use gap.

\[
G_{CS}=C-S
\]
Causal-steering gap.

\[
G_{MO}=M_{\mathrm{corruption}}-M_{\mathrm{output\ error}}
\]
Internal-alarm / output gap.

\[
G_{SR}=S-R_S
\]
Steering-robustness gap.

## Claim ladder

| Evidence | Allowed claim |
|---|---|
| probe | decodable / encoded / correlated |
| probe + surface controls | internally decodable beyond tested surface baselines |
| causal patch | causally relevant under tested intervention |
| predicted counterfactual swap | supports functional alignment with variable |
| steering | controllable under tested operator |
| multiple operators | control not specific to one parameterization |
| transformations/models | robust within tested scope |
| adversarial tests | robust to tested attack family |

## Confounders to actively test

- probe expressivity;
- lexical leakage;
- model-generated labels;
- superposition/polysemanticity;
- off-manifold interventions;
- norm mismatches;
- base/source nuisance mismatch;
- prompt/chat-template differences;
- token-site drift;
- layer numbering mismatch;
- output scoring artifacts;
- benchmark contamination;
- CoT faithfulness assumptions;
- post-selection after scans;
- model-family module-path differences.

## Evidence ladder

```text
observation
-> matched counterfactual
-> causal intervention
-> negative controls
-> transformation robustness
-> held-out confirmation
-> second task/model
```

A robustness failure can itself be the phenomenon.
