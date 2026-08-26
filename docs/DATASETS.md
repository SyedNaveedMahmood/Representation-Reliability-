# Dataset Strategy

Use controlled datasets first, natural benchmarks second.

## Synthetic relational pairs
Generate exact latent-variable examples:
```text
Premise: Luma is north of Reko.
Question: Is Reko south of Luma?
Target: yes
```

Counterfactual operations:
- relation inversion;
- entity swap;
- premise corruption.

Advantages:
- exact labels;
- matched counterfactuals;
- no benchmark contamination.

## Formal logic carriers
Use/extend Reasoning-Flow's carrier-invariant logic setup.
Vary surface/topic while holding logical structure fixed.

## False-premise correction
Structure:
```text
Initial statement: X
Correction: Actually not X; Y.
Question requiring X/Y distinction.
```
Useful for:
- internal alarms;
- correction dynamics;
- KV cache scars.

## Natural datasets
- TriviaQA: correctness monitors.
- GSM8K: reasoning trajectories/commitment.
- NQ-Open, SQuAD, BioASQ: OOD monitor transfer.
- CounterFact: factual causal tracing.

## Transform library

Invariant:
- paraphrase;
- equivalent question framing;
- benign distractor;
- clause reorder;
- safe synonym substitution.

Controlled-change:
- negation;
- entity swap;
- relation inverse;
- numeric perturbation;
- premise corruption.

Every transform stores parent ID, transform class, and expected target relation.

## Splitting
For synthetic data, split by templates/entities where possible.

Suggested probing split:
- train 60%;
- val 15%;
- discovery-test 15%;
- confirmation 10%.

## Size ladder
- smoke 50–100;
- pilot 500–2,000;
- discovery 2,000–10,000.

Do not start with 50k if 1k can kill the idea.
