# Proposed E13-D Conversion-Response Distillation

Status: **superseded by the frozen authorized full protocol after the E13
multi-seed gate passed on 2026-08-28**. See
`docs/E13_CONVERSION_RESPONSE_FULL_PROTOCOL.md`.

The one-seed E13 pilot found saturated D and perfect B after ordinary SFT/KD,
but scalar Q remained student-like under logit KD and causal A/G did not jointly
match teacher magnitudes. This satisfies the frozen method-trigger rule.

## Proposed causal-response target

Keep the teacher frozen. On the fresh E13 training corpus, cache its
validation-standardized native-margin response to source-free semantic
perturbations at predeclared `delta_z in {-1,+1}`:

```text
r_T(delta_z) = Delta margin_T / sigma_margin_T
```

Train an otherwise identical R2 student with:

```text
L = L_logit_KD + lambda_response * mean_delta_z(
      r_S(delta_z) - stopgrad(r_T(delta_z))
    )^2
```

The response term transfers a causal input-output function rather than hidden
vectors of incompatible dimensionality.

## Required controls before authorization

- at least three seeds for R0/R1/R2 first;
- frozen general language-quality evaluation;
- equal-update/equal-teacher-forward R2 control;
- random-direction response-matching control;
- validation-only choice of one response coefficient before discovery;
- raw and standardized Q/A/G, not behavior alone;
- an untouched E13-specific confirmation namespace that remains unmaterialized
  until a separately preregistered discovery gate passes.

Do not implement this method from the present one-seed result alone.
