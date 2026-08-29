# E13 Teacher-Cache Hidden-Space Bugfix Record

Date: 2026-08-29  
Scientific protocol: unchanged (`3f3dd9a65347fc9ba6a20c29686aba11bd578f52b28d87818c399b422325846b`)

## Reproducer and root cause

`python -m representation_reliability.cli e13-method-cache` failed in
`source_free_setpoint_delta` with `ValueError: base/direction shape mismatch`.
The base activation was a freshly extracted Qwen3-1.7B teacher activation with
shape `(2048,)`. `prepare_teacher_response_cache` had called `_load_reference`,
which loads `initial_student_reference.npz`; its semantic direction had shape
`(1024,)` and originated from the Qwen3-0.6B frozen R0 reference.

That student reference incorrectly entered teacher R4 semantic edits, R5
semantic/context geometry, R6 teacher random-direction geometry, and the
teacher-cache live validation. The first R5 semantic delta rejected the shape
mismatch before any response cache or method checkpoint was written.

## Scientific impact

No method training job started, no method result was generated, and the locked
E13 confirmation namespace was not accessed. Completed R0/R1/R2/R3 discovery
evidence is unaffected. This is an implementation-conformance defect, not a
change to the frozen method definition or a scientific result.

## Fix

Teacher cache construction now fits and freezes a teacher-native reference from
teacher train/validation activations. Student method training continues to fit
and freeze a student-native reference. A model-local reference records role,
hidden size, semantic direction, q/margin scales, probe and target digests, and
resolved revision. Role, revision, activation, direction, and delta shapes are
asserted before geometry or intervention use.

R5 matched context and R6 random directions are constructed independently in
the runtime model's hidden space. Cross-model transfer remains exclusively a
comparison of standardized scalar responses; no residual vector is projected,
padded, truncated, or shared.

## Regression coverage

Tests cover valid 2048-d teacher and 1024-d student semantic edits, rejection in
both cross-space directions, teacher-cache rejection of a student reference,
factorial Q/A/G construction at both widths, model-local deterministic R6
orthogonality and norm matching, cache digests/provenance, and live-validation
identity fields. The existing R2-C zero-gradient and hook-cleanup GPU contract
also remains passing.

## Live-validation batching correction

The first cache attempt after the hidden-space fix reached the frozen live gate
and stopped. Its hashed row-level subset regrouped prompts into different
eight-row padded batches than cache construction. In BF16 this changed raw
teacher margins by as much as 0.5 and standardized responses by as much as
0.1186, exceeding the unchanged `0.02`/`0.002` tolerances. No cache was saved
and no training started.

The live subset now deterministically selects two complete original cache
batches, preserving row order, padding shape, weights, dtype, and token sites.
This corrects inference identity rather than loosening tolerance. A regression
test proves that all 16 selected rows retain their original batch boundaries.
