# E13 Multi-Seed Causal-Organization Transfer Discovery

Status date: 2026-08-28. Full open discovery only; E13 confirmation was not accessed.

Protocol SHA-256: `04daa7fcc66cc1c93f8077de23962dfec9861c9412c44367d83603ed0ccb7cac`  
Campaign: `runs\E13_MULTI_SEED\E13MS_04daa7fcc66c`

## Frozen references

| Model | B | D native | D frozen | Qz | Az | Gz | COD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 0.964933 | 1.000000 | 1.000000 | 0.163981 | 1.168644 | 0.046239 | 0.000000 |
| R0 | 0.747000 | 1.000000 | 1.000000 | -0.000147 | 0.787599 | 0.006628 | 0.799059 |

## Behavior-matched discovery

| regime | seed | selected_step | B_student | validation_B_student | validation_B_teacher | absolute_validation_B_gap | Q_z | A_z | G_z | COD |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 20261305 | 10 | 0.987267 | 0.982976 | 0.973608 | 0.009368 | 0.017990 | 1.219880 | 0.013562 | 0.551613 |
| R1 | 20261315 | 10 | 0.993800 | 0.996984 | 0.973608 | 0.023376 | 0.017551 | 1.368360 | 0.011414 | 0.585550 |
| R1 | 20261325 | 10 | 0.999511 | 0.999888 | 0.973608 | 0.026280 | 0.017897 | 1.523447 | 0.021694 | 0.550577 |
| R2 | 20261305 | 10 | 0.957511 | 0.947408 | 0.973608 | 0.026200 | 0.011149 | 1.289121 | 0.037009 | 0.733286 |
| R2 | 20261315 | 10 | 0.962356 | 0.959080 | 0.973608 | 0.014528 | -0.003058 | 1.395682 | 0.063449 | 0.876203 |
| R2 | 20261325 | 10 | 0.996111 | 0.990304 | 0.973608 | 0.016696 | -0.004694 | 1.554582 | 0.056832 | 0.742157 |
| R3 | 20261305 | 25 | 1.000000 | 1.000000 | 0.973608 | 0.026392 | -0.007894 | 1.738960 | 0.051873 | 0.707304 |
| R3 | 20261315 | 10 | 0.960200 | 0.959552 | 0.973608 | 0.014056 | 0.001315 | 1.402560 | 0.056914 | 0.893400 |
| R3 | 20261325 | 10 | 0.997200 | 0.991216 | 0.973608 | 0.017608 | -0.005299 | 1.553740 | 0.059490 | 0.743829 |

## Frozen method gate

- Gate A: PASS
- Gate B: PASS
- Gate C: PASS
- Gate D: PASS
- Gate E: PASS

**CONVERSION-RESPONSE METHOD AUTHORIZED**

Raw, validation-z, bounded-probability, strict-flip, per-example COD, representation-similarity, and quality evidence are retained in the campaign directory. Claims remain discovery-only and model/task/site specific.
