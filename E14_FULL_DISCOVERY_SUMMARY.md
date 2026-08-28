# E14 Full Discovery Summary

Status: frozen full discovery complete; the E14 confirmation holdout was not accessed.

## Primary results

| Precision | D native | D frozen | B | Q | A matched-random | G matched-random |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 0.999511 | 0.999511 | 0.952778 | 0.666042 | 3.364313 | 0.173042 |
| INT8 | 0.999867 | 0.999778 | 0.963956 | 0.737917 | 3.599438 | 0.172292 |
| INT4 | 0.999911 | 0.945467 | 0.850844 | 0.783750 | 2.490854 | 0.079854 |

## General quality

| Precision | WikiText-2 PPL | HellaSwag accuracy |
|---|---:|---:|
| BF16 | 23.379758 | 0.580000 |
| INT8 | 23.188792 | 0.576000 |
| INT4 | 38.612419 | 0.510000 |

## Paired changes from BF16

### INT8

- Q: +0.071875 (95% pair-cluster CI [+0.054167, +0.090000]).
- A: +0.235125 (95% pair-cluster CI [+0.178873, +0.290751]).
- G: -0.000750 (95% pair-cluster CI [-0.020001, +0.018402]).
- B: +0.011178 (95% pair-cluster CI [+0.004800, +0.018556]).

### INT4

- Q: +0.117708 (95% pair-cluster CI [+0.059375, +0.173135]).
- A: -0.873458 (95% pair-cluster CI [-1.158646, -0.575510]).
- G: -0.093188 (95% pair-cluster CI [-0.118396, -0.069826]).
- B: -0.101933 (95% pair-cluster CI [-0.142272, -0.066419]).

## INT4 actionability retention

- R_Q: 1.176728
- R_A: 0.740375
- R_G: 0.461474

## Trace diagnosis

A/G values are matched structured minus seed-averaged random context and use each precision's frozen validation scales.

| Precision | Layer | Q q_z | Q margin_z | A q_z | A margin_z | G q_z | G margin_z |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 17 | 1.876993 | 0.396126 | 0.000001 | 1.024752 | 0.000205 | 0.007442 |
| BF16 | 20 | 1.131529 | 0.193353 | 0.387163 | 1.020591 | 0.066809 | 0.044392 |
| BF16 | 23 | 0.944992 | 0.310404 | 0.516380 | 1.047614 | 0.091022 | 0.066949 |
| BF16 | 27 | 0.687993 | 0.213666 | 0.714354 | 1.077714 | 0.101503 | 0.054351 |
| INT8 | 17 | 1.812314 | 0.398902 | 0.000029 | 1.046001 | 0.000355 | 0.008084 |
| INT8 | 20 | 1.104344 | 0.189806 | 0.409212 | 1.073342 | 0.070689 | 0.048404 |
| INT8 | 23 | 0.909475 | 0.305573 | 0.558492 | 1.104773 | 0.094922 | 0.066193 |
| INT8 | 27 | 0.657014 | 0.229293 | 0.762753 | 1.118828 | 0.106886 | 0.054681 |
| INT4 | 17 | 1.790512 | 0.235415 | 0.000341 | 0.816864 | -0.000287 | 0.009489 |
| INT4 | 20 | 0.933892 | 0.196655 | 0.404451 | 0.860574 | 0.065373 | 0.030183 |
| INT4 | 23 | 0.735858 | 0.297278 | 0.502741 | 0.752865 | 0.063033 | 0.056489 |
| INT4 | 27 | 0.338240 | 0.233016 | 0.363185 | 0.739405 | 0.024884 | 0.024292 |

## Gate and claim boundary

- E14 confirmation gate: **PASS**.
- INT4 WikiText PPL change: +65.2%; catastrophic flag: True.
- INT4 HellaSwag accuracy change: -0.070; catastrophic flag: False.
- Because the WikiText flag fired, discovery supports mixed actionability and general degradation, not selective semantic damage.

## Integrity

- BF16: complete; finite=True; no-op max deviation=0.000e+00; trace rows=26400; run=E14_98b8d199abbc.
- INT8: complete; finite=True; no-op max deviation=0.000e+00; trace rows=26400; run=E14_31ff5aa1a9a2.
- INT4: complete; finite=True; no-op max deviation=0.000e+00; trace rows=26400; run=E14_5cb70b93e672.
