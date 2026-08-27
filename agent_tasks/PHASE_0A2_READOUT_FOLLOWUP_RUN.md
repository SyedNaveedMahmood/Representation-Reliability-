# Coding Agent Handoff — Pull and Run Phase 0A.2 Readout Follow-up

The code changes for this follow-up have already been implemented and pushed.
Do **not** redesign the experiment, modify scientific code, or begin E01.

Your job is only to:

1. pull the latest `main`;
2. run tests;
3. run the new readout follow-up for Qwen3-0.6B;
4. run the same follow-up for Qwen3-1.7B;
5. report the generated metrics and run directories.

## Commands

From the repository root on Windows PowerShell:

```powershell
git pull origin main

.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe -m representation_reliability.cli e00c-followup `
  --model configs/models/qwen3_0.6b.yaml `
  --experiment configs/experiments/E00C_readout_diagnosis.yaml `
  --layer 17

.\.venv\Scripts\python.exe -m representation_reliability.cli e00c-followup `
  --model configs/models/qwen3_1.7b.yaml `
  --experiment configs/experiments/E00C_readout_diagnosis.yaml `
  --layer 17
```

If the package is not installed editable in the active virtual environment,
run this once before the commands above:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Do not install or upgrade unrelated packages.

## What the new run computes

For each model it performs only two narrow corrections:

1. validation-only calibration of both raw-completion and Qwen non-thinking
   chat Yes/No margins;
2. layer-17 truth probing in the model's **actual final-normalized hidden
   space**, followed by a signed cosine against the exact LM-head
   Yes-minus-No direction in that same coordinate system.

It also verifies that LM-head logits reconstructed from the normalized states
match the existing exact `final_norm + LM head` computation.

## Expected output location

New runs are created under:

```text
runs/E00CF/
```

Each completed run should include:

```text
config.resolved.yaml
manifest.json
status.json
behavior_followup_predictions.parquet
readout_followup_metrics.json
READOUT_FOLLOWUP_SUMMARY.md
```

The activation cache is stored under the schema-v2 E00CF cache namespace.

## Return only this report

### Tests

```text
pytest:
```

### Qwen3-0.6B

```text
run_dir:
raw threshold-0 balanced accuracy:
raw calibrated balanced accuracy:
raw margin AUROC:
chat threshold-0 balanced accuracy:
chat calibrated balanced accuracy:
chat margin AUROC:
chat selected validation threshold:

normalized-space probe AUROC:
native-readout AUROC:
signed probe/native cosine:
absolute probe/native cosine:
probe/native-margin correlation:
probe/raw-behavior-margin correlation:
native/raw-behavior-margin correlation:
readout reconstruction max abs deviation:
behavior-error subset n:
behavior-error subset probe AUROC:
```

### Qwen3-1.7B

Return the same fields.

### Interpretation

Answer only:

1. Does calibration explain most of the 0.6B chat failure?
2. Does the mathematically corrected normalized-space geometry still support
   weak probe/native-readout alignment at 0.6B?
3. Does 1.7B show stronger alignment than 0.6B?
4. Based on these two checks, is the repository ready for a targeted E01?

Do **not** implement E01 and do not edit the diagnostic report yet. Return the
numbers first so the next scientific decision can be made from the corrected
evidence.
