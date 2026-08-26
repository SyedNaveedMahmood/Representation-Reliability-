# GPU / Runtime Plan

Goal: each experiment normally <=12 GPU-hours on a 16GB-class GPU.

## Model ladder
Qwen3-0.6B:
- smoke;
- trajectory;
- dense scans.

Qwen3-1.7B:
- primary D/C/S;
- component work;
- monitor pilots.

3B:
- replication/cross-model.

7B/8B:
- only after held-out + transformation gates.

## VRAM rules
- bf16 where supported;
- inference mode for extraction;
- batch by token count;
- log peak VRAM;
- store selected sites/tokens, not everything by default.

## Run tiers
Smoke:
- <=100 examples;
- one seed;
- 3–5 layers.

Pilot:
- 500–2,000;
- coarse all-layer scan;
- one/two seeds.

Discovery:
- 2k–10k;
- refined sites;
- full controls.

Confirmation:
- frozen hypothesis;
- held-out split;
- second task/model;
- no post-hoc site search.

## Parallelize by
- seed;
- layer blocks;
- alpha;
- transforms;
- model.

Never allow workers to write the same shard.

## Budget handling
Config:
```yaml
budget:
  max_gpu_hours: 12
  max_examples: 5000
```

If budget reached:
- finish current shard;
- mark `partial_budget_stop`;
- keep evidence.
