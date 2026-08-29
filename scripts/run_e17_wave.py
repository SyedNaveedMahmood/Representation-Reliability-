"""Sequential GPU scheduler for the frozen E17 cross-family wave."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prune_e17_checkpoints import prune

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261705, 20261715, 20261725)
REGIMES = ("R1", "R2", "R3")


def main() -> int:
    campaign = ROOT / "runs" / "E17_CROSS_FAMILY"
    campaign.mkdir(parents=True, exist_ok=True)
    status_path = campaign / "wave_status.json"
    state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    for regime in REGIMES:
        for seed in SEEDS:
            key = f"{regime}_seed_{seed}"
            if state.get(key, {}).get("state") == "complete":
                continue
            command = [
                sys.executable, "-m", "representation_reliability.cli",
                "e17-job", "--regime", regime, "--seed", str(seed),
            ]
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            log = campaign / "logs"
            log.mkdir(parents=True, exist_ok=True)
            with (log / f"{key}.log").open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command, cwd=ROOT, stdout=handle, stderr=handle, check=False
                )
            state[key] = {
                "state": "complete" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "started_at": started,
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            status_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"E17 job {key} failed with exit {result.returncode}")
            # A 1.48B student writes roughly 34 GiB of resume-only artifacts per
            # job. Prune each completed job immediately so the wave's disk
            # footprint stays bounded instead of accumulating past the device.
            for job_dir in sorted((campaign / "jobs").glob(f"{key}_*")):
                freed = prune(job_dir)
                if freed:
                    print(f"pruned {job_dir.name}: {freed / 2**30:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
