"""Prune resume-only artifacts from completed E17 jobs.

A completed job needs only its derived evidence: per-checkpoint factorial rows
and metrics, the job summary, the B-matched selection, losses, and quality. The
optimizer state and the saved model weights exist purely to support exact resume
and re-evaluation, and together account for roughly 34 GB per job on a 1.48B
student. They are removed once the job is complete.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "runs" / "E17_CROSS_FAMILY" / "jobs"


def prune(job_dir: Path, *, force: bool = False) -> int:
    summary = job_dir / "job_summary.json"
    if not force:
        if not summary.exists():
            return 0
        if json.loads(summary.read_text(encoding="utf-8")).get("status") != "complete":
            return 0
    freed = 0
    for checkpoint in sorted((job_dir / "checkpoints").glob("step_*")):
        optimizer = checkpoint / "optimizer.pt"
        if optimizer.exists():
            freed += optimizer.stat().st_size
            optimizer.unlink()
        model = checkpoint / "model"
        if model.is_dir():
            freed += sum(f.stat().st_size for f in model.rglob("*") if f.is_file())
            shutil.rmtree(model)
    return freed


def main() -> int:
    total = 0
    for job_dir in sorted(JOBS.glob("*")):
        if not job_dir.is_dir():
            continue
        freed = prune(job_dir)
        if freed:
            print(f"pruned {job_dir.name}: {freed / 2**30:.1f} GiB")
        total += freed
    print(f"total freed: {total / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
