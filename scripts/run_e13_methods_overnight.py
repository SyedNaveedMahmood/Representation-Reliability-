"""GPU-aware scheduler for the authorized E13 method wave."""

from __future__ import annotations

import argparse
import datetime as dt
import queue
import sys
import threading
from pathlib import Path
from typing import Any

from run_e13_overnight import _atomic_json, _utcnow, detect_gpus, run_process

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261305, 20261315, 20261325)
REGIMES = ("R4", "R5", "R6", "R2C")
PROTOCOL_SHA256 = "3f3dd9a65347fc9ba6a20c29686aba11bd578f52b28d87818c399b422325846b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--parent-failed-campaign", default=None)
    parser.add_argument("--failure-reason", default=None)
    parser.add_argument("--fix-commit", default=None)
    args = parser.parse_args()
    gpus = detect_gpus()
    campaign_id = args.campaign_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_methods"
    )
    campaign_dir = ROOT / "runs" / "E13_OVERNIGHT" / campaign_id
    if campaign_dir.exists():
        raise FileExistsError(f"overnight campaign directory already exists: {campaign_dir}")
    campaign_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_id": campaign_id,
        "protocol_sha256": PROTOCOL_SHA256,
        "started_at": _utcnow(),
        "python": sys.executable,
        "gpus": gpus,
        "one_process_per_gpu": True,
        "retry_policy": "only demonstrated transient/OOM failures; none assumed",
        "parent_failed_campaign": args.parent_failed_campaign,
        "parent_failure_reason": args.failure_reason,
        "fix_commit": args.fix_commit,
        "confirmation_accessed": False,
    }
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    state: dict[str, Any] = {}
    lock = threading.Lock()
    errors: list[str] = []
    cache_command = [
        sys.executable,
        "-m",
        "representation_reliability.cli",
        "e13-method-cache",
    ]
    try:
        run_process("teacher_cache", cache_command, gpus[0], campaign_dir, state, lock)
    except RuntimeError as exc:
        errors.append(str(exc))
    pending: queue.Queue[tuple[str, list[str]]] = queue.Queue()
    if not errors:
        for regime in REGIMES:
            for seed in SEEDS:
                pending.put(
                    (
                        f"{regime}_seed_{seed}",
                        [
                            sys.executable,
                            "-m",
                            "representation_reliability.cli",
                            "e13-method-job",
                            "--regime",
                            regime,
                            "--seed",
                            str(seed),
                        ],
                    )
                )

    def worker(gpu: dict[str, Any]) -> None:
        while not errors:
            try:
                name, command = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_process(name, command, gpu, campaign_dir, state, lock)
            except RuntimeError as exc:
                errors.append(str(exc))
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not errors:
        analysis_command = [
            sys.executable,
            "-m",
            "representation_reliability.cli",
            "e13-method-analyze",
        ]
        try:
            run_process("method_analysis", analysis_command, gpus[0], campaign_dir, state, lock)
        except RuntimeError as exc:
            errors.append(str(exc))
    manifest["finished_at"] = _utcnow()
    manifest["status"] = "failed" if errors else "method_complete"
    manifest["errors"] = errors
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
