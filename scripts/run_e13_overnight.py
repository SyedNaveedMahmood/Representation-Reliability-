"""GPU-aware E13 overnight scheduler with immutable logs and job status."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261305, 20261315, 20261325)
REGIMES = ("R1", "R2", "R3")


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def detect_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    output = []
    for line in completed.stdout.splitlines():
        index, name, total, used, utilization = [part.strip() for part in line.split(",")]
        output.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib_at_start": int(used),
                "utilization_percent_at_start": int(utilization),
            }
        )
    if not output:
        raise RuntimeError("no NVIDIA GPU detected")
    return output


def _gpu_memory_used(index: int) -> int | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def run_process(
    name: str,
    command: list[str],
    gpu: dict[str, Any],
    campaign_dir: Path,
    state: dict[str, Any],
    lock: threading.Lock,
) -> None:
    logs = campaign_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    started = _utcnow()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        with lock:
            state[name] = {
                "state": "running",
                "pid": process.pid,
                "gpu_index": gpu["index"],
                "command": command,
                "started_at": started,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "peak_observed_gpu_memory_mib": gpu["memory_used_mib_at_start"],
                "attempt": 1,
            }
            _atomic_json(campaign_dir / "job_status.json", state)
        while process.poll() is None:
            used = _gpu_memory_used(gpu["index"])
            if used is not None:
                with lock:
                    state[name]["peak_observed_gpu_memory_mib"] = max(
                        int(state[name]["peak_observed_gpu_memory_mib"]), used
                    )
            time.sleep(5)
        exit_code = int(process.returncode)
    with lock:
        state[name].update(
            {
                "state": "complete" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "finished_at": _utcnow(),
            }
        )
        _atomic_json(campaign_dir / "job_status.json", state)
    if exit_code != 0:
        raise RuntimeError(f"E13 job {name} failed with exit code {exit_code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default=None)
    args = parser.parse_args()
    gpus = detect_gpus()
    campaign_id = args.campaign_id or dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%S"
    )
    campaign_dir = ROOT / "runs" / "E13_OVERNIGHT" / campaign_id
    if campaign_dir.exists():
        raise FileExistsError(f"overnight campaign directory already exists: {campaign_dir}")
    campaign_dir.mkdir(parents=True)
    manifest = {
        "campaign_id": campaign_id,
        "started_at": _utcnow(),
        "python": sys.executable,
        "gpus": gpus,
        "one_process_per_gpu": True,
        "confirmation_accessed": False,
    }
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    state: dict[str, Any] = {}
    lock = threading.Lock()
    reference_command = [
        sys.executable,
        "-m",
        "representation_reliability.cli",
        "e13-multiseed-reference",
    ]
    run_process("reference", reference_command, gpus[0], campaign_dir, state, lock)
    pending: queue.Queue[tuple[str, list[str]]] = queue.Queue()
    for regime in REGIMES:
        for seed in SEEDS:
            pending.put(
                (
                    f"{regime}_seed_{seed}",
                    [
                        sys.executable,
                        "-m",
                        "representation_reliability.cli",
                        "e13-multiseed-job",
                        "--regime",
                        regime,
                        "--seed",
                        str(seed),
                    ],
                )
            )
    errors: list[str] = []

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

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not errors:
        analysis_command = [
            sys.executable,
            "-m",
            "representation_reliability.cli",
            "e13-multiseed-analyze",
        ]
        try:
            run_process("baseline_analysis", analysis_command, gpus[0], campaign_dir, state, lock)
        except RuntimeError as exc:
            errors.append(str(exc))
    manifest["finished_at"] = _utcnow()
    manifest["status"] = "failed" if errors else "baseline_complete"
    manifest["errors"] = errors
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
