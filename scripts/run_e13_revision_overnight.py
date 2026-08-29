"""GPU-aware staged scheduler for the frozen E13 method-revision campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any

from run_e13_overnight import _atomic_json, _utcnow, detect_gpus, run_process

ROOT = Path(__file__).resolve().parents[1]
BOUNDED_SEED = 20261305
FULL_SEEDS = (20261305, 20261315, 20261325)
REGIMES = tuple(f"R{index}" for index in range(7, 17))
REVISION_CAMPAIGN = "E13MR_08b45d7912c2_79d9141a25b4"
SPECIFICITY_SHA256 = "08b45d7912c2110b87fb2915b983dcc9e4d2b66a0c961efa2bb8c46438d078ba"
OBJECTIVE_SHA256 = "79d9141a25b420ed6b8fcd290297384cb9d31e920d53dec634bb0462b6a180b0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default=None)
    args = parser.parse_args()
    gpus = detect_gpus()
    campaign_id = args.campaign_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_e13_revision"
    )
    campaign_dir = ROOT / "runs" / "E13_METHOD_REVISION" / campaign_id
    if campaign_dir.exists():
        raise FileExistsError(f"revision scheduler campaign already exists: {campaign_dir}")
    campaign_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_id": campaign_id,
        "scientific_campaign": REVISION_CAMPAIGN,
        "specificity_protocol_sha256": SPECIFICITY_SHA256,
        "objective_protocol_sha256": OBJECTIVE_SHA256,
        "started_at": _utcnow(),
        "python": sys.executable,
        "gpus": gpus,
        "one_process_per_gpu": True,
        "confirmation_accessed": False,
    }
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    state: dict[str, Any] = {}
    lock = threading.Lock()
    errors: list[str] = []

    def run_jobs(jobs: list[tuple[str, list[str]]]) -> None:
        pending: queue.Queue[tuple[str, list[str]]] = queue.Queue()
        for job in jobs:
            pending.put(job)

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

    bounded_jobs = [
        (
            f"{regime}_seed_{BOUNDED_SEED}",
            [
                sys.executable,
                "-m",
                "representation_reliability.cli",
                "e13-revision-job",
                "--regime",
                regime,
                "--seed",
                str(BOUNDED_SEED),
            ],
        )
        for regime in REGIMES
    ]
    run_jobs(bounded_jobs)
    if not errors:
        try:
            run_process(
                "bounded_analysis",
                [sys.executable, "-m", "representation_reliability.cli", "e13-revision-analyze"],
                gpus[0],
                campaign_dir,
                state,
                lock,
            )
        except RuntimeError as exc:
            errors.append(str(exc))
    selection_path = ROOT / "runs" / "E13_METHOD_REVISION" / REVISION_CAMPAIGN / "selection.json"
    selected = None
    if not errors:
        selected = json.loads(selection_path.read_text(encoding="utf-8")).get("selected_regime")
    if selected:
        full_jobs = [
            (
                f"{selected}_seed_{seed}",
                [
                    sys.executable,
                    "-m",
                    "representation_reliability.cli",
                    "e13-revision-job",
                    "--regime",
                    selected,
                    "--seed",
                    str(seed),
                ],
            )
            for seed in FULL_SEEDS
            if seed != BOUNDED_SEED
        ]
        run_jobs(full_jobs)
        if not errors:
            try:
                run_process(
                    "final_analysis",
                    [
                        sys.executable,
                        "-m",
                        "representation_reliability.cli",
                        "e13-revision-analyze",
                    ],
                    gpus[0],
                    campaign_dir,
                    state,
                    lock,
                )
            except RuntimeError as exc:
                errors.append(str(exc))
    manifest.update(
        {
            "finished_at": _utcnow(),
            "selected_regime": selected,
            "status": "failed" if errors else "complete",
            "errors": errors,
        }
    )
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
