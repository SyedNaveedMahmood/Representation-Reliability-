"""Full run manifest: environment, git state, versions, GPU, seeds, timings.

Fields that cannot be obtained are stored as ``null`` with an explanatory
note rather than silently omitted.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .status import atomic_write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_REPOS_DIR = REPO_ROOT / "external" / "repos"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def project_git_state() -> dict[str, Any]:
    sha = _git(["rev-parse", "HEAD"], REPO_ROOT)
    if sha is None:
        return {"sha": None, "dirty": None,
                "note": "git not available or repository root not found"}
    status = _git(["status", "--porcelain"], REPO_ROOT)
    dirty = None
    if status is not None:
        dirty_lines = [ln for ln in status.splitlines() if not ln.startswith("??")]
        dirty = bool(dirty_lines)
    return {"sha": sha, "dirty": dirty, "porcelain": status}


def external_repo_shas() -> dict[str, Any]:
    if not EXTERNAL_REPOS_DIR.exists():
        return {"present": False, "repos": {}}
    repos = {}
    for child in sorted(EXTERNAL_REPOS_DIR.iterdir()):
        if (child / ".git").exists():
            sha = _git(["rev-parse", "HEAD"], child)
            entry: dict[str, Any] = {"sha": sha}
            if sha is None:
                entry["note"] = "could not resolve HEAD"
            repos[child.name] = entry
    return {"present": True, "repos": repos}


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": None, "device_name": None,
        "total_vram_bytes": None, "cuda_version": None, "device_count": None,
    }
    try:
        import torch
    except Exception as exc:
        info["note"] = f"torch unavailable: {exc}"
        return info
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["total_vram_bytes"] = int(getattr(props, "total_memory", 0)) or None
    return info


def environment_manifest() -> dict[str, Any]:
    g = gpu_info()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "project_git": project_git_state(),
        "external_repos": external_repo_shas(),
        "package_versions": {
            "torch": g.get("torch_version") or package_version("torch"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "safetensors": package_version("safetensors"),
            "nnsight": package_version("nnsight"),
            "pyvene": package_version("pyvene"),
            "scikit-learn": package_version("scikit-learn"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
        },
        "gpu": g,
    }


class RunManifest:
    """Builds and writes manifest.json for one run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "manifest.json"
        self.manifest: dict[str, Any] = {
            "start_time": None,
            "finish_time": None,
            "wall_time_s": None,
            "model": {
                "id": None, "revision": None, "dtype": None,
                "tokenizer_id": None, "tokenizer_revision": None,
                "num_layers": None, "hidden_size": None,
                "resolved_native_modules": {},
                "notes": {},
            },
            "dataset": {
                "split_hash": None, "prompt_hash_sample": None, "seeds": {},
            },
            "config_hash": None,
            "provenance": {},
            "peak_vram_allocated_bytes": None,
            "peak_vram_reserved_bytes": None,
            "runs_summary": [],
        }

    def set_start(self, config_hash_value: str | None, provenance: dict | None,
                  seeds: dict[str, Any]) -> None:
        self.manifest["start_time"] = _utcnow()
        self.manifest["environment"] = environment_manifest()
        self.manifest["config_hash"] = config_hash_value
        self.manifest["provenance"] = provenance or {}
        self.manifest["dataset"]["seeds"] = seeds
        self.write()

    def update_model_info(self, **kwargs: Any) -> None:
        model_sec = self.manifest["model"]
        for key, value in kwargs.items():
            if key == "notes":
                model_sec.setdefault("notes", {}).update(value)
            elif isinstance(value, dict):
                model_sec.setdefault(key, {})
                model_sec[key].update(value)
            else:
                model_sec[key] = value
        self.write()

    def update_dataset_info(self, **kwargs: Any) -> None:
        self.manifest["dataset"].update(kwargs)
        self.write()

    def capture_vram_peaks(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                self.manifest["peak_vram_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
                self.manifest["peak_vram_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved()
                )
        except Exception as exc:
            self.manifest.setdefault("notes", {})[
                "vram_capture"
            ] = f"failed: {exc}"
        self.write()

    def finish(self, runs_summary: list[dict[str, Any]] | None = None) -> None:
        start = self.manifest.get("start_time")
        now = _utcnow()
        self.manifest["finish_time"] = now
        wall = None
        if start:
            try:
                t0 = _dt.datetime.fromisoformat(start)
                t1 = _dt.datetime.fromisoformat(now)
                wall = (t1 - t0).total_seconds()
            except ValueError:
                wall = None
        self.manifest["wall_time_s"] = wall
        if runs_summary is not None:
            self.manifest["runs_summary"] = runs_summary
        self.capture_vram_peaks()
        self.write()

    def write(self) -> None:
        atomic_write_json(self.path, self.manifest)


def dataset_split_hash(split_assignment: dict[str, str]) -> str:
    payload = json.dumps(split_assignment, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
