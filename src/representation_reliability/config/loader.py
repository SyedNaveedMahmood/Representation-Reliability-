"""Configuration loading, merging, hashing and saving.

Merge order (later wins)::

    configs/base.yaml  ->  configs/models/*.yaml  ->  configs/experiments/*.yaml
    -> CLI overrides (dot-paths, e.g. ``runtime.batch_size=4``)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .schema import AppConfig

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


class ConfigError(ValueError):
    """Raised when configuration loading or validation fails."""


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file does not exist: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"top-level YAML in {p} must be a mapping, got {type(data).__name__}"
        )
    return data


def deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_override(raw: str) -> tuple[str, Any]:
    """Parse a dot-path CLI override like ``runtime.batch_size=4``."""
    if "=" not in raw:
        raise ConfigError(f"override must look like 'a.b.c=value', got {raw!r}")
    path, _, raw_value = raw.partition("=")
    path = path.strip()
    if not path:
        raise ConfigError(f"override has empty dotted path: {raw!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse override value {raw_value!r}") from exc
    return path, value


def apply_override(doc: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _normalize_representation(section: dict) -> dict:
    """Accept either ``token_selector`` (model configs) or ``token_selectors``."""
    rep = dict(section)
    if "token_selectors" not in rep and "token_selector" in rep:
        sel = rep.pop("token_selector")
        rep["token_selectors"] = [sel] if isinstance(sel, str) else list(sel)
    elif "token_selector" in rep:
        rep.pop("token_selector")  # legacy singular key ignored when plural present
    return rep


def build_merged_doc(
    base_path: str | Path,
    model_path: str | Path | None,
    experiment_path: str | Path,
    overrides: Sequence[str] | Mapping[str, Any] = (),
) -> tuple[dict, dict[str, Any]]:
    """Return ``(merged_raw_doc, provenance)``."""
    provenance: dict[str, Any] = {"sources": {}, "cli_overrides": {}}
    doc: dict = {}

    doc = load_yaml(base_path)
    provenance["sources"]["base"] = str(Path(base_path))

    if model_path is not None:
        model_doc = load_yaml(model_path)
        if "representation" in model_doc:
            model_doc = dict(model_doc)
            model_doc["representation"] = _normalize_representation(
                model_doc["representation"]
            )
        doc = deep_merge(doc, model_doc)
        provenance["sources"]["model"] = str(Path(model_path))

    exp_doc = load_yaml(experiment_path)
    if "representation" in exp_doc:
        exp_doc = dict(exp_doc)
        exp_doc["representation"] = _normalize_representation(exp_doc["representation"])
    doc = deep_merge(doc, exp_doc)
    provenance["sources"]["experiment"] = str(Path(experiment_path))

    if isinstance(overrides, Mapping):
        for path, value in overrides.items():
            apply_override(doc, path, value)
            provenance["cli_overrides"][path] = value
    else:
        for raw in overrides:
            path, value = parse_override(raw)
            apply_override(doc, path, value)
            provenance["cli_overrides"][path] = value

    if "representation" in doc:
        doc["representation"] = _normalize_representation(doc["representation"])
    return doc, provenance


def resolve_config(
    base_path: str | Path | None = None,
    model_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    overrides: Sequence[str] | Mapping[str, Any] = (),
) -> tuple[AppConfig, dict]:
    """Load, merge and validate the full configuration.

    Returns ``(AppConfig, provenance)``. Raises :class:`ConfigError` on any
    invalid or unknown configuration content.
    """
    base_path = base_path or CONFIG_ROOT / "base.yaml"
    experiment_path = (
        experiment_path or CONFIG_ROOT / "experiments" / "E00_cartography.yaml"
    )
    if model_path is None:
        model_path = CONFIG_ROOT / "models" / "qwen3_0.6b.yaml"

    doc, provenance = build_merged_doc(
        base_path, model_path, experiment_path, overrides
    )

    missing = [
        k
        for k in (
            "project", "runtime", "budget", "storage", "statistics",
            "model", "representation", "dataset", "experiment", "probe",
        )
        if k not in doc
    ]
    if missing:
        raise ConfigError(f"missing required config section(s): {missing}")

    try:
        cfg = AppConfig.model_validate(doc)
    except Exception as exc:
        raise ConfigError(f"configuration validation failed: {exc}") from exc
    return cfg, provenance


def config_hash(cfg: AppConfig) -> str:
    """Deterministic SHA-256 of the fully resolved scientific configuration.

    Provenance (file paths, CLI override spellings) is deliberately excluded:
    two runs reaching the identical resolved configuration hash identically.
    """
    payload = json.dumps(
        cfg.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_resolved_config(
    cfg: AppConfig, path: str | Path, provenance: dict | None = None
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = cfg.to_dict()
    if provenance is not None:
        doc["_provenance"] = provenance
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=True, default_flow_style=False)

