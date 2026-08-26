"""Artifact table writers (parquet/JSON/markdown) shared by runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def save_json(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    if df.empty:
        return "(no rows)"
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return float_fmt.format(v) if v == v else "nan"
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)
