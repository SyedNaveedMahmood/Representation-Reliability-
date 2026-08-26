"""Core dataset helpers shared by generators and runners."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import Sample


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_completion_prompt(premise: str, question: str) -> tuple[str, str]:
    """Completion-style prompt without a chat template.

    Returns ``(prompt, target_span_text)`` where ``target_span_text`` is the
    exact question sentence (the span used by the ``target_span_last``
    token selector).
    """
    prompt = (
        "Premise: "
        + premise.strip()
        + "\nQuestion: "
        + question.strip()
        + "\nAnswer:"
    )
    return prompt, question.strip()


def find_subspan(prompt: str, subtext: str) -> tuple[int, int] | None:
    """Locate ``(char_start, char_end_exclusive)`` of ``subtext`` in ``prompt``."""
    idx = prompt.find(subtext)
    if idx < 0:
        return None
    return idx, idx + len(subtext)


def samples_to_dataframe(samples: Sequence[Sample]) -> pd.DataFrame:
    rows = []
    for s in samples:
        row: dict[str, Any] = {
            "sample_id": s.sample_id,
            "prompt": s.prompt,
            "target_label": s.target_label,
            "task_name": s.task_name,
        }
        for k, v in s.metadata.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def answer_to_label(answer: str | int) -> int:
    if isinstance(answer, int):
        return int(answer)
    lower = answer.strip().lower()
    if lower in ("yes", "true", "1", "correct"):
        return 1
    if lower in ("no", "false", "0", "incorrect"):
        return 0
    raise ValueError(f"cannot map answer {answer!r} to a binary label")


def check_label_balance(labels: Sequence[int]) -> dict[str, float]:
    arr = np.asarray(labels)
    n = len(arr)
    if n == 0:
        return {"n": 0, "frac_positive": float("nan")}
    return {
        "n": int(n),
        "n_positive": int((arr == 1).sum()),
        "n_negative": int((arr == 0).sum()),
        "frac_positive": float((arr == 1).mean()),
    }
