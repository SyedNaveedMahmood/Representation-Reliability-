"""Frozen external general-quality controls for E14."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
HELLASWAG_SEED = 20261402


def block_scored_token_count(token_count: int, block_size: int) -> int:
    if int(token_count) < 0 or int(block_size) < 2:
        raise ValueError("token count must be nonnegative and block size at least two")
    return sum(
        max(0, min(int(block_size), int(token_count) - start) - 1)
        for start in range(0, int(token_count), int(block_size))
    )


def deterministic_hellaswag_indices(rows: list[dict[str, Any]], n: int = 500) -> list[int]:
    if len(rows) < int(n):
        raise ValueError("HellaSwag split is smaller than the frozen subset")
    ranked = sorted(
        range(len(rows)),
        key=lambda index: hashlib.sha256(
            f"{HELLASWAG_SEED}|{rows[index]['ind']}".encode()
        ).hexdigest(),
    )
    return ranked[: int(n)]


def score_wikitext(
    adapter, *, token_budget: int = 10_000, block_size: int = 512
) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=WIKITEXT_REVISION,
    )
    text = "\n".join(str(value) for value in dataset["text"] if str(value).strip())
    token_ids = adapter.tokenizer(text, add_special_tokens=False)["input_ids"][:token_budget]
    if len(token_ids) < token_budget:
        raise RuntimeError("WikiText token stream is shorter than frozen budget")
    total_loss = 0.0
    scored = 0
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        if len(block) < 2:
            continue
        ids = torch.as_tensor([block], dtype=torch.long, device=adapter.device)
        with torch.inference_mode():
            logits = adapter.model(input_ids=ids).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="sum"
            )
        total_loss += float(loss.detach().cpu())
        scored += len(block) - 1
    nll = total_loss / scored
    if scored != block_scored_token_count(len(token_ids), block_size):
        raise RuntimeError("WikiText scored-token accounting mismatch")
    return {
        "dataset": "Salesforce/wikitext",
        "configuration": "wikitext-2-raw-v1",
        "split": "test",
        "revision": WIKITEXT_REVISION,
        "token_budget": token_budget,
        "scored_tokens": scored,
        "token_weighted_nll": nll,
        "perplexity": float(math.exp(min(nll, 50.0))),
    }


def score_hellaswag(
    adapter, *, n_examples: int = 500, batch_size: int = 16
) -> tuple[dict[str, Any], pd.DataFrame]:
    from datasets import load_dataset

    dataset = load_dataset(
        "Rowan/hellaswag", split="validation", revision=HELLASWAG_REVISION
    )
    all_rows = [dict(dataset[index]) for index in range(len(dataset))]
    selected = [all_rows[index] for index in deterministic_hellaswag_indices(all_rows, n_examples)]
    scores = adapter.score_continuations(
        [str(row["ctx"]) for row in selected],
        [list(map(str, row["endings"])) for row in selected],
        batch_size=batch_size,
    )
    output: list[dict[str, Any]] = []
    for row, choices in zip(selected, scores):
        normalized = np.asarray([choice["logp_mean"] for choice in choices], dtype=float)
        predicted = int(np.argmax(normalized))
        gold = int(row["label"])
        output.append(
            {
                "ind": int(row["ind"]),
                "gold": gold,
                "prediction": predicted,
                "correct": int(predicted == gold),
                "gold_logp_mean": float(normalized[gold]),
                "choice_logp_means": normalized.tolist(),
            }
        )
    frame = pd.DataFrame(output)
    ids = frame["ind"].astype(str).tolist()
    return (
        {
            "dataset": "Rowan/hellaswag",
            "configuration": "default",
            "split": "validation",
            "revision": HELLASWAG_REVISION,
            "subset_seed": HELLASWAG_SEED,
            "n_examples": len(frame),
            "subset_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "accuracy": float(frame["correct"].mean()),
            "mean_gold_logp_normalized": float(frame["gold_logp_mean"].mean()),
        },
        frame,
    )
