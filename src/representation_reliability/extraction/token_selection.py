"""Explicit token selectors for activation sites.

A selector resolves to a concrete record containing the strategy, the
resolved integer token index, the token id, the decoded string, the
character span, the sequence length, and whether a chat template was used.
``-1`` is never an acceptable resolved site description in stored evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

SELECTOR_NAMES = ("last_prompt", "target_span_last", "explicit")


@dataclass(frozen=True)
class ResolvedToken:
    """Everything that must be known about *where* an activation was taken."""

    strategy: str
    index: int                     # resolved non-negative token index
    token_id: int
    token_text: str
    char_start: int | None
    char_end: int | None
    sequence_length: int
    chat_template_used: bool
    selector_params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector_strategy": self.strategy,
            "token_index": self.index,
            "token_id": self.token_id,
            "token_text": self.token_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "sequence_length": self.sequence_length,
            "chat_template_used": self.chat_template_used,
        }


class TokenizerLike(Protocol):
    """Minimal tokenizer interface used by selectors (HF-compatible)."""

    def __call__(self, text: str, **kwargs: Any):  # returns BatchEncoding-like
        ...


def _encode(encoding_target: str, tokenizer: Any, text: str):
    return tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )


def resolve_token_selection(
    strategy: str,
    tokenizer: Any,
    prompt_text: str,
    *,
    chat_template_used: bool = False,
    target_text: str | None = None,
    explicit_index: int | None = None,
) -> ResolvedToken:
    """Resolve ``strategy`` against a concrete prompt.

    Strategies:
      - ``last_prompt``         : final prompt token (the model's "reading" position).
      - ``target_span_last``    : last token inside ``target_text`` located in the prompt.
      - ``explicit``            : user-provided ``explicit_index`` (>= 0).
    """
    if strategy not in SELECTOR_NAMES:
        raise ValueError(f"unknown token selector {strategy!r}")

    enc = tokenizer(prompt_text, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = list(enc["input_ids"])
    offsets = enc.get("offset_mapping") if hasattr(enc, "get") else None
    seq_len = len(input_ids)

    if strategy == "last_prompt":
        idx = seq_len - 1
        params: dict[str, Any] = {}
    elif strategy == "target_span_last":
        if not target_text:
            raise ValueError(
                "target_span_last requires `target_text` on the sample"
            )
        char_pos = prompt_text.find(target_text)
        if char_pos < 0:
            raise ValueError(
                f"target_text {target_text!r} not found in prompt; "
                "cannot resolve target span"
            )
        span_start = char_pos
        span_end = char_pos + len(target_text)
        params = {"target_char_start": span_start, "target_char_end": span_end}
        if not offsets:
            raise ValueError("tokenizer did not provide offset mappings")
        # Select the LAST token whose *start* lies inside the requested span.
        # (BPE commonly merges the final character with trailing whitespace,
        # e.g. '?\n'; requiring end<=span_end would wrongly skip it.)
        idx = -1
        for i, (s, e) in enumerate(offsets):
            if s >= span_start and s < span_end and e > s:
                idx = i
        if idx < 0:
            raise ValueError(
                "no tokens overlap the requested target span (empty selection)"
            )
        tok_s, tok_e = int(offsets[idx][0]), int(offsets[idx][1])
        decoded = _decode_single(tokenizer, input_ids, idx)
        return ResolvedToken(
            strategy=strategy,
            index=idx,
            token_id=int(input_ids[idx]),
            token_text=decoded,
            char_start=tok_s,
            char_end=tok_e,
            sequence_length=seq_len,
            chat_template_used=chat_template_used,
            selector_params=params,
        )
    else:  # explicit
        if explicit_index is None or explicit_index < 0:
            raise ValueError(
                "explicit selector requires a non-negative explicit_index"
            )
        if explicit_index >= seq_len:
            raise ValueError(
                f"explicit_index {explicit_index} out of range (seq_len={seq_len})"
            )
        idx = explicit_index
        params = {"requested_index": explicit_index}

    decoded = _decode_single(tokenizer, input_ids, idx)
    tok_s = tok_e = None
    if offsets:
        s, e = offsets[idx]
        tok_s, tok_e = int(s), int(e)
    return ResolvedToken(
        strategy=strategy,
        index=idx,
        token_id=int(input_ids[idx]),
        token_text=decoded,
        char_start=tok_s,
        char_end=tok_e,
        sequence_length=seq_len,
        chat_template_used=chat_template_used,
        selector_params=params,
    )


def _decode_single(tokenizer: Any, input_ids: list[int], idx: int) -> str:
    token_id = int(input_ids[idx])
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:  # pragma: no cover - defensive
        return "<decode-failed>"
