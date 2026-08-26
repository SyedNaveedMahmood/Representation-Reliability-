import pytest

from representation_reliability.extraction.token_selection import (
    resolve_token_selection,
)


class _DummyTokenizer:
    """Whitespace tokenizer with offsets, mimicking the HF interface subset."""

    def __call__(self, text: str, add_special_tokens=True, return_offsets_mapping=False):
        tokens = text.split(" ")
        input_ids = [abs(hash(t)) % 50_000 + 1 for t in tokens]
        offsets = []
        pos = 0
        for t in tokens:
            start = text.index(t, pos)
            offsets.append((start, start + len(t)))
            pos = start + len(t)
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def decode(self, token_ids, skip_special_tokens=False):
        return "<tok>"


TOK = _DummyTokenizer()
PROMPT = "Premise: Luma is north of Reko.\nQuestion: Is Reko south of Luma?\nAnswer:"
TARGET = "Is Reko south of Luma?"


def test_last_prompt_selects_final_token():
    r = resolve_token_selection(
        "last_prompt", TOK, PROMPT, chat_template_used=False
    )
    assert r.strategy == "last_prompt"
    assert r.index == r.sequence_length - 1
    assert r.index >= 0
    assert r.sequence_length > 0
    assert r.char_start is not None and r.char_end > r.char_start
    assert not r.chat_template_used


def test_target_span_last_lands_inside_span():
    enc = TOK(PROMPT)
    offsets = enc["offset_mapping"]
    span_start = PROMPT.index(TARGET)
    span_end = span_start + len(TARGET)
    r = resolve_token_selection(
        "target_span_last", TOK, PROMPT,
        target_text=TARGET,
    )
    s, e = offsets[r.index]
    # Selected token must START inside the span; its end may exceed into
    # trailing characters (mirrors real BPE merge behavior).
    assert s >= span_start
    assert s < span_end or e <= span_end


def test_explicit_selector_and_bounds():
    r = resolve_token_selection("explicit", TOK, PROMPT, explicit_index=4)
    assert r.index == 4
    with pytest.raises(ValueError):
        resolve_token_selection("explicit", TOK, PROMPT, explicit_index=-2)
    with pytest.raises(ValueError):
        resolve_token_selection("explicit", TOK, PROMPT, explicit_index=10_000)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        resolve_token_selection("middle_somewhere", TOK, PROMPT)


def test_every_record_has_full_provenance_fields():
    for strategy in ("last_prompt", "target_span_last"):
        r = resolve_token_selection(
            strategy, TOK, PROMPT,
            target_text=TARGET if strategy == "target_span_last" else None,
            chat_template_used=False,
        )
        d = r.as_dict()
        for field in ("selector_strategy", "token_index", "token_id",
                      "token_text", "sequence_length", "chat_template_used"):
            assert field in d, f"missing {field}"
        assert d["token_index"] != -1
