"""Stable local residual-stream intervention wrapper for HFAdapter.

Scientific runners call this module rather than registering model-specific hooks
or traversing Hugging Face module paths themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .hf import HFAdapter


def forward_resid_post_intervention(
    adapter: "HFAdapter",
    prompts: Sequence[str],
    *,
    layer: int,
    token_indices: Sequence[int],
    deltas: np.ndarray,
    output_token_ids: Sequence[int],
    capture_layers: Sequence[int] = (),
) -> dict[str, Any]:
    """Run one forward pass with per-example additive ``resid_post`` edits.

    Parameters
    ----------
    adapter:
        Loaded :class:`HFAdapter`.
    prompts:
        Raw prompts. Right-padding follows the adapter's tokenizer policy.
    layer:
        0-indexed decoder layer whose *output residual* is edited.
    token_indices:
        One resolved token position per prompt.
    deltas:
        ``[batch, hidden]`` additive edits in raw residual coordinates.
    output_token_ids:
        Vocabulary token IDs whose logits are returned at each edited token
        position. For E01A these are the first Yes/No continuation tokens.
    capture_layers:
        Residual-post layers at or after ``layer`` to capture after the edit
        has propagated. The intervention layer itself records the edited state.

    Returns
    -------
    dict
        ``selected_logits`` is ``[batch, len(output_token_ids)]``;
        ``captured`` maps layer -> ``[batch, hidden]`` at the same token site.

    Notes
    -----
    This is intentionally additive-only. Replacement is represented as
    ``delta = source - base``. The function clones the hooked residual before
    editing and removes every hook in ``finally``.
    """
    if adapter.model is None or adapter.tokenizer is None:
        raise RuntimeError("adapter.load() must be called before intervention")
    n = len(prompts)
    if n == 0:
        raise ValueError("intervention batch may not be empty")
    if len(token_indices) != n:
        raise ValueError("token_indices/prompts length mismatch")
    delta_np = np.asarray(deltas)
    if delta_np.shape != (n, adapter.hidden_size):
        raise ValueError(
            f"deltas must have shape {(n, adapter.hidden_size)}, got {delta_np.shape}"
        )
    if not (0 <= int(layer) < adapter.num_layers):
        raise ValueError(f"intervention layer {layer} out of range")
    capture = sorted({int(x) for x in capture_layers})
    bad = [x for x in capture if x < int(layer) or x >= adapter.num_layers]
    if bad:
        raise ValueError(
            "capture_layers must be at/after the intervention and in range: "
            f"{bad}"
        )
    if not output_token_ids:
        raise ValueError("output_token_ids may not be empty")

    batch = adapter.tokenize(prompts)
    input_ids = batch["input_ids"].to(adapter.device)
    attention_mask = batch["attention_mask"].to(adapter.device)
    idx = torch.as_tensor(
        list(map(int, token_indices)), dtype=torch.long, device=adapter.device
    )
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    if torch.any(idx < 0) or torch.any(idx >= lengths):
        raise ValueError("one or more intervention token indices are outside prompt length")

    delta_t = torch.as_tensor(
        delta_np,
        device=adapter.device,
        dtype=next(adapter.model.parameters()).dtype,
    )
    row_ids = torch.arange(n, device=adapter.device)
    captured: dict[int, torch.Tensor] = {}
    hooks: list[Any] = []

    def _replace_output(output: Any, new_hidden: torch.Tensor) -> Any:
        if isinstance(output, tuple):
            return (new_hidden, *output[1:])
        return new_hidden

    def intervention_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[row_ids, idx] = edited[row_ids, idx] + delta_t
        if int(layer) in capture:
            captured[int(layer)] = edited[row_ids, idx].detach()
        return _replace_output(output, edited)

    def capture_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_id] = hidden[row_ids, idx].detach()
        return hook

    hooks.append(adapter._raw_layer(int(layer)).register_forward_hook(intervention_hook))
    for layer_id in capture:
        if layer_id == int(layer):
            continue
        hooks.append(
            adapter._raw_layer(layer_id).register_forward_hook(capture_hook(layer_id))
        )

    try:
        with torch.inference_mode():
            outputs = adapter.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
    finally:
        for handle in hooks:
            handle.remove()

    logits = outputs.logits[row_ids, idx]
    token_ids = torch.as_tensor(
        list(map(int, output_token_ids)), dtype=torch.long, device=adapter.device
    )
    selected = logits.index_select(-1, token_ids).detach().float().cpu().numpy()

    missing = [layer_id for layer_id in capture if layer_id not in captured]
    if missing:
        raise RuntimeError(f"failed to capture downstream residual layers: {missing}")

    return {
        "selected_logits": selected,
        "captured": {
            layer_id: tensor.float().cpu().numpy()
            for layer_id, tensor in captured.items()
        },
        "token_indices": list(map(int, token_indices)),
        "output_token_ids": list(map(int, output_token_ids)),
    }
