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


def forward_selected_token_logits(
    adapter: HFAdapter,
    prompts: Sequence[str],
    *,
    token_indices: Sequence[int],
    output_token_ids: Sequence[int],
) -> np.ndarray:
    """Run an unhooked forward and return selected next-token logits."""
    if adapter.model is None or adapter.tokenizer is None:
        raise RuntimeError("adapter.load() must be called before forward")
    n = len(prompts)
    if n == 0 or len(token_indices) != n:
        raise ValueError("prompts must be non-empty with one token index per row")
    if not output_token_ids:
        raise ValueError("output_token_ids may not be empty")
    batch = adapter.tokenize(prompts)
    input_ids = batch["input_ids"].to(adapter.device)
    attention_mask = batch["attention_mask"].to(adapter.device)
    idx = torch.as_tensor(list(map(int, token_indices)), dtype=torch.long, device=adapter.device)
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    if torch.any(idx < 0) or torch.any(idx >= lengths):
        raise ValueError("one or more token indices are outside prompt length")
    with torch.inference_mode():
        outputs = adapter.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    row_ids = torch.arange(n, device=adapter.device)
    logits = outputs.logits[row_ids, idx]
    token_ids = torch.as_tensor(
        list(map(int, output_token_ids)), dtype=torch.long, device=adapter.device
    )
    return logits.index_select(-1, token_ids).detach().float().cpu().numpy()


def forward_resid_post_intervention(
    adapter: HFAdapter,
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
        raise ValueError(f"deltas must have shape {(n, adapter.hidden_size)}, got {delta_np.shape}")
    if not (0 <= int(layer) < adapter.num_layers):
        raise ValueError(f"intervention layer {layer} out of range")
    capture = sorted({int(x) for x in capture_layers})
    bad = [x for x in capture if x < int(layer) or x >= adapter.num_layers]
    if bad:
        raise ValueError(f"capture_layers must be at/after the intervention and in range: {bad}")
    if not output_token_ids:
        raise ValueError("output_token_ids may not be empty")

    batch = adapter.tokenize(prompts)
    input_ids = batch["input_ids"].to(adapter.device)
    attention_mask = batch["attention_mask"].to(adapter.device)
    idx = torch.as_tensor(list(map(int, token_indices)), dtype=torch.long, device=adapter.device)
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
        hooks.append(adapter._raw_layer(layer_id).register_forward_hook(capture_hook(layer_id)))

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
            layer_id: tensor.float().cpu().numpy() for layer_id, tensor in captured.items()
        },
        "token_indices": list(map(int, token_indices)),
        "output_token_ids": list(map(int, output_token_ids)),
    }


def differentiable_resid_post_logits(
    adapter: HFAdapter,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    token_indices: torch.Tensor,
    deltas: torch.Tensor,
    output_token_ids: Sequence[int],
) -> torch.Tensor:
    """Differentiable selected logits under a batched additive residual edit."""
    if adapter.model is None:
        raise RuntimeError("adapter.load() must be called before intervention")
    batch_size = int(input_ids.shape[0])
    if deltas.shape != (batch_size, adapter.hidden_size):
        raise ValueError("differentiable intervention delta shape mismatch")
    if token_indices.shape != (batch_size,):
        raise ValueError("differentiable intervention token-index shape mismatch")
    row_ids = torch.arange(batch_size, device=input_ids.device)
    delta = deltas.to(device=input_ids.device, dtype=next(adapter.model.parameters()).dtype)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[row_ids, token_indices] = edited[row_ids, token_indices] + delta
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = adapter._raw_layer(int(layer)).register_forward_hook(hook)
    try:
        outputs = adapter.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    logits = outputs.logits[row_ids, token_indices]
    ids = torch.as_tensor(output_token_ids, dtype=torch.long, device=logits.device)
    return logits.index_select(-1, ids)


def forward_with_resid_post_capture(
    adapter: HFAdapter,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
) -> tuple[Any, torch.Tensor]:
    """Differentiable model forward with one canonical residual site captured."""
    if adapter.model is None:
        raise RuntimeError("adapter.load() must be called before capture")
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = adapter._raw_layer(int(layer)).register_forward_hook(hook)
    try:
        outputs = adapter.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one residual capture, observed {len(captured)}")
    return outputs, captured[0]


def resid_post_hook_count(adapter: HFAdapter, *, layer: int) -> int:
    """Diagnostic count for hook-leak contract tests at a canonical site."""
    if adapter.model is None:
        raise RuntimeError("adapter.load() must be called before hook inspection")
    return len(adapter._raw_layer(int(layer))._forward_hooks)


def forward_resid_post_edit(
    adapter: HFAdapter,
    prompts: Sequence[str],
    *,
    edit_layer: int,
    edit_token_indices: Sequence[int],
    deltas: np.ndarray,
    readout_token_indices: Sequence[int],
    output_token_ids: Sequence[int],
    capture_layers: Sequence[int] = (),
    capture_token_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Additive resid_post edit at one token, readout at a *different* token.

    ``forward_resid_post_intervention`` edits and reads at the same position,
    which is right for a single-question prompt. Trajectory experiments need to
    edit an earlier step's carrier and read the decision produced many tokens
    later, so this variant separates the three positions:

    ``edit_token_indices``
        per-row token position whose resid_post at ``edit_layer`` is edited;
    ``readout_token_indices``
        per-row token position whose next-token logits are returned;
    ``capture_token_indices``
        per-row token position captured at each layer in ``capture_layers``
        (defaults to ``readout_token_indices``).

    ``capture_layers`` must lie strictly after ``edit_layer``: an edit to the
    resid_post of layer L at position p cannot change layer L at any other
    position, so capturing propagation at L itself would be vacuous.

    All hooks are removed in ``finally``. The edited residual is cloned before
    modification so the untouched forward is never mutated in place.
    """
    if adapter.model is None or adapter.tokenizer is None:
        raise RuntimeError("adapter.load() must be called before intervention")
    n = len(prompts)
    if n == 0:
        raise ValueError("intervention batch may not be empty")
    if len(edit_token_indices) != n or len(readout_token_indices) != n:
        raise ValueError("edit/readout token index length mismatch")
    delta_np = np.asarray(deltas)
    if delta_np.shape != (n, adapter.hidden_size):
        raise ValueError(
            f"deltas must have shape {(n, adapter.hidden_size)}, got {delta_np.shape}"
        )
    if not np.isfinite(delta_np).all():
        raise ValueError("intervention deltas must be finite")
    if not (0 <= int(edit_layer) < adapter.num_layers):
        raise ValueError(f"edit layer {edit_layer} out of range")
    if not output_token_ids:
        raise ValueError("output_token_ids may not be empty")
    capture = sorted({int(x) for x in capture_layers})
    bad = [x for x in capture if x <= int(edit_layer) or x >= adapter.num_layers]
    if bad:
        raise ValueError(
            "capture_layers must lie strictly after the edit layer and in range: "
            f"{bad}; edit_layer={edit_layer}, num_layers={adapter.num_layers}"
        )
    capture_idx_source = (
        readout_token_indices if capture_token_indices is None else capture_token_indices
    )
    if len(capture_idx_source) != n:
        raise ValueError("capture token index length mismatch")

    batch = adapter.tokenize(prompts)
    input_ids = batch["input_ids"].to(adapter.device)
    attention_mask = batch["attention_mask"].to(adapter.device)
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

    def _as_index(values: Sequence[int], name: str) -> torch.Tensor:
        tensor = torch.as_tensor(
            list(map(int, values)), dtype=torch.long, device=adapter.device
        )
        if torch.any(tensor < 0) or torch.any(tensor >= lengths):
            raise ValueError(f"{name} contains positions outside the prompt length")
        return tensor

    edit_idx = _as_index(edit_token_indices, "edit_token_indices")
    readout_idx = _as_index(readout_token_indices, "readout_token_indices")
    capture_idx = _as_index(capture_idx_source, "capture_token_indices")

    delta_t = torch.as_tensor(
        delta_np, device=adapter.device, dtype=next(adapter.model.parameters()).dtype
    )
    row_ids = torch.arange(n, device=adapter.device)
    captured: dict[int, torch.Tensor] = {}
    hooks: list[Any] = []

    edited_state: dict[str, torch.Tensor] = {}

    def edit_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[row_ids, edit_idx] = edited[row_ids, edit_idx] + delta_t
        # The post-edit carrier state is free here and is the only faithful
        # source for setpoint fidelity diagnostics at the edit layer.
        edited_state["carrier"] = edited[row_ids, edit_idx].detach()
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def capture_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_id] = hidden[row_ids, capture_idx].detach()

        return hook

    hooks.append(adapter._raw_layer(int(edit_layer)).register_forward_hook(edit_hook))
    for layer_id in capture:
        hooks.append(
            adapter._raw_layer(layer_id).register_forward_hook(capture_hook(layer_id))
        )

    try:
        with torch.inference_mode():
            outputs = adapter.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in hooks:
            handle.remove()

    logits = outputs.logits[row_ids, readout_idx]
    token_ids = torch.as_tensor(
        list(map(int, output_token_ids)), dtype=torch.long, device=adapter.device
    )
    selected = logits.index_select(-1, token_ids).detach().float().cpu().numpy()

    missing = [layer_id for layer_id in capture if layer_id not in captured]
    if missing:
        raise RuntimeError(f"failed to capture propagation layers: {missing}")

    if "carrier" not in edited_state:
        raise RuntimeError("edit hook did not fire; carrier state unavailable")

    return {
        "selected_logits": selected,
        "edited_carrier_state": edited_state["carrier"].float().cpu().numpy(),
        "captured": {
            layer_id: tensor.float().cpu().numpy()
            for layer_id, tensor in captured.items()
        },
        "edit_token_indices": list(map(int, edit_token_indices)),
        "readout_token_indices": list(map(int, readout_token_indices)),
        "capture_token_indices": list(map(int, capture_idx_source)),
        "output_token_ids": list(map(int, output_token_ids)),
    }


def forward_multi_capture(
    adapter: HFAdapter,
    prompts: Sequence[str],
    *,
    readout_token_indices: Sequence[int],
    output_token_ids: Sequence[int],
    capture_specs: Sequence[tuple[str, int, Sequence[int]]] = (),
) -> dict[str, Any]:
    """Unedited forward returning selected logits plus many named site captures.

    ``capture_specs`` is a sequence of ``(name, layer, token_indices)``: each entry
    gathers ``resid_post`` of ``layer`` at a per-row token position. One hook is
    registered per distinct layer regardless of how many names share it, so a
    trajectory experiment can read a carrier, an irrelevant-state site, an
    intermediate step and the decision position from a single forward pass.

    This is the paired clean reference for :func:`forward_resid_post_edit`.
    """
    if adapter.model is None or adapter.tokenizer is None:
        raise RuntimeError("adapter.load() must be called before forward")
    n = len(prompts)
    if n == 0 or len(readout_token_indices) != n:
        raise ValueError("prompts must be non-empty with one readout index per row")
    if not output_token_ids:
        raise ValueError("output_token_ids may not be empty")
    specs = [(str(name), int(layer), list(map(int, idx))) for name, layer, idx in capture_specs]
    if len({name for name, _l, _i in specs}) != len(specs):
        raise ValueError("capture spec names must be unique")
    for name, layer, idx in specs:
        if not (0 <= layer < adapter.num_layers):
            raise ValueError(f"capture spec {name!r} layer {layer} out of range")
        if len(idx) != n:
            raise ValueError(f"capture spec {name!r} token index length mismatch")

    batch = adapter.tokenize(prompts)
    input_ids = batch["input_ids"].to(adapter.device)
    attention_mask = batch["attention_mask"].to(adapter.device)
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

    def _as_index(values: Sequence[int], name: str) -> torch.Tensor:
        tensor = torch.as_tensor(
            list(map(int, values)), dtype=torch.long, device=adapter.device
        )
        if torch.any(tensor < 0) or torch.any(tensor >= lengths):
            raise ValueError(f"{name} contains positions outside the prompt length")
        return tensor

    readout_idx = _as_index(readout_token_indices, "readout_token_indices")
    spec_idx = {name: _as_index(idx, name) for name, _layer, idx in specs}
    by_layer: dict[int, list[str]] = {}
    for name, layer, _idx in specs:
        by_layer.setdefault(layer, []).append(name)

    row_ids = torch.arange(n, device=adapter.device)
    captured: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def capture_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            for name in by_layer[layer_id]:
                captured[name] = hidden[row_ids, spec_idx[name]].detach()

        return hook

    for layer_id in sorted(by_layer):
        hooks.append(
            adapter._raw_layer(layer_id).register_forward_hook(capture_hook(layer_id))
        )

    try:
        with torch.inference_mode():
            outputs = adapter.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in hooks:
            handle.remove()

    logits = outputs.logits[row_ids, readout_idx]
    token_ids = torch.as_tensor(
        list(map(int, output_token_ids)), dtype=torch.long, device=adapter.device
    )
    selected = logits.index_select(-1, token_ids).detach().float().cpu().numpy()
    missing = [name for name, _l, _i in specs if name not in captured]
    if missing:
        raise RuntimeError(f"failed to capture sites: {missing}")
    return {
        "selected_logits": selected,
        "captured": {
            name: captured[name].float().cpu().numpy() for name, _l, _i in specs
        },
    }


def forward_resid_post_span_edit(
    adapter: HFAdapter,
    prompts: Sequence[str],
    *,
    edit_layer: int,
    edit_token_indices: Sequence[Sequence[int]],
    deltas: Sequence[np.ndarray],
    readout_token_indices: Sequence[int],
    output_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Additive resid_post edit over a ragged multi-token span per row.

    ``forward_resid_post_edit`` edits exactly one position per row, which cannot
    express "replace the whole clause". This variant takes, for each row, a list
    of token positions and a matching ``[n_positions, hidden]`` delta block, so a
    counterfactual span replacement is a single hook.

    Spans may differ in length across rows; positions are scattered individually
    rather than assuming a rectangular window. The readout position is separate
    and may lie outside every edited span.

    Returns the selected readout logits plus the post-edit states at the edited
    positions, so span fidelity can be checked the same way single-token fidelity
    is.
    """
    if adapter.model is None or adapter.tokenizer is None:
        raise RuntimeError("adapter.load() must be called before intervention")
    n = len(prompts)
    if n == 0:
        raise ValueError("intervention batch may not be empty")
    if len(edit_token_indices) != n or len(deltas) != n or len(readout_token_indices) != n:
        raise ValueError("span edit inputs must all have one entry per prompt")
    if not output_token_ids:
        raise ValueError("output_token_ids may not be empty")
    if not (0 <= int(edit_layer) < adapter.num_layers):
        raise ValueError(f"edit layer {edit_layer} out of range")

    blocks: list[np.ndarray] = []
    for row, (positions, block) in enumerate(zip(edit_token_indices, deltas)):
        array = np.asarray(block, dtype=np.float64)
        if array.ndim != 2 or array.shape != (len(positions), adapter.hidden_size):
            raise ValueError(
                f"row {row} delta block must be [n_positions, hidden]; got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("intervention deltas must be finite")
        if len(set(map(int, positions))) != len(positions):
            raise ValueError(f"row {row} repeats an edit position")
        blocks.append(array)

    batch = adapter.tokenize(prompts)
    input_ids = batch["input_ids"].to(adapter.device)
    attention_mask = batch["attention_mask"].to(adapter.device)
    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)

    readout_idx = torch.as_tensor(
        list(map(int, readout_token_indices)), dtype=torch.long, device=adapter.device
    )
    if torch.any(readout_idx < 0) or torch.any(readout_idx >= lengths):
        raise ValueError("readout_token_indices contains positions outside the prompt")

    # Flatten the ragged spans into (row, position) scatter coordinates.
    flat_rows: list[int] = []
    flat_cols: list[int] = []
    flat_delta: list[np.ndarray] = []
    row_lengths = lengths.tolist()
    for row, positions in enumerate(edit_token_indices):
        for offset, position in enumerate(positions):
            index = int(position)
            if index < 0 or index >= int(row_lengths[row]):
                raise ValueError(
                    f"row {row} edit position {index} is outside the prompt length"
                )
            flat_rows.append(row)
            flat_cols.append(index)
            flat_delta.append(blocks[row][offset])
    if not flat_rows:
        raise ValueError("span edit received no positions to edit")

    param_dtype = next(adapter.model.parameters()).dtype
    rows_t = torch.as_tensor(flat_rows, dtype=torch.long, device=adapter.device)
    cols_t = torch.as_tensor(flat_cols, dtype=torch.long, device=adapter.device)
    delta_t = torch.as_tensor(
        np.stack(flat_delta), device=adapter.device, dtype=param_dtype
    )
    captured: dict[str, torch.Tensor] = {}

    def edit_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[rows_t, cols_t] = edited[rows_t, cols_t] + delta_t
        captured["edited"] = edited[rows_t, cols_t].detach()
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = adapter._raw_layer(int(edit_layer)).register_forward_hook(edit_hook)
    try:
        with torch.inference_mode():
            outputs = adapter.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    if "edited" not in captured:
        raise RuntimeError("span edit hook did not fire")
    logits = outputs.logits[torch.arange(n, device=adapter.device), readout_idx]
    token_ids = torch.as_tensor(
        list(map(int, output_token_ids)), dtype=torch.long, device=adapter.device
    )
    selected = logits.index_select(-1, token_ids).detach().float().cpu().numpy()

    edited_flat = captured["edited"].float().cpu().numpy()
    edited_by_row: list[np.ndarray] = []
    cursor = 0
    for positions in edit_token_indices:
        span = len(positions)
        edited_by_row.append(edited_flat[cursor : cursor + span].copy())
        cursor += span

    return {
        "selected_logits": selected,
        "edited_states": edited_by_row,
        "edit_token_indices": [list(map(int, p)) for p in edit_token_indices],
        "readout_token_indices": list(map(int, readout_token_indices)),
        "output_token_ids": list(map(int, output_token_ids)),
    }
