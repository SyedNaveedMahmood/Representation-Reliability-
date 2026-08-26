"""Hugging Face model adapter.

Stable local API over HF internals. Scientific code must not touch
model-specific module paths directly; it resolves canonical sites through
:meth:`HFAdapter.resolve_site`.

Canonical sites (0-indexed layers):

- ``resid_pre``  : input of decoder layer ``i``
- ``attn_out``   : output of ``model.layers[i].self_attn``
- ``mlp_out``    : output of ``model.layers[i].mlp``
- ``resid_post`` : output of decoder layer ``i``

Extraction uses two independent mechanisms that can be cross-checked:
``output_hidden_states=True`` for residual streams, and forward hooks for
attention/MLP outputs (and optionally the residuals).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

SITE_NAMES = ("resid_pre", "attn_out", "mlp_out", "resid_post")


@dataclass(frozen=True)
class SiteResolution:
    site: str
    layer: int                       # 0-indexed transformer layer
    native_module_path: str          # human-readable model-native path
    native_module_name: str          # attribute path within the nn.Module tree


class HFAdapter:
    def __init__(self, model_cfg: Any, runtime_cfg: Any | None = None) -> None:
        self.model_id: str = model_cfg.id
        self.revision = (
            None if model_cfg.revision in (None, "", "null") else str(model_cfg.revision)
        )
        self.torch_dtype = TORCH_DTYPES[model_cfg.dtype]
        self.device_map = model_cfg.device_map
        self.trust_remote_code = bool(model_cfg.trust_remote_code)
        self.runtime_cfg = runtime_cfg
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._num_layers: int | None = None
        self._hidden_size: int | None = None
        self._decoder_module: Any | None = None
        self._prefix: str = ""
        # Verified mapping convention for output_hidden_states (see calibrate()).
        # transformers >=4.5x: hidden_states[k] = residual stream ENTERING layer k;
        # the final element is post-final-norm (never a raw resid_post).
        self.hs_enters_layer_convention: bool | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> HFAdapter:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok_kwargs: dict[str, Any] = {}
        if self.revision:
            tok_kwargs["revision"] = self.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.revision:
            model_kwargs["revision"] = self.revision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        )
        self.model.eval()

        self._decoder_module, self._prefix = self._find_decoder_stack()
        self._num_layers = len(self._decoder_module.layers)
        cfg = getattr(self.model, "config", None)
        self._hidden_size = int(getattr(cfg, "hidden_size", 0)) or None
        self.calibrate()
        return self

    def calibrate(self) -> bool:
        """One probe forward verifying the output_hidden_states convention.

        Checks ``hs[1] == raw output of layer 0`` (entering-layer convention,
        transformers >= 4.5x).  Because of this convention the FINAL element is
        post-final-norm and can never serve as ``resid_post`` of the last
        layer; extraction therefore defaults to hooks for residual sites.
        """
        enc = self.tokenizer("hello world", return_tensors="pt")
        captured: dict[str, torch.Tensor] = {}
        h = self._raw_layer(0).register_forward_hook(
            lambda m, i, o: captured.__setitem__(
                "out", (o[0] if isinstance(o, tuple) else o).detach()
            )
        )
        try:
            with torch.inference_mode():
                out = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                    output_hidden_states=True,
                )
        finally:
            h.remove()
        if not getattr(out, "hidden_states", None):
            self.hs_enters_layer_convention = False
            return False
        n_layers = self.num_layers
        ok_last_normed = (
            len(out.hidden_states) == n_layers + 1
            or len(out.hidden_states) == n_layers
        )
        if len(out.hidden_states) > 1:
            dev = float(
                (out.hidden_states[1][0].float() - captured["out"][0].float())
                .abs()
                .max()
            )
            scale = max(float(out.hidden_states[1][0].abs().max()), 1e-9)
            self.hs_enters_layer_convention = bool(ok_last_normed and dev / scale < 1e-3)
        else:
            self.hs_enters_layer_convention = False
        return self.hs_enters_layer_convention

    def _raw_layer(self, layer: int):
        parts = f"{self._prefix}layers"
        obj = self.model
        for part in filter(None, parts.replace("[", ".").replace("]", "").split(".")):
            obj = getattr(obj, part)
        return obj[layer]

    def _find_decoder_stack(self) -> tuple[Any, str]:
        """Locate the Module holding ``layers`` across common HF architectures."""
        for prefix in ("model.", "transformer.", "decoder.", ""):
            obj = self.model
            ok = True
            for part in filter(None, prefix.split(".")):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and hasattr(obj, "layers"):
                return obj, prefix
        raise AttributeError(
            f"could not locate a decoder layer stack on {type(self.model).__name__}; "
            "add an explicit mapping to adapters/hf.py"
        )

    @property
    def num_layers(self) -> int:
        if self._num_layers is None:
            raise RuntimeError("adapter.load() must be called first")
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is None:
            raise RuntimeError("adapter.load() must be called first")
        return self._hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def resolve_site(self, site: str, layer: int) -> SiteResolution:
        """Map a canonical site + 0-indexed layer onto native modules."""
        if site not in SITE_NAMES:
            raise ValueError(f"unknown site {site!r}")
        if not (0 <= layer < self.num_layers):
            raise ValueError(f"layer {layer} out of range [0, {self.num_layers})")
        layer_attr = f"{self._prefix}layers[{layer}]"
        if site == "resid_post":
            native = layer_attr
        elif site == "resid_pre":
            native = f"{layer_attr} (input)"
        elif site == "attn_out":
            native = f"{layer_attr}.self_attn"
        else:  # mlp_out
            native = f"{layer_attr}.mlp"
        return SiteResolution(
            site=site, layer=layer,
            native_module_path=native, native_module_name=native,
        )

    def native_modules_for_sites(self, sites: Sequence[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        seen: set[tuple[str, int]] = set()
        for site in sites:
            n = self.num_layers if True else 0
            for layer in range(n):
                if (site, layer) in seen:
                    continue
                seen.add((site, layer))
                res = self.resolve_site(site, layer)
                out[f"{site}:{layer}"] = res.native_module_name
        return out

    # ---------------------------------------------------------------- tokenize
    def tokenize(self, prompts: Sequence[str], add_special_tokens: bool = True):
        return self.tokenizer(
            list(prompts),
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=False,
        )

    def encode_with_offsets(self, text: str) -> dict[str, Any]:
        enc = self.tokenizer(
            text, add_special_tokens=True, return_offsets_mapping=True
        )
        return {
            "input_ids": list(enc["input_ids"]),
            "offset_mapping": [tuple(map(int, pair)) for pair in enc["offset_mapping"]],
        }

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def default_generation_kwargs(self) -> dict[str, Any]:
        gen = getattr(self.runtime_cfg, "generation", None) if self.runtime_cfg else None
        max_new = getattr(gen, "max_new_tokens", 256)
        do_sample = bool(getattr(gen, "do_sample", False))
        return {"max_new_tokens": int(max_new), "do_sample": do_sample}

    def generate(self, prompts: Sequence[str], **gen_kwargs: Any) -> list[dict[str, Any]]:
        assert self.model is not None and self.tokenizer is not None
        kwargs = self.default_generation_kwargs()
        kwargs.update(gen_kwargs)
        enc = self.tokenize(prompts)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        results: list[dict[str, Any]] = []
        n_prompt = input_ids.shape[1]
        for i in range(out.shape[0]):
            new_tokens = out[i, n_prompt:].tolist()
            results.append({
                "prompt": prompts[i],
                "token_ids": new_tokens,
                "text": self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            })
        return results

    def forward_logits(
        self,
        prompts: Sequence[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Final-layer logits as float32 numpy ``[N, T, vocab]``."""
        assert self.model is not None
        if input_ids is None:
            if prompts is None:
                raise ValueError("forward_logits needs prompts or input_ids")
            enc = self.tokenize(prompts)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
                if attention_mask is not None else None,
            )
        return out.logits.float().cpu().numpy()

    # ---------------------------------------------------------------- extract
    def extract(
        self,
        prompts: Sequence[str],
        requests: Sequence[tuple[str, int]],
        token_indices: Sequence[int],
        use_hooks_for_resid: bool = True,
    ) -> dict[tuple[str, int], np.ndarray]:
        """One forward pass returning activations per requested site/layer.

        Returns ``{(site, layer): np.ndarray [N, hidden]}`` where row ``n`` is
        gathered at prompt ``n``'s ``token_indices[n]``.

        Hooks are the production path for ALL sites (correct by construction).
        ``use_hooks_for_resid=False`` forces the ``output_hidden_states`` path,
        which is only valid for interior layers under the entering-layer
        convention (the final element is post-final-norm) and is used for
        cross-validation testing.
        """
        assert self.model is not None and self.tokenizer is not None
        invalid = [r for r in requests if r[0] not in SITE_NAMES]
        if invalid:
            raise ValueError(f"unknown sites in extraction requests: {invalid}")
        for (site, layer) in requests:
            self.resolve_site(site, layer)

        resid_sites_requested = {
            s for s, _ in requests if s in ("resid_pre", "resid_post")
        }
        resid_via_hidden = bool(resid_sites_requested) and not use_hooks_for_resid
        if resid_via_hidden and self.hs_enters_layer_convention is not True:
            raise RuntimeError(
                "hidden_states extraction path requested but the convention "
                "was not verified at load(); use the default hook path"
            )

        captured: dict[tuple[str, int], list[torch.Tensor]] = {}
        hooks: list[Any] = []

        def out_hook(key: tuple[str, int]):
            def hook(_m: Any, _inputs: Any, output: Any):
                t = output[0] if isinstance(output, tuple) else output
                captured.setdefault(key, []).append(t.detach())
            return hook

        def in_hook(key: tuple[str, int]):
            def hook(_m: Any, inputs: Any, _output: Any):
                captured.setdefault(key, []).append(inputs[0].detach())
            return hook

        for (site, layer) in requests:
            if site == "attn_out":
                hooks.append(self._raw_layer(layer).self_attn.register_forward_hook(out_hook((site, layer))))
            elif site == "mlp_out":
                hooks.append(self._raw_layer(layer).mlp.register_forward_hook(out_hook((site, layer))))
            elif not resid_via_hidden:
                # Hook path for residuals (production default).
                if site == "resid_post":
                    hooks.append(self._raw_layer(layer).register_forward_hook(out_hook((site, layer))))
                else:
                    hooks.append(self._raw_layer(layer).register_forward_pre_hook(in_hook((site, layer))))

        batch = self.tokenize(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=resid_via_hidden,
                )
        finally:
            for h in hooks:
                h.remove()

        idx = torch.as_tensor(list(token_indices), dtype=torch.long).to(self.device)

        def gather(tensor: torch.Tensor) -> np.ndarray:
            rows = tensor[torch.arange(tensor.shape[0], device=tensor.device), idx]
            return rows.detach().float().cpu().numpy()

        results: dict[tuple[str, int], np.ndarray] = {}
        if resid_via_hidden:
            hs = outputs.hidden_states  # hs[k] = stream entering layer k; final el = post-norm.
            max_usable = self.num_layers - 2  # interior-only guarantee
            for (site, layer) in requests:
                if site == "resid_post":
                    if layer > max_usable:
                        raise RuntimeError(
                            f"hidden_states path cannot serve resid_post of the "
                            f"final layer ({self.num_layers - 1}); use hooks"
                        )
                    results[(site, layer)] = gather(hs[layer + 1])
                elif site == "resid_pre":
                    results[(site, layer)] = gather(hs[layer])
        for key, chunks in captured.items():
            full = torch.cat(chunks, dim=0)
            results[key] = gather(full)

        missing = [r for r in requests if r not in results]
        if missing:
            raise RuntimeError(f"extraction failed to produce: {missing}")
        return results


# --- PART4 ---


