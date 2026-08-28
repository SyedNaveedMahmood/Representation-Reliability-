"""Stable local adapter for the frozen E14 weight-only quantization ladder."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any

E14_PRECISIONS = ("bf16", "int8", "int4")


def normalize_precision(value: str) -> str:
    precision = str(value).strip().lower()
    if precision not in E14_PRECISIONS:
        raise ValueError(f"precision must be one of {E14_PRECISIONS}, got {value!r}")
    return precision


def quantize_weight_only(model: Any, precision: str) -> dict[str, Any]:
    """Apply the preregistered Quanto ladder in-place and return provenance."""
    precision = normalize_precision(precision)
    if precision == "bf16":
        return {
            "backend": "none",
            "backend_version": None,
            "precision": precision,
            "weight_qtype": None,
            "activation_qtype": None,
            "calibration": None,
            "quantized_modules": [],
            "excluded_module_classes": ["Embedding", "RMSNorm"],
        }
    try:
        from optimum.quanto import freeze, qint4, qint8, quantize
    except ImportError as exc:  # pragma: no cover - installation failure
        raise RuntimeError(
            "E14 requires the quantization extra: pip install -e .[quantization]"
        ) from exc

    qtype = qint8 if precision == "int8" else qint4
    quantize(model, weights=qtype, activations=None)
    freeze(model)
    modules: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        weight_qtype = getattr(module, "weight_qtype", None)
        if weight_qtype is None:
            continue
        weight = getattr(module, "weight", None)
        modules.append(
            {
                "name": str(name),
                "module_class": type(module).__name__,
                "weight_tensor_class": type(weight).__name__ if weight is not None else None,
                "weight_qtype": str(weight_qtype),
                "weight_group_size": getattr(module, "weight_group_size", None),
                "activation_qtype": (
                    str(module.activation_qtype)
                    if getattr(module, "activation_qtype", None) is not None
                    else None
                ),
            }
        )
    if not modules:
        raise RuntimeError("Quanto did not replace any model modules")
    return {
        "backend": "optimum-quanto",
        "backend_version": version("optimum-quanto"),
        "precision": precision,
        "weight_qtype": str(qtype),
        "activation_qtype": None,
        "compute_dtype": "bfloat16",
        "calibration": None,
        "zero_point": False,
        "quantized_modules": modules,
        "excluded_module_classes": ["Embedding", "RMSNorm"],
    }
