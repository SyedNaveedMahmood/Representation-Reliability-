from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

SiteName = Literal["resid_pre", "attn_out", "mlp_out", "resid_post"]
InterventionOperator = Literal["replace", "add", "ablate", "rotate", "kv_replace"]

@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str
    target_label: str | int | float
    task_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    pair_id: str | None = None
    counterfactual_id: str | None = None
    expected_counterfactual_label: str | int | float | None = None

@dataclass(frozen=True)
class TokenSelection:
    strategy: str
    index: int
    token_id: int
    token_text: str
    char_start: int | None = None
    char_end: int | None = None

@dataclass(frozen=True)
class RepresentationSite:
    site: SiteName
    layer: int
    token: TokenSelection
    component: str | None = None
    native_module_name: str | None = None

@dataclass(frozen=True)
class ActivationKey:
    sample_id: str
    model_id: str
    site: RepresentationSite

@dataclass
class ActivationBatch:
    keys: Sequence[ActivationKey]
    values: np.ndarray
    dtype: str

@dataclass(frozen=True)
class InterventionSpec:
    operator: InterventionOperator
    site: RepresentationSite
    alpha: float = 1.0
    source_sample_id: str | None = None
    direction_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Prediction:
    sample_id: str
    text: str
    token_ids: Sequence[int]
    score: float | None = None
    is_correct: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class InterventionResult:
    base: Prediction
    intervened: Prediction
    spec: InterventionSpec
    target_metric_before: float
    target_metric_after: float
    delta_norm: float
    activation_norm: float
    output_kl: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ProbeResult:
    site: RepresentationSite
    auroc: float
    auprc: float
    balanced_accuracy: float
    coefficients_path: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SteeringPoint:
    alpha: float
    target_effect: float
    side_effect: float | None
    delta_norm: float

class ModelAdapter(Protocol):
    model_id: str

    def tokenize(self, prompts: Sequence[str]) -> Mapping[str, Any]:
        ...

    def generate(self, prompts: Sequence[str], **kwargs: Any) -> Sequence[Prediction]:
        ...

    def extract(
        self,
        samples: Sequence[Sample],
        sites: Sequence[RepresentationSite],
    ) -> ActivationBatch:
        ...

    def intervene(
        self,
        samples: Sequence[Sample],
        specs: Sequence[InterventionSpec],
    ) -> Sequence[InterventionResult]:
        ...

class ActivationStore(Protocol):
    def put(self, batch: ActivationBatch) -> None:
        ...

    def get(self, keys: Sequence[ActivationKey]) -> ActivationBatch:
        ...

    def has(self, keys: Sequence[ActivationKey]) -> Sequence[bool]:
        ...

class Probe(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        ...

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        ...

    def coefficients(self) -> np.ndarray:
        ...

class Transform(Protocol):
    name: str
    transform_class: Literal["invariant", "controlled_change"]

    def apply(self, sample: Sample, seed: int) -> Sample:
        ...
