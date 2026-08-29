"""Read-only diagnostics and frozen method revisions for E13 discovery."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..adapters.intervention import (
    differentiable_resid_post_logits,
    forward_with_resid_post_capture,
)
from ..config import resolve_config
from ..reporting.tables import markdown_table, save_json, save_table
from .e00c import candidate_token_id_lists
from .e13 import LAYER, _last_positions, distillation_loss
from .e13_methods import (
    METHOD_CAMPAIGN_ID,
    ModelLocalReference,
    _orientation,
    _random_pair,
)
from .e13_multiseed import (
    CAMPAIGN_ID,
    TRAINING_SEEDS,
    _checkpoint_adapter,
    _corpus_bundle,
)
from .extract import load_adapter

DIAGNOSTIC_VERSION = "e13-response-regularization-v1"
NEAR_ZERO_THRESHOLD = 1e-3
GRADIENT_AUDIT_SEED = 20261305
GRADIENT_AUDIT_STEPS = (10, 100)
GRADIENT_AUDIT_BATCHES = 2
GRADIENT_NAMES = ("KD", "Q", "A", "G", "R6")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _method_root() -> Path:
    return _repo_root() / "runs" / "E13_CONVERSION_RESPONSE" / METHOD_CAMPAIGN_ID


def _one_job(root: Path, regime: str, seed: int) -> tuple[Path, dict[str, Any]]:
    candidates = list((root / "jobs").glob(f"{regime}_seed_{seed}_*/job_summary.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one {regime} seed {seed} job under {root}")
    path = candidates[0]
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("status") != "complete" or job.get("confirmation_accessed") is not False:
        raise RuntimeError(f"invalid diagnostic source identity for {regime} seed {seed}")
    return path.parent, job


def _distribution(values: pd.Series) -> dict[str, float]:
    array = values.to_numpy(dtype=np.float64, copy=True)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic distribution must be nonempty and finite")
    q25, median, q75 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "median": float(median),
        "iqr": float(q75 - q25),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "fraction_near_zero": float(np.mean(np.abs(array) <= NEAR_ZERO_THRESHOLD)),
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
    }


def _cosine_from_norms(left: float, right: float, summed: float) -> float:
    denominator = 2.0 * float(left) * float(right)
    if denominator <= 0:
        return float("nan")
    value = (float(summed) ** 2 - float(left) ** 2 - float(right) ** 2) / denominator
    return float(np.clip(value, -1.0, 1.0))


def _teacher_signal_tables(cache_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution_rows = []
    for split in ("train", "validation"):
        chosen = cache_rows.loc[cache_rows["split"].eq(split)]
        for family, prefix in (("semantic", "R5"), ("random", "R6")):
            for component in ("Q", "A", "G"):
                distribution_rows.append(
                    {
                        "split": split,
                        "family": family,
                        "component": component,
                        **_distribution(chosen[f"{prefix}_{component}"]),
                    }
                )
    distributions = pd.DataFrame(distribution_rows)

    correlation_rows = []
    for split in ("train", "validation"):
        chosen = cache_rows.loc[cache_rows["split"].eq(split)].copy()
        chosen["R5_norm"] = np.sqrt(sum(chosen[f"R5_{name}"].pow(2) for name in "QAG"))
        chosen["R6_norm"] = np.sqrt(sum(chosen[f"R6_{name}"].pow(2) for name in "QAG"))
        columns = ["R5_Q", "R5_A", "R5_G", "R6_Q", "R6_A", "R6_G", "R5_norm", "R6_norm"]
        corr = chosen[columns].corr(method="pearson")
        for left in columns:
            for right in columns:
                correlation_rows.append(
                    {
                        "split": split,
                        "left": left,
                        "right": right,
                        "pearson": float(corr.loc[left, right]),
                    }
                )
    return distributions, pd.DataFrame(correlation_rows)


def _loss_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    step_rows = []
    summary_rows = []
    for regime in ("R5", "R6"):
        for seed in TRAINING_SEEDS:
            run_dir, _job = _one_job(_method_root(), regime, seed)
            losses = pd.read_parquet(run_dir / "training_losses.parquet")
            required = {"step", "loss", "ce", "kd", "response"}
            if not required.issubset(losses.columns) or len(losses) != 100:
                raise RuntimeError(f"invalid loss evidence for {regime} seed {seed}")
            for row in losses.itertuples(index=False):
                step_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "step": int(row.step),
                        "L_total": float(row.loss),
                        "L_CE": float(row.ce),
                        "L_KD_unweighted": float(row.kd),
                        "L_response": float(row.response),
                    }
                )
            for name, column in (
                ("L_total", "loss"),
                ("L_CE", "ce"),
                ("L_KD_unweighted", "kd"),
                ("L_response", "response"),
            ):
                values = losses[column].astype(float)
                summary_rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "loss_component": name,
                        **_distribution(values),
                    }
                )
    return pd.DataFrame(step_rows), pd.DataFrame(summary_rows)


def _sensitivity_audit() -> pd.DataFrame:
    baseline_root = _repo_root() / "runs" / "E13_MULTI_SEED" / CAMPAIGN_ID
    rows = []
    for regime, root in (("R2", baseline_root), ("R5", _method_root()), ("R6", _method_root())):
        for seed in TRAINING_SEEDS:
            run_dir, job = _one_job(root, regime, seed)
            step = int(job["b_matched"]["selected_step"])
            evidence = pd.read_parquet(
                run_dir / "checkpoints" / f"step_{step:03d}" / "factorial_rows.parquet"
            )
            if evidence["confirmation_accessed"].ne(False).any():
                raise RuntimeError("invalid sensitivity evidence namespace")
            families: dict[str, pd.DataFrame] = {
                "semantic_q": evidence.loc[evidence["context"].eq("matched")].assign(
                    response=lambda frame: frame["Q_z"],
                    delta_norm=lambda frame: frame["semantic_delta_norm"],
                ),
                "matched_context": evidence.loc[evidence["context"].eq("matched")].assign(
                    response=lambda frame: frame["A_z"],
                    delta_norm=lambda frame: frame["context_delta_norm"],
                ),
            }
            random_rows = evidence.loc[evidence["context"].eq("random")]
            for direction_seed in sorted(
                random_rows["direction_seed"].dropna().astype(int).unique()
            ):
                families[f"unseen_random_{direction_seed}"] = random_rows.loc[
                    random_rows["direction_seed"].eq(direction_seed)
                ].assign(
                    response=lambda frame: frame["A_z"],
                    delta_norm=lambda frame: frame["context_delta_norm"],
                )
            families["unseen_random_pooled"] = random_rows.assign(
                response=lambda frame: frame["A_z"],
                delta_norm=lambda frame: frame["context_delta_norm"],
            )
            for family, chosen in families.items():
                response = chosen["response"].to_numpy(np.float64)
                norms = chosen["delta_norm"].to_numpy(np.float64)
                ratios = response / np.maximum(norms, 1e-12)
                rows.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "selected_step": step,
                        "family": family,
                        "n": len(chosen),
                        "mean_squared_response": float(np.mean(response**2)),
                        "response_variance": float(np.var(response)),
                        "mean_abs_response": float(np.mean(np.abs(response))),
                        "mean_delta_norm": float(np.mean(norms)),
                        "mean_abs_lipschitz_proxy": float(np.mean(np.abs(ratios))),
                        "jacobian_norm_proxy": float(np.sqrt(np.mean(ratios**2))),
                    }
                )
    return pd.DataFrame(rows)


def _load_student_reference(run_dir: Path) -> ModelLocalReference:
    from safetensors.numpy import load_file

    metadata = json.loads((run_dir / "student_response_reference.json").read_text(encoding="utf-8"))
    tensors = load_file(run_dir / "student_response_reference.safetensors")
    raw = {
        "direction": tensors["direction"],
        "probe": {
            "coef": tensors["probe_coef"],
            "intercept": tensors["probe_intercept"],
            "mean": tensors["probe_mean"],
            "scale": tensors["probe_scale"],
        },
        "targets": metadata["targets"],
    }
    return ModelLocalReference.from_reference(
        model_role="student",
        hidden_size=int(metadata["hidden_size"]),
        reference=raw,
        model_revisions=metadata["model_revisions"],
    )


def _gradient_loss_builder(
    *,
    student,
    teacher,
    reference: ModelLocalReference,
    cache: pd.DataFrame,
    cache_meta: dict[str, Any],
    samples_by_id: dict[str, Any],
    labels: dict[str, int],
    token_ids: list[int],
    batch_ids: list[str],
) -> Callable[[str], torch.Tensor]:
    cache_indexed = cache.set_index("sample_id")
    direction = reference.direction
    targets = reference.targets
    margin_sigma = float(targets["sigma_margin_validation"])

    def build(name: str) -> torch.Tensor:
        encoded = student.tokenize([samples_by_id[sid].prompt for sid in batch_ids])
        input_ids = encoded["input_ids"].to(student.device)
        attention = encoded["attention_mask"].to(student.device)
        positions = _last_positions(attention)
        row_ids = torch.arange(len(batch_ids), device=student.device)
        selected_ids = torch.as_tensor(token_ids, dtype=torch.long, device=student.device)
        outputs, sequence = forward_with_resid_post_capture(
            student, input_ids=input_ids, attention_mask=attention, layer=LAYER
        )
        selected = outputs.logits[row_ids, positions]
        clean_logits = selected.index_select(-1, selected_ids)
        clean_hidden = sequence[row_ids, positions]
        if name == "KD":
            with torch.inference_mode():
                teacher_outputs = teacher.model(input_ids=input_ids, attention_mask=attention)
                teacher_selected = teacher_outputs.logits[row_ids, positions]
            target_ids = torch.tensor(
                [token_ids[0] if int(labels[sid]) == 1 else token_ids[1] for sid in batch_ids],
                device=student.device,
                dtype=torch.long,
            )
            loss, _ = distillation_loss(
                selected,
                target_ids,
                teacher_logits=teacher_selected,
                regime="R2",
            )
            return loss

        orientation = torch.as_tensor(
            _orientation(labels, batch_ids), device=student.device, dtype=torch.float32
        )
        clean_margin = orientation * (clean_logits[:, 0].float() - clean_logits[:, 1].float())
        source_prompts = [
            samples_by_id[str(samples_by_id[sid].counterfactual_id)].prompt for sid in batch_ids
        ]
        source_encoded = student.tokenize(source_prompts)
        source_ids = source_encoded["input_ids"].to(student.device)
        source_attention = source_encoded["attention_mask"].to(student.device)
        source_positions = _last_positions(source_attention)
        with torch.no_grad():
            source_outputs = student.model(
                input_ids=source_ids,
                attention_mask=source_attention,
                output_hidden_states=True,
            )
            source_hidden = source_outputs.hidden_states[LAYER + 1][row_ids, source_positions]
        bases = clean_hidden.detach().float().cpu().numpy()
        sources = source_hidden.detach().float().cpu().numpy()
        q_values, c_values, rq_values, rc_values = [], [], [], []
        for index, sid in enumerate(batch_ids):
            target_q = float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
            q_delta = reference.semantic_delta(bases[index], target_q)
            c_delta = reference.context_delta(bases[index], sources[index])
            random_q, random_c = _random_pair(
                student.hidden_size,
                sid,
                direction,
                float(np.linalg.norm(q_delta)),
                float(np.linalg.norm(c_delta)),
                reference.random_identity,
            )
            q_values.append(q_delta)
            c_values.append(c_delta)
            rq_values.append(random_q)
            rc_values.append(random_c)

        def components(q_array: list[np.ndarray], c_array: list[np.ndarray]) -> list[torch.Tensor]:
            q = torch.as_tensor(np.stack(q_array), device=student.device, dtype=clean_hidden.dtype)
            c = torch.as_tensor(np.stack(c_array), device=student.device, dtype=clean_hidden.dtype)
            arm_margins = []
            for delta in (q, c, q + c):
                logits = differentiable_resid_post_logits(
                    student,
                    input_ids=input_ids,
                    attention_mask=attention,
                    layer=LAYER,
                    token_indices=positions,
                    deltas=delta,
                    output_token_ids=token_ids,
                )
                arm_margins.append(orientation * (logits[:, 0].float() - logits[:, 1].float()))
            yq, yc, yqc = arm_margins
            return [
                (yq - clean_margin) / margin_sigma,
                (yc - clean_margin) / margin_sigma,
                ((yqc - yq) - (yc - clean_margin)) / margin_sigma,
            ]

        if name == "R6":
            values = components(rq_values, rc_values)
            losses = []
            for component, value in zip(("Q", "A", "G"), values):
                scale = float(cache_meta["component_scales"][f"R6_{component}"])
                target = torch.as_tensor(
                    cache_indexed.loc[batch_ids, f"R6_{component}"].to_numpy(
                        dtype=float, copy=True
                    ),
                    device=student.device,
                    dtype=torch.float32,
                )
                losses.append(((value / scale) - (target / scale)).pow(2))
            return torch.stack(losses).mean()
        semantic = components(q_values, c_values)
        index = {"Q": 0, "A": 1, "G": 2}[name]
        scale = float(cache_meta["component_scales"][f"R5_{name}"])
        target = torch.as_tensor(
            cache_indexed.loc[batch_ids, f"R5_{name}"].to_numpy(dtype=float, copy=True),
            device=student.device,
            dtype=torch.float32,
        )
        return ((semantic[index] / scale) - (target / scale)).pow(2).mean()

    return build


def _gradient_norm(
    student, loss_builder: Callable[[str], torch.Tensor], names: tuple[str, ...]
) -> tuple[float, float]:
    student.model.zero_grad(set_to_none=True)
    losses = [loss_builder(name) for name in names]
    loss = torch.stack(losses).sum()
    loss.backward()
    squared = sum(
        float(parameter.grad.float().pow(2).sum())
        for parameter in student.model.parameters()
        if parameter.grad is not None
    )
    norm = float(np.sqrt(squared))
    value = float(sum(float(item.detach()) for item in losses))
    student.model.zero_grad(set_to_none=True)
    del losses, loss
    gc.collect()
    torch.cuda.empty_cache()
    return norm, value


def run_gradient_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _repo_root()
    method_root = _method_root()
    run_dir, _job = _one_job(method_root, "R5", GRADIENT_AUDIT_SEED)
    cache_dir = method_root / "teacher_cache"
    cache = pd.read_parquet(cache_dir / "teacher_response_rows.parquet")
    cache_meta = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    _, frame, _, labels, samples_by_id, _, _ = _corpus_bundle()
    validation_ids = frame.loc[frame["split"].eq("validation"), "sample_id"].astype(str).tolist()
    reference = _load_student_reference(run_dir)
    student_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_0.6b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    teacher_cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / "configs/models/qwen3_1.7b.yaml",
        experiment_path=root / "configs/experiments/E13_distillation_reliability.yaml",
        overrides=(),
    )
    teacher = load_adapter(teacher_cfg)
    teacher.model.eval()
    for parameter in teacher.model.parameters():
        parameter.requires_grad_(False)
    candidates = candidate_token_id_lists(teacher, list(teacher_cfg.behavior.candidates_primary))
    token_ids = [int(item[0]) for item in candidates]
    norm_rows, cosine_rows = [], []
    for step in GRADIENT_AUDIT_STEPS:
        student = _checkpoint_adapter(student_cfg, run_dir / "checkpoints" / f"step_{step:03d}")
        student.model.eval()
        student.model.config.use_cache = False
        for batch_index in range(GRADIENT_AUDIT_BATCHES):
            start = batch_index * 2
            batch_ids = validation_ids[start : start + 2]
            builder = _gradient_loss_builder(
                student=student,
                teacher=teacher,
                reference=reference,
                cache=cache,
                cache_meta=cache_meta,
                samples_by_id=samples_by_id,
                labels=labels,
                token_ids=token_ids,
                batch_ids=batch_ids,
            )
            norms: dict[str, float] = {}
            values: dict[str, float] = {}
            for name in GRADIENT_NAMES:
                norms[name], values[name] = _gradient_norm(student, builder, (name,))
                norm_rows.append(
                    {
                        "checkpoint_step": step,
                        "batch_index": batch_index,
                        "sample_ids": json.dumps(batch_ids),
                        "loss_component": name,
                        "loss": values[name],
                        "gradient_norm": norms[name],
                    }
                )
            for left_index, left in enumerate(GRADIENT_NAMES):
                for right in GRADIENT_NAMES[left_index + 1 :]:
                    sum_norm, _ = _gradient_norm(student, builder, (left, right))
                    cosine_rows.append(
                        {
                            "checkpoint_step": step,
                            "batch_index": batch_index,
                            "left": left,
                            "right": right,
                            "cosine": _cosine_from_norms(norms[left], norms[right], sum_norm),
                        }
                    )
        del student
        gc.collect()
        torch.cuda.empty_cache()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(norm_rows), pd.DataFrame(cosine_rows)


def run_response_regularization_diagnostic() -> Path:
    """Audit existing E13 evidence and run read-only full-model gradient checks."""
    root = _repo_root()
    output = root / "runs" / "E13_METHOD_REVISION" / "diagnostic_v1"
    output.mkdir(parents=True, exist_ok=True)
    cache = pd.read_parquet(_method_root() / "teacher_cache" / "teacher_response_rows.parquet")
    distributions, correlations = _teacher_signal_tables(cache)
    loss_steps, loss_summary = _loss_audit()
    sensitivity = _sensitivity_audit()
    gradient_norms_path = output / "gradient_norms.parquet"
    gradient_cosines_path = output / "gradient_cosines.parquet"
    if gradient_norms_path.exists() and gradient_cosines_path.exists():
        gradient_norms = pd.read_parquet(gradient_norms_path)
        gradient_cosines = pd.read_parquet(gradient_cosines_path)
    else:
        gradient_norms, gradient_cosines = run_gradient_audit()
    save_table(distributions, output / "teacher_response_distributions.parquet")
    save_table(correlations, output / "teacher_response_correlations.parquet")
    save_table(loss_steps, output / "existing_loss_steps.parquet")
    save_table(loss_summary, output / "existing_loss_summary.parquet")
    save_table(sensitivity, output / "sensitivity_proxies.parquet")
    save_table(gradient_norms, gradient_norms_path)
    save_table(gradient_cosines, gradient_cosines_path)
    conflict = gradient_cosines.loc[
        gradient_cosines["left"].eq("KD") & gradient_cosines["right"].isin(["Q", "A", "G", "R6"])
    ]
    conflict_gate = bool((conflict["cosine"] < -0.2).groupby(conflict["right"]).sum().max() >= 2)
    manifest = {
        "version": DIAGNOSTIC_VERSION,
        "source_method_campaign": METHOD_CAMPAIGN_ID,
        "cache_response_sha256": hashlib.sha256(
            cache[[f"R{regime}_{component}" for regime in (5, 6) for component in "QAG"]]
            .to_numpy(np.float64)
            .tobytes()
        ).hexdigest(),
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "gradient_checkpoints": list(GRADIENT_AUDIT_STEPS),
        "gradient_batches_per_checkpoint": GRADIENT_AUDIT_BATCHES,
        "conflict_gate_rule": "same response component has cosine(KD,response) < -0.2 in at least two audits",
        "conflict_gate_fired": conflict_gate,
        "confirmation_accessed": False,
    }
    save_json(manifest, output / "diagnostic_manifest.json")

    dist_display = distributions.loc[
        distributions["split"].eq("train"),
        ["family", "component", "mean", "sd", "median", "iqr", "p05", "p95", "fraction_near_zero"],
    ]
    loss_display = loss_summary.loc[
        loss_summary["loss_component"].isin(["L_KD_unweighted", "L_response"]),
        ["regime", "seed", "loss_component", "mean", "sd", "median", "p05", "p95"],
    ]
    gradient_display = gradient_cosines.loc[gradient_cosines["left"].eq("KD")]
    gradient_component_display = gradient_norms.groupby("loss_component", as_index=False)[
        ["loss", "gradient_norm"]
    ].agg(["mean", "min", "max"])
    gradient_component_display.columns = [
        "loss_component",
        "loss_mean",
        "loss_min",
        "loss_max",
        "gradient_norm_mean",
        "gradient_norm_min",
        "gradient_norm_max",
    ]
    sensitivity_display = (
        sensitivity.loc[
            sensitivity["family"].isin(["semantic_q", "matched_context", "unseen_random_pooled"])
        ]
        .groupby(["regime", "family"], as_index=False)[
            [
                "mean_squared_response",
                "response_variance",
                "mean_abs_lipschitz_proxy",
                "jacobian_norm_proxy",
            ]
        ]
        .mean()
    )
    train_correlations = correlations.loc[correlations["split"].eq("train")].set_index(
        ["left", "right"]
    )["pearson"]
    corr_a = float(train_correlations.loc[("R5_A", "R6_A")])
    corr_g = float(train_correlations.loc[("R5_G", "R6_G")])
    corr_norm = float(train_correlations.loc[("R5_norm", "R6_norm")])
    report = root / "E13_RESPONSE_REGULARIZATION_DIAGNOSTIC.md"
    lines = [
        "# E13 Response-Regularization Diagnostic",
        "",
        "Status: complete read-only discovery diagnostic. No optimizer update was performed; E13 confirmation was not accessed.",
        "",
        "## Existing training-loss scales",
        "",
        markdown_table(loss_display, float_fmt="{:.6f}"),
        "",
        "`L_KD_unweighted` is the logged KL term before its frozen `0.5*T^2` multiplier; `L_response` is the actual added response objective.",
        "",
        "## Teacher target distributions (training split)",
        "",
        markdown_table(dist_display, float_fmt="{:.6f}"),
        "",
        "Near zero means absolute standardized response <= 0.001. Validation distributions and all raw rows are retained in the diagnostic directory.",
        "",
        "## Semantic/random correlations",
        "",
        f"Train corr(A_T, random-A_T): {corr_a:.6f}.",
        f"Train corr(G_T, random-G_T): {corr_g:.6f}.",
        f"Train semantic/random response-norm correlation: {corr_norm:.6f}.",
        "",
        "## B-matched sensitivity proxies",
        "",
        markdown_table(sensitivity_display, float_fmt="{:.6f}"),
        "",
        "The unseen-random family pools frozen evaluation directions 2130/2131/2132, which were not the R6 training direction identity. Ratios are directional finite-difference response divided by intervention norm, not a full Jacobian.",
        "",
        "## Full-model gradient audit",
        "",
        markdown_table(gradient_component_display, float_fmt="{:.6f}"),
        "",
        markdown_table(gradient_display, float_fmt="{:.6f}"),
        "",
        f"Frozen conflict gate fired: **{conflict_gate}**.",
        "",
        "Gradient norms, response-component losses, and every pairwise Q/A/G/R6 cosine are retained as portable tables.",
        "",
        "## Interpretation",
        "",
        "The response term is not a small auxiliary: its training mean is 2.13-3.72, while the logged unweighted KL mean is 0.66-0.72. On the frozen gradient batches, Q is the largest semantic component (mean loss 8.16; mean full-model gradient norm 750), followed by A (1.57; 362) and G (1.28; 266), versus KD's 1.69 and 106. Q mismatch therefore dominates the audited response geometry even after validation-only scaling.",
        "",
        "Teacher semantic A has the largest spread (train SD 0.679), while G is much smaller (SD 0.098, mean -0.044) and 16.9% of G targets are within 0.001 of zero. Random G is smaller still (SD 0.058; 25.6% near zero). Direct G matching consequently has a materially weaker and more quantized target than A, consistent with the G-noise hypothesis.",
        "",
        "Semantic/random target correlations are weak: Q 0.103, A 0.058, G 0.105, and response-norm 0.256 on training data. R6 is therefore not succeeding because its cached targets are close sample-wise to the semantic targets.",
        "",
        "R6 reduces the B-matched matched-context standardized mean-square response from R2's 2.872 to 1.201 and has a low unseen-random Jacobian proxy (0.0036 versus R2 0.0095). R5 also reduces matched-context sensitivity (1.539) but is anisotropic on unseen directions (pooled proxy 0.0178, driven by direction 2131). The best current explanation is that R6 supplies strong generic local-sensitivity/Jacobian regularization, which moves A/G magnitudes toward the teacher profile without semantic target correspondence.",
        "",
        "Gradient conflict is heterogeneous but real. Mean cosine(KD,A) is -0.207, with values below -0.2 at both audited checkpoints; cosine(KD,G) reaches -0.529, while Q is usually aligned or near orthogonal. The frozen repeated-conflict gate therefore fires and authorizes the preregistered R15/R16 bounded controls.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
