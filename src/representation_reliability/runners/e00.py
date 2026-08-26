"""E00 — Representation Cartography.

Exploratory decodability experiment:
"At what layers and token positions is a controlled latent variable decodable
from the model's hidden states, and how stable is that decodability profile?"

This runner establishes DECODABILITY ONLY (D). Nothing here supports claims
that the model *uses* the variable — that requires causal interventions (E01).

Pipeline:
    generate -> split (group-based) -> extract (sharded/resumable)
    -> per-layer probes (val-only C selection) -> controls -> figures -> summary

The confirmation split is never extracted and never read during discovery.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.base import check_label_balance, samples_to_dataframe
from ..data.splits import (
    apply_splits,
    assign_group_splits,
    discovery_view,
    validate_splits,
)
from ..data.synthetic import generate_synthetic_relations
from ..metrics.bootstrap import bootstrap_ci
from ..metrics.decoding import (
    class_balance,
    classification_metrics,
    majority_baseline_accuracy,
)
from ..probes.linear import (
    evaluate_probe,
    fit_probe,
    random_feature_baseline,
    save_probe,
    shuffle_labels,
    transform_features,
)
from ..reporting.plots import plot_d_by_layer, plot_d_combined
from ..reporting.tables import markdown_table, save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .extract import expand_layer_plan, load_adapter, run_extraction_stage
from .probe import ShardedMatrixLoader, design_matrices_for, text_baseline_metrics

logger = logging.getLogger(__name__)

RANDOM_LABEL_SEEDS_DEFAULT = [0, 1, 2]


def _generate_dataset(cfg):
    samples = generate_synthetic_relations(
        n_samples=int(cfg.dataset.n_samples),
        seed=int(cfg.reproducibility.data_seed),
        n_entities=int(cfg.dataset.n_entities),
    )
    df = samples_to_dataframe(samples)
    assignment = assign_group_splits(
        groups=df["pair_id"].tolist(),
        seed=int(cfg.reproducibility.split_seed),
    )
    df = apply_splits(df, assignment)
    validate_splits(df)
    return samples, df


def run_e00(
    base_path: str | Path | None = None,
    model_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    overrides: tuple[str, ...] = (),
) -> Path:
    """Execute the full E00 pipeline; returns the run directory."""
    cfg, provenance = resolve_config(
        base_path=base_path, model_path=model_path,
        experiment_path=experiment_path, overrides=overrides,
    )
    c_hash = config_hash(cfg)
    logger.info("E00 config hash: %s", c_hash)

    # -- dataset & splits -------------------------------------------------
    samples, df = _generate_dataset(cfg)
    split_hash = dataset_split_hash(dict(zip(df["sample_id"], df["split"])))

    run_id = make_run_id(
        experiment_id=cfg.experiment.id,
        config_hash=c_hash,
        seed=int(cfg.reproducibility.seed),
        model_revision=cfg.model.revision or "main-unpinned",
        dataset_split_hash=split_hash,
    )
    repo_root = Path(__file__).resolve().parents[3]
    output_root = repo_root / cfg.project.output_root
    run_dir = allocate_run_dir(output_root, cfg.experiment.id, run_id)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "controls").mkdir(parents=True, exist_ok=True)
    (run_dir / "probes").mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, run_dir / "config.resolved.yaml", provenance)
    status = StatusFile.create(run_dir, run_id, cfg.experiment.id)
    manifest = RunManifest(run_dir)
    manifest.set_start(c_hash, provenance, cfg.effective_seeds())

    try:
        manifest.update_dataset_info(
            n_samples=len(df),
            split_hash=split_hash,
            split_summary={
                s: int((df["split"] == s).sum()) for s in df["split"].unique()
            },
            dataset_type=cfg.dataset.type,
            anti_leakage_design="counterfactual twins share premise/question-word/entities",
            apply_transforms=bool(cfg.dataset.apply_transforms),
        )
        status.update(progress={"stage": "dataset", "n_samples": len(df)})

        df.to_parquet(run_dir / "samples.parquet", index=False)

        # Confirmation rows are excluded from EVERYTHING below.
        discovery_df = discovery_view(df)          # no confirmation rows
        extracted_ids = set(discovery_df["sample_id"])
        label_of = dict(zip(df["sample_id"], df["target_label"].astype(int)))

        # -- model + extraction -------------------------------------------
        status.update(progress={"stage": "model_load"})
        adapter = load_adapter(cfg)
        layers = expand_layer_plan(cfg, adapter.num_layers)
        manifest.update_model_info(
            id=cfg.model.id,
            revision=None,
            dtype=cfg.model.dtype,
            num_layers=adapter.num_layers,
            hidden_size=adapter.hidden_size,
            notes={"revision": "not pinned; resolution deferred to hub metadata"},
        )

        cache_root = repo_root / cfg.project.cache_root
        cache_dir = cache_root / "activations" / f"{cfg.experiment.id}_{c_hash[:8]}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        status.update(progress={"stage": "extraction"})
        samples_extract = [s for s in samples if s.sample_id in extracted_ids]
        index_df, info = run_extraction_stage(
            cfg, adapter, samples_extract, cache_dir,
            split_of=lambda sid: str(df.loc[df["sample_id"] == sid, "split"].iloc[0]),
        )
        index_df.to_parquet(run_dir / "activation_index.parquet", index=False)
        status.update(progress={"stage": "extraction_done", **{
            k: v for k, v in info.items() if isinstance(v, (int, float, str))
        }})

        # -- probing per (selector, layer) --------------------------------
        status.update(progress={"stage": "probing"})
        loader = ShardedMatrixLoader(cache_dir, index_df)
        selectors = list(cfg.representation.token_selectors)
        c_grid = [float(c) for c in cfg.probe.C_grid]
        probe_seed = int(cfg.reproducibility.probe_seed)

        rows_probe, best_fits = [], {}
        for selector in selectors:
            for layer in layers:
                mats = design_matrices_for(
                    loader, index_df, df.reset_index(drop=True),
                    selector, layer, lambda sid: int(label_of[sid]),
                )
                need = ("train", "validation", "discovery_test")
                if any(s not in mats or len(mats[s]["y"]) == 0 for s in need):
                    logger.warning("skipping %s L%d: incomplete splits", selector, layer)
                    continue
                fit = fit_probe(
                    mats["train"]["X"], mats["train"]["y"],
                    mats["validation"]["X"], mats["validation"]["y"],
                    c_grid=c_grid, seed=probe_seed,
                    class_weight=cfg.probe.class_weight,
                    standardize=bool(cfg.probe.standardize),
                )
                Xte, yte = mats["discovery_test"]["X"], mats["discovery_test"]["y"]
                scores_te = None
                Xt = transform_features(fit, Xte)
                scores_te = fit["classifier"].decision_function(Xt)
                metrics = classification_metrics(yte, scores_te)
                ci = bootstrap_ci(
                    yte, scores_te,
                    metric_fn=_auroc_fn,
                    n_bootstraps=int(cfg.statistics.bootstrap_samples),
                    confidence_level=float(cfg.statistics.confidence_level),
                    seed=int(cfg.reproducibility.bootstrap_seed) + layer,
                )
                cb_eval = class_balance(yte)
                rows_probe.append({
                    "token_selector": selector,
                    "layer": layer,
                    **metrics,
                    **ci,
                    "chosen_C": fit["chosen_C"],
                    "class_balance_n": cb_eval["n"],
                    "class_balance_frac_positive": cb_eval["frac_positive"],
                    "majority_accuracy_discovery_test":
                        majority_baseline_accuracy(yte),
                    "n_train": len(mats["train"]["y"]),
                    "n_validation": len(mats["validation"]["y"]),
                })
                if cfg.outputs.save_coefficients:
                    save_probe(fit, run_dir / "probes" / f"{selector}_layer{layer:02d}.npz")
                best_fits[(selector, layer)] = fit

        metrics_df = pd.DataFrame(rows_probe)
        save_table(metrics_df, run_dir / "probe_metrics.parquet")
        save_json(metrics_df.to_dict(orient="records"), run_dir / "probe_metrics.json")

        # -- controls ------------------------------------------------------
        status.update(progress={"stage": "controls"})
        rl_seeds = list(cfg.controls.random_label_seeds) or RANDOM_LABEL_SEEDS_DEFAULT
        rows_rl = []
        for selector in selectors:
            for layer in layers:
                mats = design_matrices_for(
                    loader, index_df, df.reset_index(drop=True),
                    selector, layer, lambda sid: int(label_of[sid]),
                )
                if not all(k in mats for k in ("train", "validation", "discovery_test")):
                    continue
                for rl_seed in rl_seeds:
                    y_train_shuffled = shuffle_labels(mats["train"]["y"], seed=rl_seed)
                    try:
                        fit = fit_probe(
                            mats["train"]["X"], y_train_shuffled,
                            mats["validation"]["X"], mats["validation"]["y"],
                            c_grid=c_grid, seed=int(cfg.reproducibility.control_seed),
                            class_weight=cfg.probe.class_weight,
                            standardize=bool(cfg.probe.standardize),
                        )
                    except ValueError as exc:
                        logger.warning("random-label control failed %s L%d s%d: %s",
                                       selector, layer, rl_seed, exc)
                        continue
                    Xte = transform_features(fit, mats["discovery_test"]["X"])
                    scores = fit["classifier"].decision_function(Xte)
                    metrics = classification_metrics(
                        mats["discovery_test"]["y"], scores
                    )
                    rows_rl.append({
                        "token_selector": selector,
                        "layer": layer,
                        "control": "random_labels",
                        "seed": rl_seed,
                        **metrics,
                    })
        rl_df = pd.DataFrame(rows_rl)
        save_table(rl_df, run_dir / "controls" / "random_label_metrics.parquet")

        # Text (surface) baseline: same splits, TF-IDF over raw prompt text.
        text_metrics = {}
        if cfg.controls.text_baseline:
            texts_by_split, y_by_split = {}, {}
            for split_name, group in discovery_df.groupby("split"):
                texts_by_split[split_name] = group["prompt"].tolist()
                y_by_split[split_name] = group["target_label"].astype(int).tolist()
            if all(s in texts_by_split for s in ("train", "validation", "discovery_test")):
                text_metrics = text_baseline_metrics(
                    texts_by_split,
                    y_by_split,
                    c_grid=c_grid,
                    seed=int(cfg.reproducibility.control_seed),
                    class_weight=cfg.probe.class_weight,
                )
                save_json(text_metrics, run_dir / "controls" / "text_baseline_metrics.json")

        # Optional random-feature control.
        rand_feature_metrics = {}
        if cfg.controls.random_features:
            hidden_dim = adapter.hidden_size
            base_seed = int(cfg.reproducibility.control_seed)
            for selector in selectors:
                layer0 = layers[len(layers) // 2]
                mats = design_matrices_for(
                    loader, index_df, df.reset_index(drop=True),
                    selector, layer0, lambda sid: int(label_of[sid]),
                )
                if not all(k in mats for k in ("train", "validation", "discovery_test")):
                    continue
                Xr = random_feature_baseline(len(df[df.split != "confirmation"]), hidden_dim, base_seed)
                split_lookup = dict(zip(index_df[index_df.token_selector == selector].sample_id,
                                        index_df[index_df.token_selector == selector].split))
                idx_of = {sid: i for i, sid in enumerate(sorted(split_lookup))}
                mats_r = {}
                for sname in ("train", "validation", "discovery_test"):
                    ids = [sid for sid, sp in split_lookup.items() if sp == sname]
                    sel_idx = [idx_of[sid] for sid in ids]
                    ys = np.asarray([int(label_of[sid]) for sid in ids])
                    # order-aligned
                    m_sorted = sorted(range(len(ids)), key=lambda j: ids[j])
                    mats_r[sname] = {
                        "X": Xr[[sel_idx[j] for j in range(len(ids))]],
                        "y": ys[[m_sorted[j] for j in range(len(ids))]] if False else ys,
                    }
                try:
                    fit = fit_probe(
                        mats_r["train"]["X"], mats_r["train"]["y"],
                        mats_r["validation"]["X"], mats_r["validation"]["y"],
                        c_grid=c_grid, seed=base_seed + 99,
                    )
                except ValueError as exc:
                    logger.warning("random-feature baseline skipped: %s", exc)
                    continue
                metrics = evaluate_probe(fit, mats_r["discovery_test"]["X"],
                                         mats_r["discovery_test"]["y"])
                rand_feature_metrics[selector] = metrics
            save_json(rand_feature_metrics,
                      run_dir / "controls" / "random_feature_metrics.json")

        # -- figures --------------------------------------------------------
        status.update(progress={"stage": "figures"})
        for sel in selectors:
            plot_d_by_layer(
                metrics_df[metrics_df["token_selector"] == sel],
                run_dir / "figures" / f"D_by_layer_{sel}.png",
                title=f"E00 decodability by layer — {sel}",
            )
        plot_d_combined(metrics_df, rl_df,
                        run_dir / "figures" / "D_by_layer_combined.png")

        # -- discovery summary ----------------------------------------------
        summary_text = _render_summary(
            cfg=cfg, metrics_df=metrics_df, rl_df=rl_df,
            text_metrics=text_metrics, rand_feature_metrics=rand_feature_metrics,
            df=df, selectors=selectors,
        )
        (run_dir / "DISCOVERY_SUMMARY.md").write_text(summary_text, encoding="utf-8")

        runs_summary = []
        for sel in selectors:
            sub = metrics_df[metrics_df["token_selector"] == sel]
            if len(sub):
                best = sub.loc[sub["auroc"].idxmax()]
                runs_summary.append({
                    "selector": sel,
                    "best_discovery_layer": int(best["layer"]),
                    "auroc": float(best["auroc"]),
                    "auprc": float(best["auprc"]),
                })
        manifest.finish(runs_summary=runs_summary)
        status.complete(message="E00 pipeline complete")
        return run_dir
    except Exception as exc:
        logger.exception("E00 run failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise

def _auroc_fn(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _render_summary(
    *,
    cfg,
    metrics_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    text_metrics: dict,
    rand_feature_metrics: dict,
    df: pd.DataFrame,
    selectors: list[str],
) -> str:
    """Descriptive discovery summary. Contains no causal claims."""
    lines: list[str] = []
    n_conf = int((df["split"] == "confirmation").sum())
    lines.append("# E00 — Representation Cartography (discovery summary)")
    lines.append("")
    lines.append(
        "**Scope**: descriptive evidence about linear decodability only. "
        "Nothing here establishes that the model *uses* the target variable; "
        "causal testing is deferred to E01."
    )
    lines.append("")
    for sel in selectors:
        sub = metrics_df[metrics_df["token_selector"] == sel]
        if not len(sub):
            continue
        best = sub.loc[sub["auroc"].idxmax()]
        lines.append(f"## Token selector: `{sel}`")
        lines.append("")
        ba = best.get("balanced_accuracy")
        lines.append(
            f"- Best discovery-test layer: **{int(best['layer'])}** (0-indexed), "
            f"AUROC = {float(best['auroc']):.3f} "
            f"[{float(best['ci_low']):.3f}, {float(best['ci_high']):.3f}] "
            f"(95% bootstrap CI), AUPRC = {float(best['auprc']):.3f}, "
            f"balanced accuracy = {float(ba):.3f}"
        )
        rl_sub = rl_df[rl_df["token_selector"] == sel] if len(rl_df) else None
        if rl_sub is not None and len(rl_sub):
            lines.append(
                f"- Random-label control on same splits: AUROC mean = "
                f"{float(rl_sub['auroc'].mean()):.3f} ± "
                f"{float(rl_sub['auroc'].std()):.3f} (range "
                f"{float(rl_sub['auroc'].min()):.3f}–"
                f"{float(rl_sub['auroc'].max()):.3f}) over {len(rl_sub)} fits"
            )
        text_auroc = text_metrics.get("auroc") if isinstance(text_metrics, dict) else None
        if text_auroc is not None:
            lines.append(
                f"- Text-only baseline (TF-IDF + logistic regression): "
                f"discovery AUROC = {float(text_auroc):.3f}"
            )
        rfm = (rand_feature_metrics or {}).get(sel)
        if rfm and rfm.get("auroc") is not None:
            lines.append(
                f"- Random-feature control (mid-layer): AUROC = {float(rfm['auroc']):.3f}"
            )
        prof = sub[["layer", "auroc", "auprc", "chosen_C"]].copy().round(4)
        lines.append("")
        lines.append("### Layer profile")
        lines.append("")
        lines.append(markdown_table(prof))
        top3 = ", ".join(
            str(int(x)) for x in sub.sort_values("auroc", ascending=False)["layer"].head(3)
        )
        lines.append("")
        lines.append(f"- Top-3 layers by AUROC: {top3}")
        lines.append("")
    bal = check_label_balance(df[df["split"] != "confirmation"]["target_label"])
    lines.append("## Dataset and isolation")
    lines.append("")
    lines.append(
        f"- n_samples = {len(df)}; class balance over non-confirmation rows: "
        f"{bal}"
    )
    lines.append(
        f"- The confirmation split ({n_conf} examples) was NOT extracted, probed, "
        f"or read during this exploratory scan."
    )
    if len(metrics_df):
        lines.append(
            "- Single-seed run: stability across independent seeds must be judged "
            "from the per-seed random-label variations and future repeat runs."
        )
    anomalies: list[str] = []
    if len(metrics_df):
        max_a = float(metrics_df["auroc"].max())
        if max_a >= 1.0 - 1e-9:
            anomalies.append(
                "AUROC reached exactly 1.0 — verify no lexical leakage or "
                "construction artifact before trusting the profile."
            )
        if len(rl_df) and float(rl_df["auroc"].mean()) > 0.6:
            anomalies.append(
                "random-label control exceeded AUROC 0.6 on average — protocol "
                "problem suspected."
            )
        if (metrics_df["auroc"] < 0.35).any():
            anomalies.append(
                "some layers score below 0.35 AUROC — check probe fitting robustness."
            )
    if anomalies:
        lines.append("")
        lines.append("## Anomalies flagged for inspection")
        lines.extend(f"- {a}" for a in anomalies)
    lines.append("")
    return "\n".join(lines)





