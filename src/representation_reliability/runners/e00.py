"""E00 — Representation Cartography.

Exploratory decodability experiment:
"At what layers and token positions is a controlled latent variable decodable
from the model's hidden states, and how stable is that decodability profile?"

This runner establishes DECODABILITY ONLY (D). Nothing here supports claims
that the model *uses* the variable — that requires causal interventions (E01).

Phase 0A.1 integrity contract:
- design identity is ``site × layer × token_selector`` with exact
  sample-identity validation against expected split memberships;
- random-label controls randomize BOTH training and validation labels and are
  only evaluated against the real, untouched discovery-test labels;
- confirmation-split labels are never constructed during discovery (guard
  raises); confirmation rows carry no labels in discovery artifacts;
- extraction caches are schema-v2: semantic identity manifest + truthful
  work-unit ranges per shard; mismatch refuses loudly.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.base import check_label_balance, samples_to_dataframe
from ..data.splits import (
    ConfirmationSplitAccessError,
    apply_splits,
    assign_group_splits,
    build_discovery_label_map,
    discovery_view,
    validate_splits,
)
from ..data.synthetic import generate_synthetic_relations
from ..extraction.activations import build_cache_identity
from ..extraction.cache import LegacyCacheError  # noqa: F401 (re-export)
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
    randomized_control_labels,
    save_probe,
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
LEARNING_SPLITS = ("train", "validation", "discovery_test")


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


def _strip_confirmation_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Discovery artifact view: holdout rows carry NO label information."""
    out = df.copy()
    conf = out["split"] == "confirmation"
    for col in ("target_label", "truth_label"):
        if col in out.columns:
            out.loc[conf, col] = pd.NA
    return out


def _confirmation_ids_digest(df: pd.DataFrame) -> str:
    ids = sorted(df.loc[df["split"] == "confirmation", "sample_id"].astype(str))
    return hashlib.sha256("\x1e".join(ids).encode("utf-8")).hexdigest()


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
        model_revision=cfg.model.revision or "unpinned",
        dataset_split_hash=split_hash,
    )
    repo_root = Path(__file__).resolve().parents[3]
    output_root = repo_root / cfg.project.output_root
    run_dir = allocate_run_dir(output_root, cfg.experiment.id, run_id)
    for sub in ("figures", "logs", "controls", "probes"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, run_dir / "config.resolved.yaml", provenance)
    status = StatusFile.create(run_dir, run_id, cfg.experiment.id)
    manifest = RunManifest(run_dir)
    manifest.set_start(c_hash, provenance, cfg.effective_seeds())

    try:
        # Confirmation labels are never constructed during discovery.
        discovery_df = discovery_view(df)
        label_of = build_discovery_label_map(discovery_df)
        grouped_ids = {
            s: g["sample_id"].tolist() for s, g in discovery_df.groupby("split")
        }
        expected_ids_by_split = {k: grouped_ids[k] for k in sorted(grouped_ids)}

        df_public = _strip_confirmation_labels(df)
        manifest.update_dataset_info(
            n_samples=len(df),
            split_hash=split_hash,
            split_summary={
                s: int((df["split"] == s).sum()) for s in df["split"].unique()
            },
            dataset_type=cfg.dataset.type,
            anti_leakage_design="counterfactual twins share premise/question-word/entities",
            apply_transforms=bool(cfg.dataset.apply_transforms),
            confirmation_labels_observed=False,
            confirmation_sample_ids_sha256=_confirmation_ids_digest(df),
        )
        status.update(progress={"stage": "dataset", "n_samples": len(df)})
        # Discovery artifact: holdout rows carry no label information.
        df_public.to_parquet(run_dir / "samples.parquet", index=False)

        # -- model + extraction -------------------------------------------
        status.update(progress={"stage": "model_load"})
        adapter = load_adapter(cfg)
        layers = expand_layer_plan(cfg, adapter.num_layers)
        sites = list(cfg.representation.sites)
        selectors = list(cfg.representation.token_selectors)
        revisions = adapter.resolved_revisions()
        manifest.update_model_info(
            id=cfg.model.id,
            revision=cfg.model.revision,
            dtype=cfg.model.dtype,
            num_layers=adapter.num_layers,
            hidden_size=adapter.hidden_size,
            resolved_revision=revisions.get("model_sha"),
            tokenizer_id=cfg.model.id,
            tokenizer_revision=revisions.get("tokenizer_sha"),
            notes={"revision_resolution": revisions},
        )

        cache_root = repo_root / cfg.project.cache_root
        cache_dir = (
            cache_root / "activations"
            / f"{cfg.experiment.id}_v2_{c_hash[:8]}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        status.update(progress={"stage": "extraction"})
        extracted_ids = set(discovery_df["sample_id"])
        samples_extract = [s for s in samples if s.sample_id in extracted_ids]
        split_map = dict(zip(discovery_df["sample_id"], discovery_df["split"]))

        def _split_of(sid: str) -> str:
            if sid not in split_map:
                raise ConfirmationSplitAccessError(
                    f"sample {sid!r} is not part of the discovery view; "
                    "confirmation/unknown samples cannot be extracted here"
                )
            return str(split_map[sid])

        identity = build_cache_identity(
            experiment_id=cfg.experiment.id,
            samples=samples_extract,
            model_id=cfg.model.id,
            model_resolved_revision=revisions.get("model_sha"),
            tokenizer_id=cfg.model.id,
            tokenizer_resolved_revision=revisions.get("tokenizer_sha"),
            sites=sites,
            layers=layers,
            token_selectors=selectors,
            model_dtype=cfg.model.dtype,
            tokenization={"padding_side": "right", "add_special_tokens": True},
        )
        index_df, info = run_extraction_stage(
            cfg, adapter, samples_extract, cache_dir,
            split_of=_split_of, identity=identity, resume=bool(cfg.runtime.resume),
        )
        index_df.to_parquet(run_dir / "activation_index.parquet", index=False)
        status.update(progress={"stage": "extraction_done", **{
            k: v for k, v in info.items() if isinstance(v, (int, float, str))
        }})

        # -- probing per (site, selector, layer) ---------------------------
        status.update(progress={"stage": "probing"})
        loader = ShardedMatrixLoader(cache_dir, index_df)
        c_grid = [float(c) for c in cfg.probe.C_grid]
        probe_seed = int(cfg.reproducibility.probe_seed)

        rows_probe: list[dict] = []
        for site in sites:
            for selector in selectors:
                for layer in layers:
                    mats = design_matrices_for(
                        loader, index_df, selector, layer, site,
                        label_of=lambda sid: int(label_of[sid]),
                        expected_ids_by_split=expected_ids_by_split,
                    )
                    if any(s not in mats for s in LEARNING_SPLITS):
                        logger.warning(
                            "skipping %s/%s L%d: incomplete splits",
                            site, selector, layer,
                        )
                        continue
                    fit = fit_probe(
                        mats["train"]["X"], mats["train"]["y"],
                        mats["validation"]["X"], mats["validation"]["y"],
                        c_grid=c_grid, seed=probe_seed,
                        class_weight=cfg.probe.class_weight,
                        standardize=bool(cfg.probe.standardize),
                    )
                    Xte = mats["discovery_test"]["X"]
                    yte = mats["discovery_test"]["y"]
                    scores_te = fit["classifier"].decision_function(
                        transform_features(fit, Xte)
                    )
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
                        "site": site,
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
                        save_probe(fit, run_dir / "probes"
                                   / f"{site}_{selector}_layer{layer:02d}.npz")

        metrics_df = pd.DataFrame(rows_probe)
        save_table(metrics_df, run_dir / "probe_metrics.parquet")
        save_json(metrics_df.to_dict(orient="records"), run_dir / "probe_metrics.json")

        # -- controls ------------------------------------------------------
        status.update(progress={"stage": "controls"})
        rl_seeds = list(cfg.controls.random_label_seeds) or RANDOM_LABEL_SEEDS_DEFAULT
        rows_rl: list[dict] = []
        for site in sites:
            for selector in selectors:
                for layer in layers:
                    mats = design_matrices_for(
                        loader, index_df, selector, layer, site,
                        label_of=lambda sid: int(label_of[sid]),
                        expected_ids_by_split=expected_ids_by_split,
                    )
                    if any(s not in mats for s in LEARNING_SPLITS):
                        continue
                    for rl_seed in rl_seeds:
                        # Clean null protocol: randomize TRAIN and VALIDATION
                        # labels with independent deterministic permutations;
                        # evaluate against REAL discovery-test labels only.
                        y_tr_r, y_val_r, perm_seeds = randomized_control_labels(
                            mats["train"]["y"],
                            mats["validation"]["y"],
                            seed=int(cfg.reproducibility.control_seed)
                            + 1000 * rl_seed,
                        )
                        try:
                            fit = fit_probe(
                                mats["train"]["X"], y_tr_r,
                                mats["validation"]["X"], y_val_r,
                                c_grid=c_grid,
                                seed=int(cfg.reproducibility.control_seed),
                                class_weight=cfg.probe.class_weight,
                                standardize=bool(cfg.probe.standardize),
                            )
                        except ValueError as exc:
                            logger.warning(
                                "random-label control failed %s/%s L%d s%d: %s",
                                site, selector, layer, rl_seed, exc,
                            )
                            continue
                        Xte = transform_features(fit, mats["discovery_test"]["X"])
                        scores = fit["classifier"].decision_function(Xte)
                        metrics = classification_metrics(
                            mats["discovery_test"]["y"], scores
                        )
                        rows_rl.append({
                            "site": site,
                            "token_selector": selector,
                            "layer": layer,
                            "control": "random_labels",
                            "seed": rl_seed,
                            "train_perm_seed": perm_seeds[0],
                            "validation_perm_seed": perm_seeds[1],
                            **metrics,
                        })
        rl_df = pd.DataFrame(rows_rl)
        save_table(rl_df, run_dir / "controls" / "random_label_metrics.parquet")

        # Text (surface) baseline: same splits, TF-IDF over raw prompt text.
        text_metrics: dict = {}
        if cfg.controls.text_baseline:
            texts_by_split, y_by_split = {}, {}
            for split_name, group in discovery_df.groupby("split"):
                texts_by_split[split_name] = group["prompt"].tolist()
                y_by_split[split_name] = group["target_label"].astype(int).tolist()
            if all(s in texts_by_split for s in LEARNING_SPLITS):
                text_metrics = text_baseline_metrics(
                    texts_by_split, y_by_split,
                    c_grid=c_grid,
                    seed=int(cfg.reproducibility.control_seed),
                    class_weight=cfg.probe.class_weight,
                )
                save_json(text_metrics,
                          run_dir / "controls" / "text_baseline_metrics.json")

        # Optional random-feature control (mid-layer per site/selector).
        rand_feature_metrics: dict = {}
        if cfg.controls.random_features:
            hidden_dim = adapter.hidden_size
            base_seed = int(cfg.reproducibility.control_seed)
            for site in sites:
                for selector in selectors:
                    layer0 = layers[len(layers) // 2]
                    mats = design_matrices_for(
                        loader, index_df, selector, layer0, site,
                        label_of=lambda sid: int(label_of[sid]),
                        expected_ids_by_split=expected_ids_by_split,
                    )
                    if any(s not in mats for s in LEARNING_SPLITS):
                        continue
                    fit_rand, metrics_rand = _random_feature_fit(
                        mats, hidden_dim, c_grid, base_seed
                    )
                    if fit_rand is None:
                        continue
                    rand_feature_metrics[f"{site}/{selector}"] = metrics_rand
            if rand_feature_metrics:
                save_json(rand_feature_metrics,
                          run_dir / "controls" / "random_feature_metrics.json")

        # -- figures & discovery summary ------------------------------------
        status.update(progress={"stage": "figures"})
        for site in sites:
            for sel in selectors:
                plot_d_by_layer(
                    metrics_df[(metrics_df["site"] == site)
                               & (metrics_df["token_selector"] == sel)],
                    run_dir / "figures" / f"D_by_layer_{site}_{sel}.png",
                    title=f"E00 decodability by layer — {site}/{sel}",
                )
            plot_d_combined(
                metrics_df[metrics_df["site"] == site],
                rl_df[rl_df["site"] == site] if len(rl_df) else rl_df,
                run_dir / "figures" / f"D_by_layer_combined_{site}.png",
            )

        summary_text = _render_summary(
            cfg=cfg, metrics_df=metrics_df, rl_df=rl_df,
            text_metrics=text_metrics,
            rand_feature_metrics=rand_feature_metrics,
            df=df, selectors=selectors, sites=sites,
        )
        (run_dir / "DISCOVERY_SUMMARY.md").write_text(summary_text, encoding="utf-8")

        runs_summary = []
        for site in sites:
            for sel in selectors:
                sub = metrics_df[
                    (metrics_df["site"] == site)
                    & (metrics_df["token_selector"] == sel)
                ]
                if len(sub):
                    best = sub.loc[sub["auroc"].idxmax()]
                    runs_summary.append({
                        "site": site,
                        "selector": sel,
                        "best_discovery_layer": int(best["layer"]),
                        "auroc": float(best["auroc"]),
                        "auprc": float(best["auprc"]),
                        "ci_low": float(best["ci_low"]),
                        "ci_high": float(best["ci_high"]),
                    })
        manifest.finish(runs_summary=runs_summary)
        status.complete(message="E00 pipeline complete (schema-v2 cache)")
        return run_dir
    except Exception as exc:
        logger.exception("E00 run failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise

def _random_feature_fit(mats, hidden_dim, c_grid, base_seed):
    """Optional random-Gaussian-feature control with the standard protocol."""
    mats_r: dict[str, dict] = {}
    for i, sname in enumerate(LEARNING_SPLITS):
        n = len(mats[sname]["y"])
        Xr = random_feature_baseline(n, hidden_dim, seed=base_seed + 99 + i)
        mats_r[sname] = {"X": Xr, "y": mats[sname]["y"]}
    try:
        fit = fit_probe(
            mats_r["train"]["X"], mats_r["train"]["y"],
            mats_r["validation"]["X"], mats_r["validation"]["y"],
            c_grid=c_grid, seed=base_seed + 99,
        )
    except ValueError as exc:
        logger.warning("random-feature baseline skipped: %s", exc)
        return None, None
    metrics = evaluate_probe(fit, mats_r["discovery_test"]["X"],
                             mats_r["discovery_test"]["y"])
    return fit, metrics


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
    sites: list[str],
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
    lines.append(
        "Integrity: cache schema v2 (semantic identity manifest), exact "
        "sample-identity-validated design matrices, randomized train AND "
        "validation labels for the random-label control. "
        "Confirmation labels were never observed in this run."
    )
    lines.append("")
    for site in sites:
        for sel in selectors:
            sub = metrics_df[
                (metrics_df["site"] == site)
                & (metrics_df["token_selector"] == sel)
            ]
            if not len(sub):
                continue
            best = sub.loc[sub["auroc"].idxmax()]
            lines.append(f"## Site `{site}` · selector `{sel}`")
            lines.append("")
            ba = best.get("balanced_accuracy")
            lines.append(
                f"- Best discovery-test layer: **{int(best['layer'])}** "
                f"(0-indexed), AUROC = {float(best['auroc']):.3f} "
                f"[{float(best['ci_low']):.3f}, {float(best['ci_high']):.3f}] "
                f"(95% bootstrap CI), AUPRC = {float(best['auprc']):.3f}, "
                f"balanced accuracy = {float(ba):.3f}"
            )
            if len(rl_df):
                rl_sub = rl_df[(rl_df["site"] == site)
                               & (rl_df["token_selector"] == sel)]
                lines.append(
                    f"- Random-label control (randomized train+validation "
                    f"labels, real test labels): AUROC mean = "
                    f"{float(rl_sub['auroc'].mean()):.3f} ± "
                    f"{float(rl_sub['auroc'].std()):.3f} over {len(rl_sub)} fits"
                )
        text_auroc = text_metrics.get("auroc") if isinstance(text_metrics, dict) else None
        if text_auroc is not None:
            lines.append(
                f"- Text-only baseline (TF-IDF + logistic regression, same "
                f"splits/protocol): discovery AUROC = {float(text_auroc):.3f}"
            )
            break  # text baseline is site-independent; print once
    prof_cols = ["site", "token_selector", "layer", "auroc", "auprc", "chosen_C"]
    prof = metrics_df[prof_cols].copy().round(4) if len(metrics_df) else metrics_df
    lines.append("")
    lines.append("### Layer profile")
    lines.append("")
    lines.append(markdown_table(prof))
    bal = check_label_balance(df[df["split"] != "confirmation"]["target_label"])
    lines.append("")
    lines.append("## Dataset and isolation")
    lines.append("")
    lines.append(
        f"- n_samples = {len(df)}; class balance over non-confirmation rows: {bal}"
    )
    lines.append(
        f"- The confirmation split ({n_conf} examples) was NOT extracted, probed, "
        f"or read during this exploratory scan; its rows carry no labels in "
        f"`samples.parquet` and only an ID digest was recorded."
    )
    anomalies: list[str] = []
    if len(metrics_df):
        max_a = float(metrics_df["auroc"].max())
        if max_a >= 1.0 - 1e-9:
            anomalies.append(
                "AUROC reached exactly 1.0 — verify no leakage or construction "
                "artifact before trusting the profile."
            )
        if len(rl_df) and float(rl_df["auroc"].mean()) > 0.6:
            anomalies.append(
                "random-label control exceeded AUROC 0.6 on average — protocol "
                "problem suspected."
            )
        if (metrics_df["auroc"] < 0.35).any():
            anomalies.append(
                "some cells score below 0.35 AUROC — check probe fitting robustness."
            )
    if anomalies:
        lines.append("")
        lines.append("## Anomalies flagged for inspection")
        lines.extend(f"- {a}" for a in anomalies)
    lines.append("")
    return "\n".join(lines)








