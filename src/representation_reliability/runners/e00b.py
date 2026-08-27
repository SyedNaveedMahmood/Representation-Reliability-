"""E00-B — Behavioral Floor / Output Readout on the SAME synthetic task.

Forced-choice conditional-likelihood readout: for prompts ending in
``Answer:``, compare ``log P(continuation | prompt)`` between the two answer
candidates and decide by argmax TOTAL log-probability (mean log-prob reported
as a sensitivity check).

This measures behavioral competence B on exactly the E00 dataset. It makes no
representational claims; B is later compared with D from corrected E00.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.splits import build_discovery_label_map, discovery_view
from ..metrics.decoding import (
    class_balance,
    classification_metrics,
    majority_baseline_accuracy,
)
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .e00 import _generate_dataset, _strip_confirmation_labels  # noqa: F401
from .extract import load_adapter

logger = logging.getLogger(__name__)


def _bootstrap_ci_binary(correct: np.ndarray, n_boot: int, level: float,
                         seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(correct)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = float(np.mean(correct[idx]))
        if np.isfinite(val):
            stats.append(val)
    alpha = 1 - level
    lo, hi = (np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if stats else (float("nan"), float("nan")))
    return {"ci_low": float(lo), "ci_high": float(hi),
            "confidence_level": level}

def run_e00b(
    base_path=None, model_path=None, experiment_path=None,
    overrides: tuple[str, ...] = (),
) -> Path:
    cfg, provenance = resolve_config(
        base_path=base_path, model_path=model_path,
        experiment_path=experiment_path, overrides=overrides,
    )
    c_hash = config_hash(cfg)
    samples, df = _generate_dataset(cfg)
    split_hash = dataset_split_hash(dict(zip(df["sample_id"], df["split"])))

    run_id = make_run_id(
        experiment_id=cfg.experiment.id, config_hash=c_hash,
        seed=int(cfg.reproducibility.seed),
        model_revision=cfg.model.revision or "unpinned",
        dataset_split_hash=split_hash,
    )
    repo_root = Path(__file__).resolve().parents[3]
    run_dir = allocate_run_dir(repo_root / cfg.project.output_root,
                               cfg.experiment.id, run_id)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    save_resolved_config(cfg, run_dir / "config.resolved.yaml", provenance)
    status = StatusFile.create(run_dir, run_id, cfg.experiment.id)
    manifest = RunManifest(run_dir)
    manifest.set_start(c_hash, provenance, cfg.effective_seeds())

    try:
        discovery_df = discovery_view(df).reset_index(drop=True)
        label_of = build_discovery_label_map(discovery_df)

        adapter = load_adapter(cfg)
        revisions = adapter.resolved_revisions()
        manifest.update_model_info(
            id=cfg.model.id, revision=cfg.model.revision,
            dtype=cfg.model.dtype,
            resolved_revision=revisions.get("model_sha"),
            tokenizer_revision=revisions.get("tokenizer_sha"),
            notes={"revision_resolution": revisions},
        )

        yes_text, no_text = cfg.behavior.candidates_primary
        sec_yes, sec_no = cfg.behavior.candidates_secondary
        gold = np.asarray(
            [int(label_of[s]) for s in discovery_df["sample_id"]], dtype=int
        )
        prompts = discovery_df["prompt"].tolist()

        records: list[dict] = []
        bs = max(4, int(cfg.runtime.batch_size) * 4)
        tok_yes = tok_no = None
        try:
            from tqdm import tqdm

            iterator = tqdm(range(0, len(prompts), bs), desc="behavior",
                            unit="batch", dynamic_ncols=True)
        except ImportError:  # pragma: no cover
            iterator = range(0, len(prompts), bs)
        for start in iterator:
            chunk = prompts[start : start + bs]
            primary = adapter.score_continuations(chunk, [yes_text, no_text])
            secondary = adapter.score_continuations(chunk, [sec_yes, sec_no])
            for i in range(len(chunk)):
                p, s = primary[i], secondary[i]
                lp_yes_t, lp_no_t = p[0]["logp_total"], p[1]["logp_total"]
                lp_yes_m, lp_no_m = p[0]["logp_mean"], p[1]["logp_mean"]
                margin_t = lp_yes_t - lp_no_t
                margin_m = lp_yes_m - lp_no_m
                if tok_yes is None:
                    tok_yes = list(p[0]["token_ids"])
                    tok_no = list(p[1]["token_ids"])
                pred = int(margin_t >= 0)          # 1 -> Yes wins
                row_i = start + i
                meta_row = discovery_df.iloc[row_i]
                records.append({
                    "sample_id": str(meta_row["sample_id"]),
                    "split": str(meta_row["split"]),
                    "gold_label": int(gold[row_i]),
                    "candidate_yes_text": yes_text.strip(),
                    "candidate_no_text": no_text.strip(),
                    "candidate_yes_token_ids": tok_yes,
                    "candidate_no_token_ids": tok_no,
                    "logp_yes_total": lp_yes_t,
                    "logp_no_total": lp_no_t,
                    "logp_yes_mean": lp_yes_m,
                    "logp_no_mean": lp_no_m,
                    "margin_total": margin_t,
                    "margin_mean": margin_m,
                    "forced_choice_prediction": pred,
                    "forced_choice_correct": int(pred == int(gold[row_i])),
                    "relation_family": meta_row["relation"],
                    "queried_word": meta_row["queried_word"],
                    "queried_side": meta_row["queried_side"],
                    "template_id": meta_row["template_id"],
                    "secondary_margin_total_agrees":
                        int((int(s[0]["logp_total"] - s[1]["logp_total"] >= 0)) == pred),
                })

        pred_df = pd.DataFrame(records)
        correct = pred_df["forced_choice_correct"].to_numpy(dtype=float)
        margins = pred_df["margin_total"].to_numpy(dtype=float)

        from sklearn.metrics import balanced_accuracy_score

        acc = float(correct.mean())
        ci = _bootstrap_ci_binary(
            correct, n_boot=int(cfg.statistics.bootstrap_samples),
            level=float(cfg.statistics.confidence_level),
            seed=int(cfg.reproducibility.bootstrap_seed),
        )
        bal_acc = float(balanced_accuracy_score(
            gold, pred_df["forced_choice_prediction"].to_numpy()
        ))
        correct_mask = pred_df["forced_choice_correct"] == 1
        behavior_metrics = {
            "primary_decision_rule": "argmax total conditional log-prob",
            "candidates_primary": [yes_text, no_text],
            "n_evaluated": len(pred_df),
            "class_balance": class_balance(gold),
            "majority_accuracy_discovery_only":
                majority_baseline_accuracy(gold),
            "forced_choice_accuracy": acc,
            "forced_choice_accuracy_ci_95pct": ci,
            "balanced_accuracy": bal_acc,
            "mean_correct_answer_margin":
                float(pred_df.loc[correct_mask, "margin_total"].mean()),
            "mean_incorrect_answer_margin":
                float(pred_df.loc[~correct_mask.astype(bool), "margin_total"].mean())
                if (~correct_mask.astype(bool)).any() else None,
            "secondary_candidates_basis_agreement_rate":
                float(pred_df["secondary_margin_total_agrees"].mean()),
            "margin_vs_gold": classification_metrics(gold, margins),
            "by_split": {
                s: {"n": len(g),
                    "forced_choice_accuracy":
                        float(g["forced_choice_correct"].mean())}
                for s, g in pred_df.groupby("split")
            },
            "per_template_accuracy": {
                t: float(g["forced_choice_correct"].mean())
                for t, g in pred_df.groupby("template_id")
            },
            "secondary_candidates": [sec_yes, sec_no],
        }
        save_json(behavior_metrics, run_dir / "behavior_metrics.json")

        by_rel = (
            pred_df.groupby("relation_family")["forced_choice_correct"]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "forced_choice_accuracy", "count": "n"})
        )
        by_rel = by_rel.sort_values("forced_choice_accuracy", ascending=False)
        save_table(by_rel, run_dir / "behavior_metrics_by_relation.parquet")

        if cfg.behavior.free_generation_diagnostic_n > 0:
            n_diag = min(int(cfg.behavior.free_generation_diagnostic_n),
                         len(prompts))
            gen = adapter.generate(prompts[:n_diag], max_new_tokens=8,
                                   do_sample=False)
            diag_df = pd.DataFrame({
                "sample_id": discovery_df["sample_id"].iloc[:n_diag].values,
                "prompt": prompts[:n_diag],
                "free_generation_text": [g["text"] for g in gen],
            })
            diag_df.to_parquet(run_dir / "behavior_freegen_sample.parquet",
                               index=False)

        manifest.finish(runs_summary=[{
            "forced_choice_accuracy": acc,
            "balanced_accuracy": bal_acc,
            "margin_auroc": behavior_metrics["margin_vs_gold"]["auroc"],
        }])
        status.complete(message="E00-B behavioral readout complete")
        return run_dir
    except Exception as exc:
        logger.exception("E00-B run failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise



