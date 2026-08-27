"""Probe runner: builds design matrices from the cache, fits probes + controls.

Integrity contracts enforced here:

- Design-matrix identity is the full triple ``site × layer × token_selector``.
- Given cache metadata rows in order R, activation row ``i`` corresponds
  exactly to metadata row ``i`` (delegated to ActivationCacheReader).
- Exact sample-identity assertions: duplicates, missing samples, unexpected
  samples, and X/metadata length mismatches all raise loudly.
- Every shard touched by the probe path is SHA-256 verified (once per
  process) via the shared reader.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..extraction.cache import ActivationCacheReader

logger = logging.getLogger(__name__)


class ShardedMatrixLoader:
    """Serves arbitrary metadata-row subsets with verified, ordered tensors."""

    def __init__(self, cache_dir, index_df: pd.DataFrame) -> None:
        self.reader = ActivationCacheReader(cache_dir)
        self.index_df = index_df.reset_index(drop=True)

    def rows_for(self, mask: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
        """Return ``(X, meta)`` where X row i is exactly meta row i."""
        meta = self.index_df[mask].reset_index(drop=True)
        X = self.reader.load_rows(meta)
        return np.asarray(X, dtype=np.float64), meta


def design_matrices_for(
    loader: ShardedMatrixLoader,
    index_df: pd.DataFrame,
    selector: str,
    layer: int,
    site: str,
    label_of: Callable[[str], int],
    expected_ids_by_split: dict[str, list[str]] | None = None,
    split_of: Callable[[str], str] | None = None,
) -> dict[str, dict]:
    """Assemble ``{split: {'X','y','sample_ids'}}`` for one design cell.

    The design identity is ``(site, layer, token_selector)``.  When
    ``expected_ids_by_split`` is provided, the cached content is validated
    against it exactly: no missing samples, no unexpected samples, no
    duplicates; otherwise raises :class:`RuntimeError` loudly.
    """
    base_mask = (
        (index_df["token_selector"] == selector)
        & (index_df["layer"] == layer)
        & (index_df["site"] == site)
    )
    sub = index_df[base_mask]

    split_names = set(sub["split"].unique().tolist())
    if expected_ids_by_split is not None:
        missing_splits = sorted(set(expected_ids_by_split) - split_names)
        if missing_splits:
            raise RuntimeError(
                f"design {site}/{layer}/{selector}: cached splits missing: "
                f"{missing_splits}"
            )
        extra_splits = sorted(split_names - set(expected_ids_by_split))
        if extra_splits:
            raise RuntimeError(
                f"design {site}/{layer}/{selector}: unexpected cached splits: "
                f"{extra_splits}"
            )

    out: dict[str, dict] = {}
    for split_name in sorted(split_names):
        split_mask = base_mask & (index_df["split"] == split_name)
        X, meta = loader.rows_for(split_mask)
        ids = meta["sample_id"].tolist()

        dupes = {s for s in ids if ids.count(s) > 1}
        if dupes:
            raise RuntimeError(
                f"design {site}/{layer}/{selector} split {split_name}: "
                f"duplicate sample rows {sorted(dupes)[:5]}"
            )
        if len(meta) != X.shape[0]:
            raise RuntimeError(   # defensive; reader contract guarantees this
                "X/metadata length mismatch"
            )
        if expected_ids_by_split is not None:
            expected = list(expected_ids_by_split[split_name])
            if set(ids) != set(expected):
                missing = sorted(set(expected) - set(ids))
                unexpected = sorted(set(ids) - set(expected))
                raise RuntimeError(
                    f"design {site}/{layer}/{selector} split {split_name} "
                    f"sample mismatch: missing={missing[:5]} "
                    f"unexpected={unexpected[:5]}"
                )

        y = np.asarray([label_of(sid) for sid in ids], dtype=int)
        out[split_name] = {"X": X, "y": y, "sample_ids": ids}
    return out


def text_baseline_metrics(
    texts_by_split: dict[str, list[str]],
    y_by_split: dict[str, list[int]],
    c_grid,
    seed: int,
    class_weight: str | None = "balanced",
) -> dict:
    """TF-IDF + logistic regression surface baseline, same protocol as probes."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Xtr_txt = texts_by_split["train"]
    Xval_txt = texts_by_split["validation"]
    Xte_txt = texts_by_split["discovery_test"]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Ztr = vec.fit_transform(Xtr_txt)
    Zval = vec.transform(Xval_txt)
    Zte = vec.transform(Xte_txt)

    ytr = np.asarray(y_by_split["train"])
    yval = np.asarray(y_by_split["validation"])
    yte = np.asarray(y_by_split["discovery_test"])

    best_c, best_auroc = None, -np.inf
    by_c = {}
    for c in sorted(float(x) for x in c_grid):
        clf = LogisticRegression(C=c, class_weight=class_weight,
                                 max_iter=2000, solver="lbfgs", random_state=seed)
        clf.fit(Ztr, ytr)
        auroc = float(roc_auc_score(yval, clf.decision_function(Zval)))
        by_c[str(c)] = auroc
        if auroc > best_auroc:
            best_c, best_auroc = c, auroc

    final = LogisticRegression(C=float(best_c), class_weight=class_weight,
                               max_iter=2000, solver="lbfgs", random_state=seed)
    final.fit(Ztr, ytr)
    test_scores = final.decision_function(Zte)
    from ..metrics.decoding import classification_metrics

    metrics = classification_metrics(yte, test_scores)
    metrics["chosen_C"] = float(best_c)
    metrics["validation_auroc_by_C"] = by_c
    metrics["baseline_type"] = "tfidf_logreg"
    return metrics
