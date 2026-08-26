"""Probe runner: builds design matrices from the cache, fits probes + controls."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..extraction.cache import ActivationCacheReader, shard_dir

logger = logging.getLogger(__name__)


class ShardedMatrixLoader:
    """Loads shard tensors once and serves arbitrary metadata-row subsets."""

    def __init__(self, cache_dir, index_df: pd.DataFrame) -> None:
        self.reader = ActivationCacheReader(cache_dir)
        self.index_df = index_df.reset_index(drop=True)
        self._shards: dict[int, np.ndarray] = {}

    def _load_shard(self, sid: int) -> np.ndarray:
        if sid not in self._shards:
            import json

            from safetensors.numpy import load_file

            d = shard_dir(self.reader.root, int(sid))
            with (d / "_complete.json").open(encoding="utf-8") as fh:
                marker = json.load(fh)
            arr = load_file(str(d / "activations.safetensors"))["activations"]
            assert arr.shape[0] == marker["n_rows"], "marker/tensor mismatch"
            self._shards[sid] = np.asarray(arr, dtype=np.float32)
        return self._shards[sid]

    def rows_for(self, mask: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
        """Return stacked activations and their metadata rows for ``mask``."""
        meta = self.index_df[mask].sort_values(["tensor_key"]).reset_index(drop=True)
        pieces = []
        for sid, group in meta.groupby("shard", sort=True):
            arr = self._load_shard(int(sid))
            keys = group["tensor_key"].to_numpy().astype(int)
            pieces.append(arr[keys])
        X = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 0), dtype=np.float32)
        return np.asarray(X, dtype=np.float64), meta


def design_matrices_for(
    loader: ShardedMatrixLoader,
    index_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    selector: str,
    layer: int,
    label_of: Callable[[str], int],
) -> dict[str, dict]:
    """Assemble ``{split: {'X', 'y', 'sample_ids'}}`` for one (selector, layer).

    Excludes confirmation-split rows unless explicit allow flag flows in from
    callers (E00 never passes it).
    """
    sub = index_df[
        (index_df["token_selector"] == selector) & (index_df["layer"] == layer)
    ]
    out: dict[str, dict] = {}
    for split_name, sub_split in sub.groupby("split"):
        ids = sub_split["sample_id"].tolist()
        X, meta = loader.rows_for(
            (index_df["token_selector"] == selector)
            & (index_df["layer"] == layer)
            & (index_df["split"] == split_name)
        )
        if len(meta) != len(ids):
            raise RuntimeError("row alignment failure between index and metadata")
        y = np.asarray([label_of(sid) for sid in meta["sample_id"]], dtype=int)
        out[split_name] = {"X": X, "y": y, "sample_ids": meta["sample_id"].tolist()}
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

    pipe_base = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
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
