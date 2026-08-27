import numpy as np

from representation_reliability.metrics.bootstrap import bootstrap_ci
from representation_reliability.metrics.decoding import (
    classification_metrics,
    majority_baseline_accuracy,
)
from representation_reliability.probes.linear import (
    evaluate_probe,
    fit_probe,
    save_probe,
    shuffle_labels,
)


def _blobs(n=400, d=8, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] > 0).astype(int)
    X[y == 1, 0] += sep
    X[y == 0, 0] -= sep
    return X, y


def test_probe_beats_majority_and_selects_C_on_validation():
    X, y = _blobs()
    n = len(y)
    Xtr, ytr = X[:240], y[:240]
    Xv, yv = X[240:320], y[240:320]
    Xte, yte = X[320:], y[320:]
    fit = fit_probe(Xtr, ytr, Xv, yv, c_grid=[0.01, 0.1, 1.0], seed=0)
    m = evaluate_probe(fit, Xte, yte)
    assert m["auroc"] is not None and m["auroc"] > 0.9
    assert majority_baseline_accuracy(yte) < 0.7
    assert set(fit["validation_auroc_by_C"]) == {"0.01", "0.1", "1.0"}
    assert fit["chosen_C"] in (0.01, 0.1, 1.0)


def test_random_label_control_is_at_chance():
    """Shuffled training labels must NOT transfer the real signal to held-out
    data. Scores have heavy ties under L2 regularization, so individual AUROCs
    can swing well off 0.5 in either direction at small eval sizes; the key
    property is that no seed approaches the separable-signal level."""
    X, y = _blobs(sep=6.0)
    Xtr, ytr = X[:300], y[:300]
    Xv, yv = X[300:340], y[300:340]
    Xte, yte = X[340:], y[340:]
    aurocs = []
    for seed in (0, 1, 2):
        y_shuf = shuffle_labels(ytr, seed=seed)
        fit = fit_probe(Xtr, y_shuf, Xv, yv, c_grid=[1.0], seed=seed)
        m = evaluate_probe(fit, Xte, yte)   # REAL labels: must fail to transfer
        aurocs.append(m["auroc"])
    assert max(abs(a - 0.5) for a in aurocs) < 0.4, aurocs


def test_save_probe_roundtrip(tmp_path):
    X, y = _blobs(n=200)
    fit = fit_probe(X[:120], y[:120], X[120:160], y[160 - 40 + 20:-20],
                    c_grid=[1.0])
    path = save_probe(fit, tmp_path / "probe.npz")
    with np.load(path) as z:
        assert z["coef"].shape == (1, X.shape[1])
        assert np.isfinite(z["intercept"]).all()


def test_classification_metrics_degenerate():
    m = classification_metrics(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]))
    assert m["auroc"] is None and "note" in m


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(4)
    n = 500
    y = rng.integers(0, 2, size=n)
    scores = y * 2.0 - 1.0 + rng.normal(0, 0.4, size=n)
    ci = bootstrap_ci(y, scores, lambda yt, s: __import__(
        "sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yt, s),
        n_bootstraps=400, seed=1)
    assert ci["ci_low"] <= ci["point_estimate"] <= ci["ci_high"]
    assert ci["ci_low"] > 0.55   # strongly separable case
