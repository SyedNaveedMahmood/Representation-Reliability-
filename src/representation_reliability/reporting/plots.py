"""Layerwise decodability figures using matplotlib defaults."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_d_by_layer(
    metrics_df: pd.DataFrame,
    out_path: str | Path,
    selector_col: str = "token_selector",
    layer_col: str = "layer",
    metric_col: str = "auroc",
    title: str = "AUROC by layer (discovery test)",
    random_label_ref: dict[str, float] | None = None,
):
    """Line-per-selector AUROC-vs-layer plot.

    ``random_label_ref`` may provide {'mean': x, 'std': y} drawn as a shaded
    reference band using default colors.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    selectors = sorted(metrics_df[selector_col].unique())
    for sel in selectors:
        sub = metrics_df[metrics_df[selector_col] == sel].sort_values(layer_col)
        ax.plot(sub[layer_col], sub[metric_col], marker="o", label=str(sel))
    if random_label_ref is not None:
        mean = random_label_ref.get("mean")
        std = random_label_ref.get("std")
        if mean is not None:
            if std is not None:
                ax.axhspan(mean - std, mean + std, alpha=0.25,
                           label="random-label ±1 sd")
            else:
                ax.axhline(mean, linestyle="--", label="random-label mean")
    ax.set_xlabel("layer (0-indexed)")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_d_combined(
    metrics_df: pd.DataFrame,
    random_label_df: pd.DataFrame,
    out_path: str | Path,
    metric_col: str = "auroc",
    title: str = "Decodability by layer with random-label control",
):
    """Combined panel: probe curves per selector + random-label mean band."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rl = (
        random_label_df.groupby(["token_selector", "layer"])[metric_col]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        if len(random_label_df) else pd.DataFrame()
    )

    fig, axes = plt.subplots(
        len(sorted(metrics_df["token_selector"].unique())), 1,
        figsize=(9, 4 * max(1, len(sorted(metrics_df["token_selector"].unique())))),
        squeeze=False, sharex=True,
    )
    for ax, sel in zip(axes.ravel(), sorted(metrics_df["token_selector"].unique())):
        sub = metrics_df[metrics_df["token_selector"] == sel].sort_values("layer")
        ax.plot(sub["layer"], sub[metric_col], marker="o", label="probe")
        rl_sel = rl[rl["token_selector"] == sel] if len(rl) else []
        if len(rl_sel):
            ax.fill_between(
                rl_sel["layer"], rl_sel["min"], rl_sel["max"],
                alpha=0.3, label="random-label range",
            )
            ax.plot(rl_sel["layer"], rl_sel["mean"], linestyle="--",
                    label="random-label mean")
        ax.set_ylabel(metric_col)
        ax.set_title(f"{title} — {sel}")
        ax.legend(loc="best")
    axes[-1, 0].set_xlabel("layer (0-indexed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
