"""Command-line interface for the representation reliability harness."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer

from .config import ConfigError, resolve_config

app = typer.Typer(
    help="Representation Reliability experiment harness.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def validate_config(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Validate a configuration without running anything."""
    try:
        cfg, _provenance = resolve_config(base, model, experiment, overrides or ())
    except ConfigError as exc:
        typer.secho(f"INVALID CONFIG: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    typer.echo("config valid")
    typer.echo(cfg.to_dict())


@app.command()
def e00(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E00 Representation Cartography (decodability scan)."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e00 import run_e00

    run_dir = run_e00(base, model, experiment, tuple(overrides or ()))
    typer.echo(f"E00 run complete: {run_dir}")


@app.command()
def e00b(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E00-B behavioral forced-choice readout (same synthetic task)."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e00b import run_e00b

    run_dir = run_e00b(base, model, experiment, tuple(overrides or ()))
    typer.echo(f"E00-B run complete: {run_dir}")


@app.command()
def e00c(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E00-C representation-origin and fixed-readout diagnostics."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e00c import run_e00c

    run_dir = run_e00c(base, model, experiment, tuple(overrides or ()))
    typer.echo(f"E00-C run complete: {run_dir}")


@app.command("e00c-followup")
def e00c_followup(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    layer: int = typer.Option(17, "--layer", help="Residual-stream layer to audit"),
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run the narrow Phase 0A.2 chat-calibration/readout-geometry follow-up."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e00c_followup import run_e00c_followup

    run_dir = run_e00c_followup(
        base,
        model,
        experiment,
        tuple(overrides or ()),
        layer=layer,
    )
    typer.echo(f"E00-C follow-up complete: {run_dir}")


@app.command()
def extract(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    overrides: list[str] = typer.Option(None, "--set"),
) -> None:
    """Standalone resumable activation-extraction stage."""
    raise typer.Exit(
        "standalone extract is exercised inside `rr e00` in Phase 0A; "
        "dedicated stages arrive with phase-specific experiments"
    )


@app.command()
def probe() -> None:
    raise typer.Exit(
        "standalone probe is exercised inside `rr e00` in Phase 0A"
    )


@app.command()
def intervene() -> None:
    """Interventions are Phase 0B+; not implemented by design yet."""
    raise typer.Exit("interventions belong to Phase 0B (E01/E02); not implemented yet")


@app.command()
def summarize(run_dir: Path) -> None:
    """Print the decodability tables recorded for a finished run."""
    import json

    probe_json = run_dir / "probe_metrics.json"
    if not probe_json.exists():
        raise typer.Exit(f"no probe_metrics.json under {run_dir}")
    rows = json.loads(probe_json.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    cols = [c for c in ("token_selector", "layer", "auroc", "auprc",
                        "balanced_accuracy", "ci_low", "ci_high") if c in df]
    df = df.sort_values(["token_selector", "auroc"], ascending=[True, False])
    with pd.option_context("display.max_rows", None):
        typer.echo(df[cols].to_string(index=False))
    summary_path = run_dir / "DISCOVERY_SUMMARY.md"
    if summary_path.exists():
        typer.echo("")
        typer.echo(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
