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
def e01a(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    layer: int = typer.Option(17, "--layer", help="0-indexed resid_post intervention layer"),
    profile: str = typer.Option("full", "--profile", help="alpha profile: smoke, pilot, or full"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="optional deterministic cap on discovery-test pairs"
    ),
    random_directions: int = typer.Option(
        10, "--random-directions", min=1, help="number of random control directions"
    ),
    trace_layers: str = typer.Option(
        "17,20,23,26,27",
        "--trace-layers",
        help="comma-separated resid_post layers captured after intervention",
    ),
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E01A truth-coordinate causal-conversion discovery."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e01a import run_e01a

    run_dir = run_e01a(
        base,
        model,
        experiment,
        tuple(overrides or ()),
        layer=layer,
        profile=profile,
        max_pairs=max_pairs,
        random_directions=random_directions,
        trace_layers=trace_layers,
    )
    typer.echo(f"E01A run complete: {run_dir}")


@app.command()
def e01b(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    layer: int = typer.Option(17, "--layer", help="0-indexed resid_post intervention layer"),
    profile: str = typer.Option("full", "--profile", help="bounded profile: smoke, pilot, or full"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="optional deterministic discovery-test pair cap"
    ),
    random_directions: int = typer.Option(
        10, "--random-directions", min=1, help="norm-matched random controls"
    ),
    orthogonal_directions: int = typer.Option(
        10,
        "--orthogonal-directions",
        min=1,
        help="norm-matched probe-orthogonal random controls",
    ),
    trace_layers: str = typer.Option(
        "17,20,23,27",
        "--trace-layers",
        help="comma-separated intervened-forward resid_post capture layers",
    ),
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E01B-1 source-free setpoint causality discovery."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e01b import run_e01b

    run_dir = run_e01b(
        base,
        model,
        experiment,
        tuple(overrides or ()),
        layer=layer,
        profile=profile,
        max_pairs=max_pairs,
        random_directions=random_directions,
        orthogonal_directions=orthogonal_directions,
        trace_layers=trace_layers,
    )
    typer.echo(f"E01B-1 run complete: {run_dir}")


@app.command()
def e01b2(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    layer: int = typer.Option(17, "--layer", help="frozen resid_post layer"),
    profile: str = typer.Option("full", "--profile", help="bounded profile: smoke, pilot, or full"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="optional deterministic discovery pair cap"
    ),
    context_strengths: str = typer.Option(
        "0.5,1.0",
        "--context-strengths",
        help="frozen comma-separated orthogonal context strengths",
    ),
    random_orthogonal_directions: int = typer.Option(
        10,
        "--random-orthogonal-directions",
        min=1,
        help="number of deterministic random orthogonal contexts",
    ),
    trace_layers: str = typer.Option(
        "17,20,23,27",
        "--trace-layers",
        help="frozen intervened-forward trace layers",
    ),
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E01B-2 fixed-setpoint orthogonal-context modulation."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e01b2 import run_e01b2

    run_dir = run_e01b2(
        base,
        model,
        experiment,
        tuple(overrides or ()),
        layer=layer,
        profile=profile,
        max_pairs=max_pairs,
        context_strengths=context_strengths,
        random_orthogonal_directions=random_orthogonal_directions,
        trace_layers=trace_layers,
    )
    typer.echo(f"E01B-2 run complete: {run_dir}")


@app.command()
def e01b3(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    layer: int = typer.Option(17, "--layer", help="frozen resid_post layer"),
    profile: str = typer.Option("full", "--profile", help="bounded profile: smoke, pilot, or full"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="optional deterministic discovery pair cap"
    ),
    context_strengths: str = typer.Option(
        "0.5,1.0",
        "--context-strengths",
        help="frozen comma-separated orthogonal context strengths",
    ),
    random_orthogonal_directions: int = typer.Option(
        10,
        "--random-orthogonal-directions",
        min=1,
        help="number of frozen E01B-2 random orthogonal contexts",
    ),
    trace_layers: str = typer.Option(
        "17,20,23,27",
        "--trace-layers",
        help="frozen intervened-forward trace layers",
    ),
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run E01B-3 additive-vs-gating factorial decomposition."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e01b3 import run_e01b3

    run_dir = run_e01b3(
        base,
        model,
        experiment,
        tuple(overrides or ()),
        layer=layer,
        profile=profile,
        max_pairs=max_pairs,
        context_strengths=context_strengths,
        random_orthogonal_directions=random_orthogonal_directions,
        trace_layers=trace_layers,
    )
    typer.echo(f"E01B-3 run complete: {run_dir}")


@app.command("confirm-actionability")
def confirm_actionability(
    protocol_commit: str = typer.Option(
        ...,
        "--protocol-commit",
        help="exact remotely pushed confirmation preregistration commit",
    ),
) -> None:
    """Execute the single locked two-checkpoint actionability confirmation."""
    logging.basicConfig(level=logging.INFO)
    from .runners.confirmation import run_confirmation

    run_dir = run_confirmation(protocol_commit=protocol_commit)
    typer.echo(f"Actionability confirmation complete: {run_dir}")


@app.command()
def e14(
    base: Path | None = None,
    model: Path | None = None,
    experiment: Path | None = None,
    precision: str = typer.Option(..., "--precision", help="bf16, int8, or int4"),
    profile: str = typer.Option(..., "--profile", help="authorized: smoke or pilot"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="optional cap within the frozen bounded profile"
    ),
    layer: int = typer.Option(17, "--layer", help="frozen resid_post layer"),
    trace_layers: str = typer.Option(
        "17,20,23,27", "--trace-layers", help="frozen trace layers"
    ),
    overrides: list[str] = typer.Option(None, "--set", help="dot-path overrides"),
) -> None:
    """Run one precision of the authorized E14 smoke/pilot ladder."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e14 import run_e14

    run_dir = run_e14(
        base,
        model,
        experiment,
        tuple(overrides or ()),
        precision=precision,
        profile=profile,
        max_pairs=max_pairs,
        layer=layer,
        trace_layers=trace_layers,
    )
    typer.echo(f"E14 run complete: {run_dir}")


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
    raise typer.Exit("standalone probe is exercised inside `rr e00` in Phase 0A")


@app.command()
def intervene() -> None:
    """Legacy placeholder; E01A now provides the first bounded causal runner."""
    raise typer.Exit("use `rr e01a` for the bounded E01A truth-coordinate intervention")


@app.command()
def summarize(run_dir: Path) -> None:
    """Print the decodability tables recorded for a finished run."""
    import json

    probe_json = run_dir / "probe_metrics.json"
    if not probe_json.exists():
        raise typer.Exit(f"no probe_metrics.json under {run_dir}")
    rows = json.loads(probe_json.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    cols = [
        c
        for c in (
            "token_selector",
            "layer",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "ci_low",
            "ci_high",
        )
        if c in df
    ]
    df = df.sort_values(["token_selector", "auroc"], ascending=[True, False])
    with pd.option_context("display.max_rows", None):
        typer.echo(df[cols].to_string(index=False))
    summary_path = run_dir / "DISCOVERY_SUMMARY.md"
    if summary_path.exists():
        typer.echo("")
        typer.echo(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
