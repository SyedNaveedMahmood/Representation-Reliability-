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
    profile: str = typer.Option(..., "--profile", help="smoke, pilot, or authorized full"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="optional cap within the frozen bounded profile"
    ),
    layer: int = typer.Option(17, "--layer", help="frozen resid_post layer"),
    trace_layers: str = typer.Option("17,20,23,27", "--trace-layers", help="frozen trace layers"),
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


@app.command("confirm-e14")
def confirm_e14(
    protocol_commit: str = typer.Option(
        ..., "--protocol-commit", help="exact E14 preregistration commit"
    ),
) -> None:
    """Execute the single locked E14 precision confirmation."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e14_confirmation import run_e14_confirmation

    run_dir = run_e14_confirmation(protocol_commit=protocol_commit)
    typer.echo(f"E14 confirmation complete: {run_dir}")


@app.command("e13-bounded")
def e13_bounded() -> None:
    """Run the preregistered one-seed R0/R1/R2 distillation diagnostic."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13 import run_e13_bounded

    run_dir = run_e13_bounded()
    typer.echo(f"E13 bounded diagnostic complete: {run_dir}")


@app.command("e13-smoke")
def e13_smoke() -> None:
    """Run the tiny preregistered E13 optimization/hook contract."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13 import run_e13_training_smoke

    result = run_e13_training_smoke()
    typer.echo(f"E13 training smoke complete: {result}")


@app.command("e13-multiseed")
def e13_multiseed() -> None:
    """Run the frozen R0/R1/R2/R3 full-discovery campaign."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_multiseed import run_e13_multiseed_campaign

    run_dir = run_e13_multiseed_campaign()
    typer.echo(f"E13 multi-seed baseline campaign complete: {run_dir}")


@app.command("e13-multiseed-reference")
def e13_multiseed_reference() -> None:
    """Prepare the immutable teacher/R0 E13 discovery reference."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_multiseed import prepare_e13_multiseed_reference

    run_dir = prepare_e13_multiseed_reference()
    typer.echo(f"E13 multi-seed reference complete: {run_dir}")


@app.command("e13-multiseed-smoke")
def e13_multiseed_smoke() -> None:
    """Run the forty-row enhanced E13 evaluation contract."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_multiseed import run_e13_multiseed_smoke

    run_dir = run_e13_multiseed_smoke()
    typer.echo(f"E13 multi-seed smoke complete: {run_dir}")


@app.command("e13-multiseed-job")
def e13_multiseed_job(
    regime: str = typer.Option(..., "--regime"),
    seed: int = typer.Option(..., "--seed"),
) -> None:
    """Run one immutable E13 regime/seed job for the overnight scheduler."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_multiseed import run_e13_multiseed_job

    run_dir = run_e13_multiseed_job(regime.upper(), seed)
    typer.echo(f"E13 multi-seed job complete: {run_dir}")


@app.command("e13-multiseed-analyze")
def e13_multiseed_analyze() -> None:
    """Analyze completed R0/R1/R2/R3 discovery and apply the frozen gate."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_multiseed import analyze_e13_multiseed_campaign

    report = analyze_e13_multiseed_campaign()
    typer.echo(f"E13 multi-seed analysis complete: {report}")


@app.command("e13-method-cache")
def e13_method_cache() -> None:
    """Build and live-validate the frozen teacher response cache."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_methods import prepare_teacher_response_cache

    cache = prepare_teacher_response_cache()
    typer.echo(f"E13 teacher response cache complete: {cache}")


@app.command("e13-method-smoke")
def e13_method_smoke() -> None:
    """Run the two-row differentiable method/control GPU contract."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_methods import run_method_training_smoke

    output = run_method_training_smoke()
    typer.echo(f"E13 method smoke complete: {output}")


@app.command("e13-method-job")
def e13_method_job(
    regime: str = typer.Option(..., "--regime"),
    seed: int = typer.Option(..., "--seed"),
) -> None:
    """Run one immutable E13 conversion-response method/control job."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_methods import run_method_job

    run_dir = run_method_job(regime.upper(), seed)
    typer.echo(f"E13 method job complete: {run_dir}")


@app.command("e13-method-analyze")
def e13_method_analyze() -> None:
    """Analyze all method jobs under the frozen primary comparison."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_methods import analyze_method_campaign

    report = analyze_method_campaign()
    typer.echo(f"E13 method analysis complete: {report}")


@app.command("e13-response-diagnostic")
def e13_response_diagnostic() -> None:
    """Audit R5/R6 response scales, sensitivity, and full-model gradients."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_revision import run_response_regularization_diagnostic

    report = run_response_regularization_diagnostic()
    typer.echo(f"E13 response diagnostic complete: {report}")


@app.command("e13-revision-smoke")
def e13_revision_smoke() -> None:
    """Run frozen R7-R16 loss and gradient-strategy GPU contracts."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_revision import run_revision_smoke

    output = run_revision_smoke()
    typer.echo(f"E13 revision smoke complete: {output}")


@app.command("e13-revision-job")
def e13_revision_job(
    regime: str = typer.Option(..., "--regime"),
    seed: int = typer.Option(..., "--seed"),
) -> None:
    """Run one immutable frozen E13 revision job."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_revision import run_revision_job

    output = run_revision_job(regime.upper(), seed)
    typer.echo(f"E13 revision job complete: {output}")


@app.command("e13-revision-analyze")
def e13_revision_analyze() -> None:
    """Apply frozen bounded selection and, when complete, three-seed gates."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e13_revision import analyze_revision_campaign

    output = analyze_revision_campaign()
    typer.echo(f"E13 revision analysis complete: {output}")


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


@app.command("e13-confirmation-dry-run")
def e13_confirmation_dry_run(
    protocol_commit: str = typer.Option(..., "--protocol-commit"),
) -> None:
    """Exercise the E13 confirmation runner on discovery rows only."""
    from .runners.e13_diagnostic_confirmation import run_e13_diagnostic_confirmation

    output = run_e13_diagnostic_confirmation(protocol_commit=protocol_commit, dry_run=True)
    typer.echo(f"E13 confirmation dry run complete: {output}")


@app.command("e13-diagnostic-confirmation")
def e13_diagnostic_confirmation(
    protocol_commit: str = typer.Option(..., "--protocol-commit"),
) -> None:
    """Access the E13 confirmation holdout exactly once and apply frozen tests."""
    from .runners.e13_diagnostic_confirmation import run_e13_diagnostic_confirmation

    output = run_e13_diagnostic_confirmation(protocol_commit=protocol_commit, dry_run=False)
    typer.echo(f"E13 diagnostic confirmation complete: {output}")


@app.command("e17-screen")
def e17_screen() -> None:
    """Screen frozen E17 candidate pairs on engineering, B and D only."""
    from .runners.e17 import screen_candidates

    output = screen_candidates()
    typer.echo(f"E17 candidate screen complete: {output}")


@app.command("e17-reference")
def e17_reference() -> None:
    """Evaluate the frozen E17 teacher and R0 and build the teacher KD cache."""
    from .runners.e17_support import prepare_e17_reference

    output = prepare_e17_reference()
    typer.echo(f"E17 reference complete: {output}")


@app.command("e17-job")
def e17_job(
    regime: str = typer.Option(..., "--regime"),
    seed: int = typer.Option(..., "--seed"),
) -> None:
    """Train and evaluate one frozen E17 regime/seed."""
    from .runners.e17_support import run_e17_job

    output = run_e17_job(regime.upper(), int(seed))
    typer.echo(f"E17 job complete: {output}")


@app.command("e17-analyze")
def e17_analyze() -> None:
    """Build the E17 cross-family discovery tables and replication verdict."""
    from .runners.e17_analysis import analyze_e17

    output = analyze_e17()
    typer.echo(f"E17 analysis complete: {output}")


@app.command("e15-stage0")
def e15_stage0(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Freeze the E15 stateful task and validate exact state labels."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import run_e15_stage0

    run_dir = run_e15_stage0(model)
    typer.echo(f"E15 Stage 0 complete: {run_dir}")


@app.command("e15-stage1")
def e15_stage1(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Establish E15 D(k) and B(k) with no intervention."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import run_e15_stage1

    run_dir = run_e15_stage1(model)
    typer.echo(f"E15 Stage 1 complete: {run_dir}")


@app.command("e15-stage2")
def e15_stage2(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Run the bounded E15 intervention smoke (engineering gate only)."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import run_e15_stage2

    run_dir = run_e15_stage2(model)
    typer.echo(f"E15 Stage 2 complete: {run_dir}")


@app.command("e15-stage3")
def e15_stage3(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Run the full E15 horizon curve and evaluate the frozen Stage 4 gate."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import run_e15_stage3

    run_dir = run_e15_stage3(model)
    typer.echo(f"E15 Stage 3 complete: {run_dir}")


@app.command("e15-stage3b")
def e15_stage3b(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Run the secondary E15 distractor-density contrast (H15.3)."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import run_e15_stage3b

    run_dir = run_e15_stage3b(model)
    typer.echo(f"E15 Stage 3b complete: {run_dir}")


@app.command("e15-stage4")
def e15_stage4() -> None:
    """Replicate E15 on the frozen second checkpoint, only if the gate passed."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import run_e15_stage4

    run_dir = run_e15_stage4()
    typer.echo(f"E15 Stage 4 complete: {run_dir}")


@app.command("e15-analyze")
def e15_analyze(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Assemble the E15 discovery report from completed stage artifacts."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15 import analyze_e15

    output = analyze_e15(model)
    typer.echo(f"E15 analysis complete: {output}")


@app.command("e15-gate1")
def e15_gate1(
    model: str = typer.Option("qwen3_1.7b", "--model", help="frozen E15 model config name"),
) -> None:
    """Run the E15 carrier-sufficiency arms (full-state patch upper bound)."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e15_gate1 import run_e15_gate1

    run_dir = run_e15_gate1(model)
    typer.echo(f"E15 Gate 1 complete: {run_dir}")


@app.command("e01-calibration-audit")
def e01_calibration_audit(
    models: str = typer.Option(
        "qwen3_0.6b,qwen3_1.7b", "--models", help="comma-separated frozen checkpoints"
    ),
) -> None:
    """Run the frozen E01 residual-fraction calibration audit for both checkpoints."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e01_calibration_audit import run_e01_calibration_audit

    names = tuple(m.strip() for m in models.split(",") if m.strip())
    output = run_e01_calibration_audit(names)
    typer.echo(f"E01 calibration audit complete: {output}")


@app.command("e01-calibration-analyze")
def e01_calibration_analyze(
    models: str = typer.Option("qwen3_0.6b,qwen3_1.7b", "--models"),
) -> None:
    """Rebuild the calibrated cross-checkpoint comparison from completed rows."""
    logging.basicConfig(level=logging.INFO)
    from .runners.e01_calibration_audit import analyze_e01_calibration_audit

    names = tuple(m.strip() for m in models.split(",") if m.strip())
    output = analyze_e01_calibration_audit(names)
    typer.echo(f"E01 calibration analysis complete: {output}")


if __name__ == "__main__":
    app()
