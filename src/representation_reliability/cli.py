from pathlib import Path
import typer

app = typer.Typer(help="Representation Reliability experiment harness.")

@app.command()
def validate_config(config: Path) -> None:
    raise NotImplementedError("Phase 0.1: implement config validation per AGENTS.md.")

@app.command()
def extract(config: Path) -> None:
    raise NotImplementedError("Phase 0.3: implement resumable extraction.")

@app.command()
def probe(config: Path) -> None:
    raise NotImplementedError("Phase 0.5: implement linear probe pipeline.")

@app.command()
def intervene(config: Path) -> None:
    raise NotImplementedError("Phase 0.7-0.9: implement intervention runners.")

@app.command()
def run(config: Path) -> None:
    raise NotImplementedError("Implement after individual stages are stable.")

@app.command()
def summarize(run_dir: Path) -> None:
    raise NotImplementedError("Phase 0.10: implement reporting.")

if __name__ == "__main__":
    app()
