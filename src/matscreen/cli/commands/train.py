from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Model training.", no_args_is_help=True)
console = Console()


@app.command()
def run(
    model: str = typer.Option("xgb", help="Model type (xgb, alignn)."),
    target: str = typer.Option("bandgap", help="Target property (bandgap, eform, kvrh)."),
    seed: int = typer.Option(42, help="Random seed."),
    output_dir: str = typer.Option("data/models", help="Output directory for checkpoints."),
) -> None:
    console.print(f"[bold]Training {model} for {target} (seed={seed})[/bold]")
    console.print("Training pipeline not yet implemented.")
