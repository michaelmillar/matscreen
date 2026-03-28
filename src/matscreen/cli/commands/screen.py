from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Materials screening.", no_args_is_help=True)
console = Console()


@app.command()
def run(
    bandgap: str = typer.Option(
        "1.0:1.5",
        help="Target band gap range as low:high (eV).",
    ),
    max_ehull: float = typer.Option(0.1, help="Maximum energy above hull (eV/atom)."),
    exclude_elements: str = typer.Option("", help="Comma-separated elements to exclude."),
    top: int = typer.Option(20, help="Number of top candidates to return."),
    output: str = typer.Option("results/screening.json", help="Output path."),
) -> None:
    parts = bandgap.split(":")
    low, high = float(parts[0]), float(parts[1])

    excluded = [e.strip() for e in exclude_elements.split(",") if e.strip()]

    console.print(f"[bold]Screening for band gap {low} to {high} eV[/bold]")
    console.print(f"  Max ehull: {max_ehull} eV/atom")
    if excluded:
        console.print(f"  Excluding: {', '.join(excluded)}")
    console.print(f"  Top {top} candidates")
    console.print("Screening pipeline not yet fully wired.")
