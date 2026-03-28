from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Report generation.", no_args_is_help=True)
console = Console()


@app.command()
def run(
    input_path: str = typer.Option("results/screening.json", help="Screening results path."),
    format: str = typer.Option("html", help="Output format (html, json)."),
    output: str = typer.Option("results/report.html", help="Output file path."),
) -> None:
    console.print(f"[bold]Generating {format} report from {input_path}[/bold]")
    console.print("Report generation not yet implemented.")
