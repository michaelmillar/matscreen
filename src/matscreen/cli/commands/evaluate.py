from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Model evaluation.", no_args_is_help=True)
console = Console()


@app.command()
def run(
    benchmark: str = typer.Option("matbench", help="Benchmark suite."),
    task: str = typer.Option("matbench_mp_gap", help="Specific task to evaluate."),
) -> None:
    console.print(f"[bold]Evaluating on {benchmark} / {task}[/bold]")
    console.print("Evaluation pipeline not yet implemented.")
