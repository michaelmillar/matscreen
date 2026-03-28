from __future__ import annotations

import typer

from matscreen.cli.commands import data, evaluate, report, screen, train

app = typer.Typer(
    name="matscreen",
    help="Uncertainty-aware materials screening with calibrated confidence intervals.",
    no_args_is_help=True,
)

app.add_typer(data.app, name="data")
app.add_typer(train.app, name="train")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(screen.app, name="screen")
app.add_typer(report.app, name="report")


if __name__ == "__main__":
    app()
