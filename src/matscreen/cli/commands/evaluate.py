from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from matscreen.models.xgboost_ensemble import XGBoostEnsemble
from matscreen.uncertainty.calibration import (
    IsotonicCalibrator,
    miscalibration_area,
    reliability_diagram,
)

app = typer.Typer(help="Model evaluation.", no_args_is_help=True)
console = Console()


@app.command()
def run(
    model_dir: str = typer.Option("data/models/bandgap", help="Model directory."),
    output_dir: str = typer.Option("results", help="Output directory for results."),
) -> None:
    model_path = Path(model_dir)
    if not (model_path / "metadata.json").exists():
        console.print("[red]No trained model found. Run 'matscreen train run' first.[/red]")
        raise typer.Exit(1)

    ensemble = XGBoostEnsemble()
    ensemble.load(model_path)
    console.print(f"[bold]Evaluating {ensemble.name} ({ensemble.n_models} members)[/bold]")

    cal_data_path = model_path / "calibration_data.npz"
    if not cal_data_path.exists():
        console.print("[red]No calibration data found.[/red]")
        raise typer.Exit(1)

    import pandas as pd
    cal_data = np.load(cal_data_path, allow_pickle=True)
    X_cal = pd.DataFrame(cal_data["X_cal"], columns=cal_data["feature_names"])
    y_cal = cal_data["y_cal"]

    means, stds = ensemble.predict(X_cal)
    mae = float(np.mean(np.abs(means - y_cal)))
    rmse = float(np.sqrt(np.mean((means - y_cal) ** 2)))

    calibrator = IsotonicCalibrator()
    cal_path = model_path / "calibrator.json"
    if cal_path.exists():
        calibrator.load(cal_path)
        stds = calibrator.calibrate(stds)

    miscal = miscalibration_area(means, stds, y_cal)
    diag = reliability_diagram(means, stds, y_cal)

    table = Table(title="Evaluation Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("MAE", f"{mae:.4f}")
    table.add_row("RMSE", f"{rmse:.4f}")
    table.add_row("Miscalibration Area", f"{miscal:.4f}")
    console.print(table)

    console.print("\n[bold]Reliability Diagram (expected vs observed coverage):[/bold]")
    for exp, obs in zip(diag["expected_coverage"][::4], diag["observed_coverage"][::4]):
        bar_len = int(obs * 40)
        bar = "=" * bar_len
        marker = "|" if abs(exp - obs) < 0.05 else "!"
        console.print(f"  {exp:.0%} expected: [{bar}{marker}] {obs:.0%} observed")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path / "reliability.npz",
        expected=diag["expected_coverage"],
        observed=diag["observed_coverage"],
        bin_counts=diag["bin_counts"],
    )
    console.print(f"\nReliability data saved to {out_path / 'reliability.npz'}")
