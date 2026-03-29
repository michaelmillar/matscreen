from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

from matscreen.data.cache import load_parquet
from matscreen.features.composition import CompositionFeaturiser
from matscreen.models.xgboost_ensemble import XGBoostEnsemble

app = typer.Typer(help="Model training.", no_args_is_help=True)
console = Console()

TARGET_COLUMNS = {
    "bandgap": "band_gap",
    "eform": "formation_energy_per_atom",
}


@app.command()
def run(
    model: str = typer.Option("xgb", help="Model type (xgb)."),
    target: str = typer.Option("bandgap", help="Target property (bandgap, eform)."),
    seed: int = typer.Option(42, help="Random seed."),
    output_dir: str = typer.Option("data/models", help="Output directory for checkpoints."),
    n_models: int = typer.Option(5, help="Ensemble size."),
) -> None:
    if target not in TARGET_COLUMNS:
        console.print(f"[red]Unknown target: {target}. Choose from: {list(TARGET_COLUMNS)}[/red]")
        raise typer.Exit(1)

    target_col = TARGET_COLUMNS[target]
    console.print(f"[bold]Training {model} ensemble ({n_models} members) for {target}[/bold]")

    mp_data = load_parquet(Path("data/raw/mp_summary.parquet"))
    jarvis_data = load_parquet(Path("data/raw/jarvis_dft3d.parquet"))

    frames = [df for df in [mp_data, jarvis_data] if df is not None]
    if not frames:
        console.print("[red]No data found. Run 'matscreen data fetch' first.[/red]")
        raise typer.Exit(1)

    import pandas as pd
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=[target_col, "formula"])
    console.print(f"  Materials with {target}: {len(data):,}")

    console.print("  Featurising compositions...")
    featuriser = CompositionFeaturiser()
    features = featuriser.featurise(data["formula"], data["material_id"])
    common_ids = features.index.intersection(data.set_index("material_id").index)
    features = features.loc[common_ids]
    targets = data.set_index("material_id").loc[common_ids, target_col].values.astype(float)
    console.print(f"  Featurised: {len(features):,} materials")

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, targets, test_size=0.4, random_state=seed,
    )
    X_val, X_cal, y_val, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed,
    )
    console.print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Cal: {len(X_cal):,}")

    ensemble = XGBoostEnsemble(n_models=n_models)
    console.print("  Training ensemble...")
    metrics = ensemble.train(X_train, y_train, X_val, y_val)

    save_path = Path(output_dir) / target
    ensemble.save(save_path)
    console.print(f"  Model saved to {save_path}")

    np.savez(
        save_path / "calibration_data.npz",
        X_cal=X_cal.values,
        y_cal=y_cal,
        feature_names=X_cal.columns.tolist(),
    )

    table = Table(title="Training Results")
    table.add_column("Metric")
    table.add_column("Value")
    for key, val in sorted(metrics.items()):
        table.add_row(key, f"{val:.4f}")
    console.print(table)
