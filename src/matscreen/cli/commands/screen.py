from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from matscreen.data.cache import load_parquet
from matscreen.data.schema import TriageLabel
from matscreen.features.composition import CompositionFeaturiser
from matscreen.models.xgboost_ensemble import XGBoostEnsemble
from matscreen.screening.engine import ScreeningEngine
from matscreen.screening.solar import (
    shockley_queisser_efficiency,
    solar_objectives,
)
from matscreen.uncertainty.calibration import IsotonicCalibrator
from matscreen.uncertainty.ood import DomainDetector
from matscreen.uncertainty.triage import TriageAssigner

app = typer.Typer(help="Materials screening.", no_args_is_help=True)
console = Console()


@app.command()
def run(
    bandgap: str = typer.Option(
        "1.0:1.8",
        help="Target band gap range as low:high (eV).",
    ),
    max_ehull: float = typer.Option(0.05, help="Maximum energy above hull (eV/atom)."),
    exclude_elements: str = typer.Option("", help="Comma-separated elements to exclude."),
    top: int = typer.Option(20, help="Number of top candidates to return."),
    model_dir: str = typer.Option("data/models/bandgap", help="Trained model directory."),
    output: str = typer.Option("results/screening.json", help="Output path."),
    export_dft_queue: bool = typer.Option(False, help="Export VERIFY materials as CSV."),
) -> None:
    parts = bandgap.split(":")
    low, high = float(parts[0]), float(parts[1])
    excluded = [e.strip() for e in exclude_elements.split(",") if e.strip()]

    console.print(f"[bold]Screening for solar absorbers (band gap {low} to {high} eV)[/bold]")

    model_path = Path(model_dir)
    if not (model_path / "metadata.json").exists():
        console.print("[red]No trained model found. Run 'matscreen train run' first.[/red]")
        raise typer.Exit(1)

    ensemble = XGBoostEnsemble()
    ensemble.load(model_path)
    console.print(f"  Loaded {ensemble.n_models}-member ensemble")

    calibrator = IsotonicCalibrator()
    cal_path = model_path / "calibrator.json"
    if cal_path.exists():
        calibrator.load(cal_path)
        console.print("  Loaded calibrator")

    detector = DomainDetector()
    ood_path = model_path / "ood"
    if (ood_path / "config.json").exists():
        detector.load(ood_path)
        console.print("  Loaded OOD detector")

    mp_data = load_parquet(Path("data/raw/mp_summary.parquet"))
    jarvis_data = load_parquet(Path("data/raw/jarvis_dft3d.parquet"))
    frames = [df for df in [mp_data, jarvis_data] if df is not None]
    if not frames:
        console.print("[red]No data. Run 'matscreen data fetch' first.[/red]")
        raise typer.Exit(1)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["formula"])
    console.print(f"  Materials loaded: {len(data):,}")

    console.print("  Featurising...")
    featuriser = CompositionFeaturiser()
    features = featuriser.featurise(data["formula"], data["material_id"])
    common_ids = features.index.intersection(data.set_index("material_id").index)
    features = features.loc[common_ids]
    data_indexed = data.set_index("material_id").loc[common_ids]

    console.print("  Predicting...")
    means, stds = ensemble.predict(features)
    calibrated_stds = calibrator.calibrate(stds)
    ood_flags = detector.is_ood(features.values, stds)

    triage_assigner = TriageAssigner()
    triage_labels = triage_assigner.assign(calibrated_stds, ood_flags)

    data_indexed["band_gap_pred"] = means
    data_indexed["bandgap_std"] = calibrated_stds
    data_indexed["ood_score"] = detector.score(features.values)
    data_indexed["triage_label"] = [label.value for label in triage_labels]
    data_indexed["sq_efficiency"] = data_indexed["band_gap_pred"].apply(shockley_queisser_efficiency)

    if "band_gap" not in data_indexed.columns:
        data_indexed["band_gap"] = data_indexed["band_gap_pred"]

    objectives = solar_objectives(low, high)
    value_columns = {
        "sq_efficiency": "sq_efficiency",
        "formation_energy": "formation_energy_per_atom",
        "uncertainty": "bandgap_std",
        "abundance": "abundance_score",
    }

    engine = ScreeningEngine(
        objectives=objectives,
        value_columns=value_columns,
        triage_assigner=triage_assigner,
    )
    results = engine.screen(
        data_indexed.reset_index(),
        max_ehull=max_ehull,
        excluded_elements=excluded or None,
        top_k=top,
    )

    summary = triage_assigner.summary(triage_labels)
    console.print()
    table = Table(title="Triage Summary")
    table.add_column("Label")
    table.add_column("Count")
    for label, count in summary.items():
        table.add_row(label.upper(), str(count))
    console.print(table)

    console.print(f"\n[bold]Top {len(results)} candidates:[/bold]")
    for _, row in results.iterrows():
        triage = row.get("triage_label", "defer")
        colour = {"trust": "green", "verify": "yellow", "defer": "red"}.get(triage, "white")
        console.print(
            f"  [{colour}]{triage.upper()}[/{colour}] "
            f"{row.get('formula', '?')} "
            f"(SQ: {row.get('sq_efficiency', 0):.1%}, "
            f"Eg: {row.get('band_gap', 0):.2f} eV, "
            f"std: {row.get('bandgap_std', 0):.3f})"
        )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_json(out_path, orient="records", indent=2)
    console.print(f"\nResults saved to {out_path}")

    if export_dft_queue:
        verify_df = results[results["triage_label"] == TriageLabel.VERIFY.value]
        if len(verify_df) > 0:
            dft_path = out_path.parent / "dft_queue.csv"
            verify_df.to_csv(dft_path, index=False)
            console.print(f"DFT queue ({len(verify_df)} materials) saved to {dft_path}")
