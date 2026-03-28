from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Data ingestion and management.", no_args_is_help=True)
console = Console()


@app.command()
def fetch(
    source: str = typer.Argument(
        "all",
        help="Data source to fetch (mp, jarvis, all).",
    ),
    force: bool = typer.Option(False, help="Force re-download even if cached."),
) -> None:
    if source in ("mp", "all"):
        console.print("[bold]Fetching Materials Project data...[/bold]")
        from matscreen.data.mp_client import fetch_mp_summary

        df = fetch_mp_summary(force=force)
        console.print(f"  Materials Project: {len(df)} materials loaded.")

    if source in ("jarvis", "all"):
        console.print("[bold]Fetching JARVIS dft_3d data...[/bold]")
        from matscreen.data.jarvis_client import fetch_jarvis_dft3d

        df = fetch_jarvis_dft3d(force=force)
        console.print(f"  JARVIS: {len(df)} materials loaded.")


@app.command()
def status() -> None:
    table = Table(title="Dataset Status")
    table.add_column("Source")
    table.add_column("Records")
    table.add_column("Path")
    table.add_column("Stale")

    from matscreen.data.cache import is_stale, load_parquet
    from matscreen.data.jarvis_client import DEFAULT_OUTPUT as JARVIS_PATH
    from matscreen.data.mp_client import DEFAULT_OUTPUT as MP_PATH

    for name, path in [("Materials Project", MP_PATH), ("JARVIS", JARVIS_PATH)]:
        df = load_parquet(path)
        if df is not None:
            table.add_row(name, str(len(df)), str(path), str(is_stale(path)))
        else:
            table.add_row(name, "not downloaded", str(path), "n/a")

    console.print(table)
