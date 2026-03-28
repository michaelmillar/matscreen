from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn

from matscreen.data.cache import is_stale, load_parquet, save_parquet
from matscreen.data.schema import DataSource, PropertySet

DEFAULT_OUTPUT = Path("data/raw/jarvis_dft3d.parquet")


def fetch_jarvis_dft3d(
    output_path: Path = DEFAULT_OUTPUT,
    force: bool = False,
) -> pd.DataFrame:
    cached = load_parquet(output_path)
    if cached is not None and not force and not is_stale(output_path):
        return cached

    from jarvis.db.figshare import data as jarvis_data

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Downloading JARVIS dft_3d dataset...", total=None)
        raw = jarvis_data("dft_3d")

    records = []
    for entry in raw:
        atoms_dict = entry.get("atoms")
        records.append({
            "material_id": entry.get("jid", ""),
            "source": DataSource.JARVIS.value,
            "formula": entry.get("formula", ""),
            "atoms_dict": atoms_dict,
            "band_gap_optb88vdw": entry.get("optb88vdw_bandgap"),
            "band_gap_mbj": entry.get("mbj_bandgap"),
            "formation_energy_per_atom": entry.get("formation_energy_peratom"),
            "energy_above_hull": entry.get("ehull"),
            "bulk_modulus_kv": entry.get("bulk_modulus_kv"),
            "space_group": entry.get("spg_symbol"),
            "crystal_system": entry.get("crys"),
        })

    df = pd.DataFrame(records)
    save_parquet(df, output_path)
    return df


def load_jarvis_data(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame | None:
    return load_parquet(path)


def jarvis_to_property_set(row: pd.Series) -> PropertySet:
    band_gap = row.get("band_gap_mbj")
    gap_type = "TBmBJ"
    if band_gap is None or pd.isna(band_gap):
        band_gap = row.get("band_gap_optb88vdw")
        gap_type = "OptB88vdW"

    return PropertySet(
        band_gap=band_gap if band_gap is not None and not pd.isna(band_gap) else None,
        band_gap_type=gap_type,
        formation_energy_per_atom=row.get("formation_energy_per_atom"),
        energy_above_hull=row.get("energy_above_hull"),
        bulk_modulus_kv=row.get("bulk_modulus_kv"),
    )
