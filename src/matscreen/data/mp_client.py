from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn

from matscreen.data.cache import is_stale, load_parquet, save_parquet
from matscreen.data.schema import DataSource, PropertySet, SymmetryInfo

DEFAULT_OUTPUT = Path("data/raw/mp_summary.parquet")

FETCH_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "symmetry",
    "density",
]


def fetch_mp_summary(
    output_path: Path = DEFAULT_OUTPUT,
    force: bool = False,
) -> pd.DataFrame:
    cached = load_parquet(output_path)
    if cached is not None and not force and not is_stale(output_path):
        return cached

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set MP_API_KEY environment variable. "
            "Get one free at https://materialsproject.org/api"
        )

    from mp_api.client import MPRester

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        progress.add_task("Fetching Materials Project data...", total=None)

        with MPRester(api_key) as mpr:
            results = mpr.materials.summary.search(
                fields=FETCH_FIELDS,
                chunk_size=1000,
            )

    records = []
    for doc in results:
        sym = doc.symmetry
        records.append({
            "material_id": doc.material_id,
            "source": DataSource.MATERIALS_PROJECT.value,
            "formula": doc.formula_pretty,
            "structure_json": doc.structure.as_dict() if doc.structure else None,
            "crystal_system": sym.crystal_system.value if sym and sym.crystal_system else None,
            "space_group": sym.symbol if sym else None,
            "band_gap": doc.band_gap,
            "band_gap_type": "PBE",
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_above_hull": doc.energy_above_hull,
            "is_stable": doc.is_stable,
            "density": doc.density,
        })

    df = pd.DataFrame(records)
    save_parquet(df, output_path)
    return df


def load_mp_data(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame | None:
    return load_parquet(path)


def mp_to_property_set(row: pd.Series) -> PropertySet:
    return PropertySet(
        band_gap=row.get("band_gap"),
        band_gap_type=row.get("band_gap_type"),
        formation_energy_per_atom=row.get("formation_energy_per_atom"),
        energy_above_hull=row.get("energy_above_hull"),
        is_stable=row.get("is_stable"),
        density=row.get("density"),
    )


def mp_to_symmetry(row: pd.Series) -> SymmetryInfo:
    return SymmetryInfo(
        crystal_system=row.get("crystal_system"),
        space_group=row.get("space_group"),
    )
