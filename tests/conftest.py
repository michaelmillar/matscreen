import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_materials_df() -> pd.DataFrame:
    return pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(10)],
        "formula": ["Si", "GaAs", "CdTe", "Fe2O3", "TiO2", "ZnO", "AlN", "SiC", "BN", "GaN"],
        "band_gap": [1.11, 1.42, 1.50, 2.20, 3.03, 3.30, 6.00, 2.36, 6.40, 3.40],
        "formation_energy_per_atom": [
            0.0, -0.20, -0.15, -0.85, -1.80, -0.70, -1.50, -0.30, -1.30, -0.55,
        ],
        "energy_above_hull": [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "bandgap_std": [0.05, 0.08, 0.10, 0.15, 0.03, 0.12, 0.20, 0.04, 0.25, 0.07],
    })


@pytest.fixture
def random_cost_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((20, 3))
