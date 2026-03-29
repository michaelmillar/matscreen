from __future__ import annotations

import numpy as np
from pymatgen.core import Composition

from matscreen.screening.objectives import Direction, Objective

SQ_TABLE_EV = np.array([
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.34,
    1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6,
    2.8, 3.0, 3.5, 4.0,
])

SQ_TABLE_EFF = np.array([
    0.04, 0.08, 0.12, 0.17, 0.21, 0.25, 0.28, 0.31, 0.33, 0.337,
    0.33, 0.31, 0.28, 0.25, 0.22, 0.19, 0.16, 0.11, 0.07, 0.04,
    0.02, 0.01, 0.002, 0.0,
])

EARTH_ABUNDANCE = {
    "O": 1.00, "Si": 0.98, "Al": 0.95, "Fe": 0.93, "Ca": 0.90,
    "Na": 0.88, "Mg": 0.87, "K": 0.85, "Ti": 0.80, "Mn": 0.75,
    "P": 0.73, "S": 0.70, "C": 0.68, "N": 0.65, "Cu": 0.50,
    "Zn": 0.48, "Sn": 0.40, "Pb": 0.35, "Ni": 0.33, "Cr": 0.30,
    "Ba": 0.28, "Sr": 0.27, "Zr": 0.26, "V": 0.25, "Li": 0.22,
    "Se": 0.20, "Ga": 0.18, "Ge": 0.15, "As": 0.14, "Cd": 0.12,
    "In": 0.10, "Te": 0.08, "Bi": 0.07, "Sb": 0.06, "Ag": 0.05,
    "Tl": 0.03, "Re": 0.02, "Hg": 0.02,
}

TOXIC_ELEMENTS = {"Cd", "Pb", "Tl", "Hg", "As", "Be"}
SUPPLY_CRITICAL_ELEMENTS = {"In", "Ga", "Te", "Ge", "Re", "Co"}


def shockley_queisser_efficiency(band_gap_ev: float) -> float:
    if band_gap_ev < SQ_TABLE_EV[0] or band_gap_ev > SQ_TABLE_EV[-1]:
        return 0.0
    return float(np.interp(band_gap_ev, SQ_TABLE_EV, SQ_TABLE_EFF))


def abundance_score(formula: str) -> float:
    try:
        comp = Composition(formula)
    except Exception:
        return 0.0

    elements = comp.get_el_amt_dict()
    if not elements:
        return 0.0

    scores = []
    total_amt = sum(elements.values())
    for el, amt in elements.items():
        weight = amt / total_amt
        el_score = EARTH_ABUNDANCE.get(str(el), 0.3)
        scores.append(el_score ** weight)

    return float(np.prod(scores))


def contains_toxic(formula: str) -> bool:
    try:
        comp = Composition(formula)
        return bool(set(str(e) for e in comp.elements) & TOXIC_ELEMENTS)
    except Exception:
        return False


def contains_critical(formula: str) -> bool:
    try:
        comp = Composition(formula)
        return bool(set(str(e) for e in comp.elements) & SUPPLY_CRITICAL_ELEMENTS)
    except Exception:
        return False


def solar_objectives(bg_low: float = 0.8, bg_high: float = 1.8) -> list[Objective]:
    return [
        Objective(
            name="sq_efficiency",
            direction=Direction.MAXIMISE,
            weight=1.0,
        ),
        Objective(
            name="formation_energy",
            direction=Direction.MINIMISE,
            weight=0.8,
        ),
        Objective(
            name="uncertainty",
            direction=Direction.MINIMISE,
            weight=0.5,
        ),
        Objective(
            name="abundance",
            direction=Direction.MAXIMISE,
            weight=0.3,
        ),
    ]


def solar_filter(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    mask = pd.Series(True, index=df.index)
    if "band_gap" in df.columns:
        mask &= (df["band_gap"] > 0) & (df["band_gap"] >= 0.5) & (df["band_gap"] <= 2.5)
    return df[mask].copy()
