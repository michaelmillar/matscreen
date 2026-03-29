from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    MATERIALS_PROJECT = "materials_project"
    JARVIS = "jarvis"


class TriageLabel(str, Enum):
    TRUST = "trust"
    VERIFY = "verify"
    DEFER = "defer"


class SymmetryInfo(BaseModel):
    crystal_system: str | None = None
    space_group: str | None = None
    space_group_number: int | None = None


class PropertySet(BaseModel):
    band_gap: float | None = None
    band_gap_type: str | None = None
    formation_energy_per_atom: float | None = None
    energy_above_hull: float | None = None
    is_stable: bool | None = None
    bulk_modulus_kv: float | None = None
    density: float | None = None


class MaterialRecord(BaseModel):
    material_id: str
    source: DataSource
    formula: str
    structure_dict: dict[str, Any] | None = None
    symmetry: SymmetryInfo = Field(default_factory=SymmetryInfo)
    properties: PropertySet = Field(default_factory=PropertySet)


class SolarProperties(BaseModel):
    sq_efficiency: float | None = None
    abundance_score: float | None = None
    contains_toxic: bool = False
    contains_critical: bool = False


class PredictionWithCI(BaseModel):
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    unit: str
    calibrated: bool = False
    triage: TriageLabel = TriageLabel.DEFER
    ood_score: float | None = None
    in_domain: bool = True


class MaterialCard(BaseModel):
    material_id: str
    formula: str
    source: DataSource
    symmetry: SymmetryInfo
    predictions: dict[str, PredictionWithCI]
    pareto_rank: int
    pareto_front: int
    tradeoff_summary: str = ""
    triage: TriageLabel = TriageLabel.DEFER
    solar: SolarProperties | None = None
    roi_rank: int | None = None
