from __future__ import annotations

import pandas as pd
from pymatgen.core import Composition

from matscreen.data.schema import TriageLabel
from matscreen.screening.filters import element_filter, stability_filter, uncertainty_filter
from matscreen.screening.objectives import Objective
from matscreen.screening.scorer import ParetoRanker
from matscreen.screening.solar import (
    abundance_score,
    contains_critical,
    contains_toxic,
    shockley_queisser_efficiency,
)
from matscreen.uncertainty.triage import TriageAssigner

TRIAGE_PRIORITY = {TriageLabel.TRUST: 0, TriageLabel.VERIFY: 1, TriageLabel.DEFER: 2}


class ScreeningEngine:
    def __init__(
        self,
        objectives: list[Objective],
        value_columns: dict[str, str] | None = None,
        triage_assigner: TriageAssigner | None = None,
    ):
        self.objectives = objectives
        self.ranker = ParetoRanker(objectives)
        self.value_columns = value_columns or {}
        self.triage_assigner = triage_assigner

    def screen(
        self,
        df: pd.DataFrame,
        max_ehull: float = 0.1,
        excluded_elements: list[str] | None = None,
        max_uncertainty_percentile: float = 95,
        uncertainty_col: str = "bandgap_std",
        top_k: int = 20,
    ) -> pd.DataFrame:
        filtered = stability_filter(df, max_ehull=max_ehull)

        if excluded_elements and "formula" in filtered.columns and "elements" not in filtered.columns:
            def _parse_elements(formula):
                try:
                    return [str(e) for e in Composition(formula).elements]
                except Exception:
                    return []
            filtered["elements"] = filtered["formula"].apply(_parse_elements)

        filtered = element_filter(filtered, excluded=excluded_elements)
        filtered = uncertainty_filter(
            filtered,
            std_col=uncertainty_col,
            max_percentile=max_uncertainty_percentile,
        )

        if len(filtered) == 0:
            return filtered

        if "band_gap" in filtered.columns and "sq_efficiency" not in filtered.columns:
            filtered["sq_efficiency"] = filtered["band_gap"].apply(shockley_queisser_efficiency)

        if "formula" in filtered.columns:
            if "abundance_score" not in filtered.columns:
                filtered["abundance_score"] = filtered["formula"].apply(abundance_score)
            if "is_toxic" not in filtered.columns:
                filtered["is_toxic"] = filtered["formula"].apply(contains_toxic)
            if "is_critical" not in filtered.columns:
                filtered["is_critical"] = filtered["formula"].apply(contains_critical)

        ranked = self.ranker.rank(filtered, self.value_columns)

        if self.triage_assigner is not None and "triage_label" in ranked.columns:
            priority = ranked["triage_label"].map(TRIAGE_PRIORITY)
            ranked = ranked.assign(_triage_priority=priority)
            ranked = ranked.sort_values(["_triage_priority", "pareto_rank"])
            ranked = ranked.drop(columns=["_triage_priority"])

        return ranked.head(top_k)
