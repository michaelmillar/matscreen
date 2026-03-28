from __future__ import annotations

import pandas as pd

from matscreen.screening.filters import element_filter, stability_filter, uncertainty_filter
from matscreen.screening.objectives import Objective
from matscreen.screening.scorer import ParetoRanker


class ScreeningEngine:
    def __init__(
        self,
        objectives: list[Objective],
        value_columns: dict[str, str] | None = None,
    ):
        self.objectives = objectives
        self.ranker = ParetoRanker(objectives)
        self.value_columns = value_columns or {}

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
        filtered = element_filter(filtered, excluded=excluded_elements)
        filtered = uncertainty_filter(
            filtered,
            std_col=uncertainty_col,
            max_percentile=max_uncertainty_percentile,
        )

        if len(filtered) == 0:
            return filtered

        ranked = self.ranker.rank(filtered, self.value_columns)
        return ranked.head(top_k)
