from __future__ import annotations

import numpy as np
import pandas as pd

from matscreen.screening.objectives import Objective
from matscreen.screening.pareto import crowding_distance, non_dominated_sort


class ParetoRanker:
    def __init__(self, objectives: list[Objective]):
        self.objectives = objectives

    def rank(self, df: pd.DataFrame, value_columns: dict[str, str]) -> pd.DataFrame:
        result = df.copy()

        cost_matrix = np.zeros((len(df), len(self.objectives)))
        for i, obj in enumerate(self.objectives):
            col = value_columns[obj.name]
            values = df[col].values.astype(float)
            scores = np.array([obj.score(v) for v in values])
            cost_matrix[:, i] = scores * obj.weight

        fronts = non_dominated_sort(cost_matrix)

        ranks = np.zeros(len(df), dtype=int)
        front_labels = np.zeros(len(df), dtype=int)
        current_rank = 1

        for front_idx, front in enumerate(fronts):
            distances = crowding_distance(cost_matrix, front)
            sorted_within = np.argsort(-distances)

            for pos in sorted_within:
                original_idx = front[pos]
                ranks[original_idx] = current_rank
                front_labels[original_idx] = front_idx
                current_rank += 1

        result["pareto_rank"] = ranks
        result["pareto_front"] = front_labels
        return result.sort_values("pareto_rank")


class WeightedScorer:
    def __init__(self, objectives: list[Objective]):
        self.objectives = objectives

    def rank(self, df: pd.DataFrame, value_columns: dict[str, str]) -> pd.DataFrame:
        result = df.copy()
        total_score = np.zeros(len(df))

        for obj in self.objectives:
            col = value_columns[obj.name]
            values = df[col].values.astype(float)
            normalised = (values - values.min()) / (values.ptp() + 1e-10)
            scores = np.array([obj.score(v) for v in normalised])
            total_score += scores * obj.weight

        result["weighted_score"] = total_score
        result["pareto_rank"] = result["weighted_score"].rank(method="min").astype(int)
        result["pareto_front"] = 0
        return result.sort_values("weighted_score")
