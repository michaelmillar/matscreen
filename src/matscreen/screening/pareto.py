from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(costs: np.ndarray) -> list[list[int]]:
    n = len(costs)
    domination_count = np.zeros(n, dtype=int)
    dominated_by: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(costs[i], costs[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(costs[j], costs[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(costs: np.ndarray, front: list[int]) -> np.ndarray:
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_costs = costs[front]
    n_objectives = front_costs.shape[1]

    for m in range(n_objectives):
        sorted_idx = np.argsort(front_costs[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = front_costs[sorted_idx[-1], m] - front_costs[sorted_idx[0], m]
        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                front_costs[sorted_idx[i + 1], m] - front_costs[sorted_idx[i - 1], m]
            ) / obj_range

    return distances
