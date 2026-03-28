from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    MINIMISE = "minimise"
    MAXIMISE = "maximise"
    TARGET_RANGE = "target_range"


@dataclass(frozen=True)
class Objective:
    name: str
    direction: Direction
    weight: float = 1.0
    target_low: float | None = None
    target_high: float | None = None

    def score(self, value: float) -> float:
        if self.direction == Direction.TARGET_RANGE:
            if self.target_low is None or self.target_high is None:
                raise ValueError(f"target_low and target_high required for {self.name}")
            if self.target_low <= value <= self.target_high:
                return 0.0
            if value < self.target_low:
                return self.target_low - value
            return value - self.target_high

        if self.direction == Direction.MINIMISE:
            return value

        return -value


def bandgap_objective(low: float, high: float, weight: float = 1.0) -> Objective:
    return Objective(
        name="bandgap",
        direction=Direction.TARGET_RANGE,
        weight=weight,
        target_low=low,
        target_high=high,
    )


def stability_objective(weight: float = 0.8) -> Objective:
    return Objective(
        name="formation_energy",
        direction=Direction.MINIMISE,
        weight=weight,
    )


def uncertainty_objective(weight: float = 0.5) -> Objective:
    return Objective(
        name="uncertainty",
        direction=Direction.MINIMISE,
        weight=weight,
    )
