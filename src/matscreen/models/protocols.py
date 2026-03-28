from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class ForwardModel(Protocol):
    def train(
        self,
        train_inputs: pd.DataFrame,
        train_targets: np.ndarray,
        val_inputs: pd.DataFrame | None = None,
        val_targets: np.ndarray | None = None,
    ) -> dict[str, float]: ...

    def predict(self, inputs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    @property
    def name(self) -> str: ...
