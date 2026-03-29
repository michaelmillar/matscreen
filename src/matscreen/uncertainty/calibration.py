from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self, confidence_level: float = 0.9):
        self.confidence_level = confidence_level
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
        self._scale_factor = 1.0

    def fit(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        z_scores = np.abs(targets - predictions) / np.maximum(uncertainties, 1e-10)
        quantile_levels = np.linspace(0.05, 0.95, 19)
        expected_z = norm.ppf((1 + quantile_levels) / 2)

        observed_coverage = np.array([
            np.mean(z_scores <= z) for z in expected_z
        ])

        self._iso.fit(quantile_levels, observed_coverage)

        target_z = norm.ppf((1 + self.confidence_level) / 2)
        raw_coverage = np.mean(z_scores <= target_z)
        if raw_coverage > 0:
            self._scale_factor = float(self.confidence_level / raw_coverage)
        else:
            self._scale_factor = 2.0

        self._fitted = True

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return uncertainties
        return uncertainties * self._scale_factor

    def prediction_interval(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calibrated = self.calibrate(uncertainties)
        z = norm.ppf((1 + self.confidence_level) / 2)
        lower = predictions - z * calibrated
        upper = predictions + z * calibrated
        return lower, upper

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "confidence_level": self.confidence_level,
            "scale_factor": self._scale_factor,
            "fitted": self._fitted,
        }
        path.write_text(json.dumps(state, indent=2))

    def load(self, path: Path) -> None:
        state = json.loads(path.read_text())
        self.confidence_level = state["confidence_level"]
        self._scale_factor = state["scale_factor"]
        self._fitted = state["fitted"]


def reliability_diagram(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    z_scores = np.abs(targets - predictions) / np.maximum(uncertainties, 1e-10)
    expected_coverage = np.linspace(1.0 / n_bins, 1.0, n_bins)
    observed_coverage = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i, level in enumerate(expected_coverage):
        z_threshold = norm.ppf((1 + level) / 2)
        within = z_scores <= z_threshold
        observed_coverage[i] = np.mean(within)
        bin_counts[i] = int(np.sum(within))

    return {
        "expected_coverage": expected_coverage,
        "observed_coverage": observed_coverage,
        "bin_counts": bin_counts,
    }


def miscalibration_area(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 100,
) -> float:
    diag = reliability_diagram(predictions, uncertainties, targets, n_bins)
    diff = np.abs(diag["observed_coverage"] - diag["expected_coverage"])
    return float(np.mean(diff))
