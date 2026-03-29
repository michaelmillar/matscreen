from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


class DomainDetector:
    def __init__(
        self,
        mahalanobis_percentile: float = 95.0,
        disagreement_factor: float = 3.0,
        n_components: int = 30,
    ):
        self.mahalanobis_percentile = mahalanobis_percentile
        self.disagreement_factor = disagreement_factor
        self.n_components = n_components
        self._pca: PCA | None = None
        self._mean: np.ndarray | None = None
        self._precision: np.ndarray | None = None
        self._distance_threshold: float = 0.0
        self._median_std: float = 0.0
        self._fitted = False

    def fit(
        self,
        train_features: np.ndarray,
        train_ensemble_stds: np.ndarray | None = None,
    ) -> None:
        n_comp = min(self.n_components, train_features.shape[1], train_features.shape[0])
        self._pca = PCA(n_components=n_comp)
        reduced = self._pca.fit_transform(train_features)

        self._mean = reduced.mean(axis=0)
        cov = np.cov(reduced, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self._precision = np.linalg.inv(cov)

        train_distances = self._mahalanobis(reduced)
        self._distance_threshold = float(
            np.percentile(train_distances, self.mahalanobis_percentile)
        )

        if train_ensemble_stds is not None:
            self._median_std = float(np.median(train_ensemble_stds))
        self._fitted = True

    def _mahalanobis(self, X_reduced: np.ndarray) -> np.ndarray:
        diff = X_reduced - self._mean
        left = diff @ self._precision
        return np.sqrt(np.sum(left * diff, axis=1))

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted or self._pca is None:
            return np.zeros(len(features))
        reduced = self._pca.transform(features)
        return self._mahalanobis(reduced)

    def is_ood(
        self,
        features: np.ndarray,
        ensemble_stds: np.ndarray | None = None,
    ) -> np.ndarray:
        distances = self.score(features)
        ood_by_distance = distances > self._distance_threshold

        if ensemble_stds is not None and self._median_std > 0:
            ood_by_disagreement = ensemble_stds > (self.disagreement_factor * self._median_std)
            return ood_by_distance | ood_by_disagreement

        return ood_by_distance

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "mahalanobis_percentile": self.mahalanobis_percentile,
            "disagreement_factor": self.disagreement_factor,
            "n_components": self.n_components,
            "distance_threshold": self._distance_threshold,
            "median_std": self._median_std,
            "fitted": self._fitted,
        }
        (path / "config.json").write_text(json.dumps(state, indent=2))
        if self._mean is not None:
            np.savez(path / "stats.npz", mean=self._mean, precision=self._precision)
        if self._pca is not None:
            with open(path / "pca.pkl", "wb") as f:
                pickle.dump(self._pca, f)

    def load(self, path: Path) -> None:
        state = json.loads((path / "config.json").read_text())
        self.mahalanobis_percentile = state["mahalanobis_percentile"]
        self.disagreement_factor = state["disagreement_factor"]
        self.n_components = state["n_components"]
        self._distance_threshold = state["distance_threshold"]
        self._median_std = state["median_std"]
        self._fitted = state["fitted"]

        stats = np.load(path / "stats.npz")
        self._mean = stats["mean"]
        self._precision = stats["precision"]
        with open(path / "pca.pkl", "rb") as f:
            self._pca = pickle.load(f)
