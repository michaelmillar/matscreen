from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DEFAULT_SEEDS = [42, 137, 256, 512, 1024]

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
}


class XGBoostEnsemble:
    def __init__(
        self,
        n_models: int = 5,
        seeds: list[int] | None = None,
        xgb_params: dict[str, Any] | None = None,
    ):
        self.n_models = n_models
        self.seeds = (seeds or DEFAULT_SEEDS)[:n_models]
        self.xgb_params = xgb_params or DEFAULT_XGB_PARAMS
        self.members: list[XGBRegressor] = []
        self._feature_names: list[str] = []
        self._metrics: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "xgboost_ensemble"

    def train(
        self,
        train_inputs: pd.DataFrame,
        train_targets: np.ndarray,
        val_inputs: pd.DataFrame | None = None,
        val_targets: np.ndarray | None = None,
    ) -> dict[str, float]:
        self._feature_names = list(train_inputs.columns)
        self.members = []
        all_train_preds = np.zeros((len(train_inputs), self.n_models))
        all_val_preds = None
        if val_inputs is not None:
            all_val_preds = np.zeros((len(val_inputs), self.n_models))

        for i, seed in enumerate(self.seeds):
            params = {**self.xgb_params, "random_state": seed}
            model = XGBRegressor(**params)

            fit_kwargs: dict[str, Any] = {}
            if val_inputs is not None and val_targets is not None:
                fit_kwargs["eval_set"] = [(val_inputs, val_targets)]
                fit_kwargs["verbose"] = False

            model.fit(train_inputs, train_targets, **fit_kwargs)
            self.members.append(model)

            all_train_preds[:, i] = model.predict(train_inputs)
            if val_inputs is not None and all_val_preds is not None:
                all_val_preds[:, i] = model.predict(val_inputs)

        train_mean = all_train_preds.mean(axis=1)
        self._metrics = {
            "train_mae": float(np.mean(np.abs(train_mean - train_targets))),
            "train_rmse": float(np.sqrt(np.mean((train_mean - train_targets) ** 2))),
        }

        if val_targets is not None and all_val_preds is not None:
            val_mean = all_val_preds.mean(axis=1)
            self._metrics["val_mae"] = float(np.mean(np.abs(val_mean - val_targets)))
            self._metrics["val_rmse"] = float(np.sqrt(np.mean((val_mean - val_targets) ** 2)))

        return self._metrics

    def predict(self, inputs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        all_preds = self.predict_all(inputs)
        return all_preds.mean(axis=1), all_preds.std(axis=1)

    def predict_all(self, inputs: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(inputs), len(self.members)))
        for i, model in enumerate(self.members):
            result[:, i] = model.predict(inputs)
        return result

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.members):
            model.save_model(str(path / f"member_{i}.json"))

        metadata = {
            "n_models": self.n_models,
            "seeds": self.seeds,
            "xgb_params": self.xgb_params,
            "feature_names": self._feature_names,
            "metrics": self._metrics,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def load(self, path: Path) -> None:
        metadata = json.loads((path / "metadata.json").read_text())
        self.n_models = metadata["n_models"]
        self.seeds = metadata["seeds"]
        self.xgb_params = metadata["xgb_params"]
        self._feature_names = metadata["feature_names"]
        self._metrics = metadata.get("metrics", {})

        self.members = []
        for i in range(self.n_models):
            model = XGBRegressor()
            model.load_model(str(path / f"member_{i}.json"))
            self.members.append(model)
