from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

from matscreen.data.cache import load_parquet, save_parquet

logger = logging.getLogger(__name__)


class CompositionFeaturiser:
    def __init__(self, cache_dir: Path = Path("data/processed")):
        self.cache_dir = cache_dir
        self._featuriser = ElementProperty.from_preset("magpie")

    def feature_names(self) -> list[str]:
        return self._featuriser.feature_labels()

    def featurise(
        self,
        formulae: pd.Series,
        material_ids: pd.Series,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = self.cache_dir / "magpie_features.parquet"
        if use_cache:
            cached = load_parquet(cache_path)
            if cached is not None:
                logger.info("Loaded cached features: %d rows", len(cached))
                return cached

        records = []
        failed = 0
        total = len(formulae)

        for i, (mid, formula) in enumerate(zip(material_ids, formulae)):
            if (i + 1) % 5000 == 0:
                logger.info("Featurising %d / %d", i + 1, total)
            try:
                comp = Composition(formula)
                features = self._featuriser.featurize(comp)
                records.append({"material_id": mid, **dict(zip(self.feature_names(), features))})
            except Exception:
                failed += 1
                continue

        if failed > 0:
            logger.warning("Failed to featurise %d / %d formulae", failed, total)

        result = pd.DataFrame(records)
        if len(result) > 0:
            result = result.set_index("material_id")
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.fillna(0.0)

        save_parquet(result.reset_index(), cache_path)
        return result
