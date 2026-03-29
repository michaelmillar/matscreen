from __future__ import annotations

import numpy as np

from matscreen.data.schema import TriageLabel


class TriageAssigner:
    def __init__(
        self,
        trust_max_std: float = 0.10,
        verify_max_std: float = 0.25,
    ):
        self.trust_max_std = trust_max_std
        self.verify_max_std = verify_max_std

    def assign(
        self,
        calibrated_stds: np.ndarray,
        is_ood: np.ndarray,
    ) -> list[TriageLabel]:
        labels = [TriageLabel.DEFER] * len(calibrated_stds)
        for i in range(len(calibrated_stds)):
            if is_ood[i]:
                continue
            if calibrated_stds[i] <= self.trust_max_std:
                labels[i] = TriageLabel.TRUST
            elif calibrated_stds[i] <= self.verify_max_std:
                labels[i] = TriageLabel.VERIFY
        return labels

    def summary(self, labels: list[TriageLabel]) -> dict[str, int]:
        return {
            "trust": sum(1 for l in labels if l == TriageLabel.TRUST),
            "verify": sum(1 for l in labels if l == TriageLabel.VERIFY),
            "defer": sum(1 for l in labels if l == TriageLabel.DEFER),
        }
