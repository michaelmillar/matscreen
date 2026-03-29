from __future__ import annotations

from matscreen.data.schema import TriageLabel


def candidates_per_dft_avoided(
    triage_labels: list[TriageLabel],
    is_true_positive: list[bool],
    naive_top_k: int,
) -> dict[str, float]:
    triage_tp = 0
    triage_dft = 0

    for label, tp in zip(triage_labels, is_true_positive):
        if label == TriageLabel.TRUST:
            if tp:
                triage_tp += 1
        elif label == TriageLabel.VERIFY:
            triage_dft += 1
            if tp:
                triage_tp += 1

    naive_tp = sum(is_true_positive[:naive_top_k])
    naive_dft = naive_top_k

    triage_efficiency = triage_tp / max(triage_dft, 1)
    naive_efficiency = naive_tp / max(naive_dft, 1)

    return {
        "triage_true_positives": float(triage_tp),
        "triage_dft_calls": float(triage_dft),
        "naive_true_positives": float(naive_tp),
        "naive_dft_calls": float(naive_dft),
        "triage_efficiency": triage_efficiency,
        "naive_efficiency": naive_efficiency,
        "roi_ratio": triage_efficiency / max(naive_efficiency, 1e-10),
    }
