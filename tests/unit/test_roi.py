from matscreen.data.schema import TriageLabel
from matscreen.evaluation.roi import candidates_per_dft_avoided


def test_roi_perfect_triage():
    labels = [TriageLabel.TRUST] * 5 + [TriageLabel.VERIFY] * 3 + [TriageLabel.DEFER] * 2
    is_tp = [True] * 5 + [True, False, False] + [False, False]
    result = candidates_per_dft_avoided(labels, is_tp, naive_top_k=10)
    assert result["triage_true_positives"] == 6
    assert result["triage_dft_calls"] == 3


def test_roi_naive_baseline():
    labels = [TriageLabel.TRUST] * 10
    is_tp = [True, True, False, False, False, False, False, False, False, False]
    result = candidates_per_dft_avoided(labels, is_tp, naive_top_k=10)
    assert result["naive_true_positives"] == 2
    assert result["naive_dft_calls"] == 10


def test_roi_all_defer():
    labels = [TriageLabel.DEFER] * 5
    is_tp = [True] * 5
    result = candidates_per_dft_avoided(labels, is_tp, naive_top_k=5)
    assert result["triage_true_positives"] == 0
    assert result["triage_dft_calls"] == 0
