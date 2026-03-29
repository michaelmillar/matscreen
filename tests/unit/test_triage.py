import numpy as np

from matscreen.data.schema import TriageLabel
from matscreen.uncertainty.triage import TriageAssigner


def test_trust_assignment():
    assigner = TriageAssigner(trust_max_std=0.10, verify_max_std=0.25)
    stds = np.array([0.05])
    ood = np.array([False])
    labels = assigner.assign(stds, ood)
    assert labels[0] == TriageLabel.TRUST


def test_verify_assignment():
    assigner = TriageAssigner(trust_max_std=0.10, verify_max_std=0.25)
    stds = np.array([0.15])
    ood = np.array([False])
    labels = assigner.assign(stds, ood)
    assert labels[0] == TriageLabel.VERIFY


def test_defer_ood():
    assigner = TriageAssigner()
    stds = np.array([0.01])
    ood = np.array([True])
    labels = assigner.assign(stds, ood)
    assert labels[0] == TriageLabel.DEFER


def test_defer_high_uncertainty():
    assigner = TriageAssigner(trust_max_std=0.10, verify_max_std=0.25)
    stds = np.array([0.50])
    ood = np.array([False])
    labels = assigner.assign(stds, ood)
    assert labels[0] == TriageLabel.DEFER


def test_summary_counts():
    assigner = TriageAssigner(trust_max_std=0.10, verify_max_std=0.25)
    stds = np.array([0.05, 0.05, 0.15, 0.15, 0.50])
    ood = np.array([False, False, False, False, False])
    labels = assigner.assign(stds, ood)
    summary = assigner.summary(labels)
    assert summary["trust"] == 2
    assert summary["verify"] == 2
    assert summary["defer"] == 1
    assert summary["trust"] + summary["verify"] + summary["defer"] == 5
