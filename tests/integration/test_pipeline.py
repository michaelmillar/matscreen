import numpy as np
import pandas as pd

from matscreen.features.composition import CompositionFeaturiser
from matscreen.models.xgboost_ensemble import XGBoostEnsemble
from matscreen.screening.engine import ScreeningEngine
from matscreen.screening.solar import solar_objectives
from matscreen.uncertainty.calibration import IsotonicCalibrator, miscalibration_area
from matscreen.uncertainty.ood import DomainDetector
from matscreen.uncertainty.triage import TriageAssigner


def _make_materials(n=80, seed=42):
    rng = np.random.default_rng(seed)
    formulas = rng.choice(
        ["Si", "GaAs", "CdTe", "ZnO", "TiO2", "Cu2O", "SnO2", "InP", "GaN", "MgO"],
        size=n,
    )
    return pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(n)],
        "formula": formulas,
        "band_gap": rng.uniform(0.5, 3.0, n),
        "formation_energy_per_atom": rng.normal(-1.0, 0.5, n),
        "energy_above_hull": np.abs(rng.exponential(0.03, n)),
    })


def test_full_pipeline(tmp_path):
    data = _make_materials(80)

    feat = CompositionFeaturiser(cache_dir=tmp_path / "features")
    features = feat.featurise(data["formula"], data["material_id"], use_cache=False)
    assert len(features) == 80

    common = features.index.intersection(data.set_index("material_id").index)
    X = features.loc[common]
    y = data.set_index("material_id").loc[common, "band_gap"].values

    X_train, X_cal = X[:60], X[60:]
    y_train, y_cal = y[:60], y[60:]

    ensemble = XGBoostEnsemble(n_models=2, seeds=[1, 2])
    metrics = ensemble.train(X_train, y_train)
    assert "train_mae" in metrics

    ensemble.save(tmp_path / "model")
    loaded = XGBoostEnsemble()
    loaded.load(tmp_path / "model")

    cal_means, cal_stds = loaded.predict(X_cal)
    calibrator = IsotonicCalibrator()
    calibrator.fit(cal_means, cal_stds, y_cal)
    calibrator.save(tmp_path / "cal.json")

    calibrated = calibrator.calibrate(cal_stds)
    assert calibrated.shape == cal_stds.shape

    detector = DomainDetector(n_components=5)
    _, train_stds = loaded.predict(X_train)
    detector.fit(X_train.values, train_stds)
    detector.save(tmp_path / "ood")

    all_means, all_stds = loaded.predict(X)
    all_calibrated = calibrator.calibrate(all_stds)
    ood_flags = detector.is_ood(X.values, all_stds)

    assigner = TriageAssigner()
    labels = assigner.assign(all_calibrated, ood_flags)
    summary = assigner.summary(labels)
    assert summary["trust"] + summary["verify"] + summary["defer"] == len(X)

    screening_df = data.set_index("material_id").loc[common].reset_index()
    screening_df["bandgap_std"] = all_calibrated
    screening_df["triage_label"] = [l.value for l in labels]

    objectives = solar_objectives()
    value_columns = {
        "sq_efficiency": "sq_efficiency",
        "formation_energy": "formation_energy_per_atom",
        "uncertainty": "bandgap_std",
        "abundance": "abundance_score",
    }
    engine = ScreeningEngine(
        objectives=objectives,
        value_columns=value_columns,
        triage_assigner=assigner,
    )
    results = engine.screen(screening_df, max_ehull=0.1, top_k=10)
    assert len(results) <= 10
    assert "pareto_rank" in results.columns
    assert "sq_efficiency" in results.columns
    assert "triage_label" in results.columns


def test_calibration_improves_coverage():
    rng = np.random.default_rng(99)
    n = 500
    targets = rng.normal(0, 1, n)
    predictions = targets + rng.normal(0, 0.5, n)
    raw_stds = np.abs(rng.normal(0.2, 0.1, n)) + 0.05

    cal = IsotonicCalibrator(confidence_level=0.9)
    cal.fit(predictions[:300], raw_stds[:300], targets[:300])

    calibrated = cal.calibrate(raw_stds[300:])
    miscal = miscalibration_area(predictions[300:], calibrated, targets[300:])
    assert miscal < 0.2


def test_ood_detector_flags_novel_chemistry():
    rng = np.random.default_rng(42)
    train_features = rng.normal(0, 1, (200, 20))
    detector = DomainDetector(n_components=10, mahalanobis_percentile=95)
    detector.fit(train_features)

    novel = rng.normal(50, 1, (5, 20))
    flags = detector.is_ood(novel)
    assert all(flags)
