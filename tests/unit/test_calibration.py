import numpy as np

from matscreen.uncertainty.calibration import (
    IsotonicCalibrator,
    miscalibration_area,
    reliability_diagram,
)


def _make_calibration_data(n=1000, seed=42, overconfident=False):
    rng = np.random.default_rng(seed)
    targets = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.3, n)
    predictions = targets + noise
    true_std = np.abs(noise) + 0.1
    if overconfident:
        uncertainties = true_std * 0.3
    else:
        uncertainties = true_std
    return predictions, uncertainties, targets


def test_perfect_calibration():
    rng = np.random.default_rng(42)
    targets = rng.normal(0, 1, 5000)
    uncertainties = np.ones(5000) * 1.0
    predictions = targets + rng.normal(0, 1, 5000)
    area = miscalibration_area(predictions, uncertainties, targets)
    assert area < 0.1


def test_overconfident_correction():
    preds, raw_unc, targets = _make_calibration_data(2000, overconfident=True)
    cal = IsotonicCalibrator(confidence_level=0.9)
    cal.fit(preds[:1000], raw_unc[:1000], targets[:1000])
    calibrated = cal.calibrate(raw_unc[1000:])
    assert np.mean(calibrated) > np.mean(raw_unc[1000:])


def test_reliability_diagram_shape():
    preds, unc, targets = _make_calibration_data()
    result = reliability_diagram(preds, unc, targets, n_bins=15)
    assert result["expected_coverage"].shape == (15,)
    assert result["observed_coverage"].shape == (15,)
    assert result["bin_counts"].shape == (15,)


def test_calibrator_save_load(tmp_path):
    preds, unc, targets = _make_calibration_data()
    cal = IsotonicCalibrator()
    cal.fit(preds, unc, targets)
    original = cal.calibrate(unc)

    cal.save(tmp_path / "cal.json")
    loaded = IsotonicCalibrator()
    loaded.load(tmp_path / "cal.json")
    restored = loaded.calibrate(unc)

    np.testing.assert_array_almost_equal(original, restored)


def test_miscalibration_area_bounds():
    preds, unc, targets = _make_calibration_data()
    area = miscalibration_area(preds, unc, targets)
    assert 0.0 <= area <= 0.5
