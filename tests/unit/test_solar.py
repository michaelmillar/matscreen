import pandas as pd

from matscreen.screening.solar import (
    abundance_score,
    contains_critical,
    contains_toxic,
    shockley_queisser_efficiency,
    solar_filter,
)


def test_sq_efficiency_peak():
    eff = shockley_queisser_efficiency(1.34)
    assert 0.33 <= eff <= 0.34


def test_sq_efficiency_zero_for_metal():
    assert shockley_queisser_efficiency(0.0) == 0.0


def test_sq_efficiency_zero_outside_range():
    assert shockley_queisser_efficiency(5.0) == 0.0


def test_abundance_score_silicon():
    score = abundance_score("Si")
    assert score > 0.9


def test_abundance_score_indium():
    score = abundance_score("InP")
    assert score < 0.5


def test_contains_toxic_cdte():
    assert contains_toxic("CdTe") is True


def test_contains_toxic_silicon():
    assert contains_toxic("Si") is False


def test_contains_critical_ingaas():
    assert contains_critical("InGaAs") is True


def test_solar_filter_removes_metals():
    df = pd.DataFrame({
        "band_gap": [0.0, 1.2, 3.0, 0.8],
        "formula": ["Cu", "GaAs", "AlN", "CdTe"],
    })
    filtered = solar_filter(df)
    assert len(filtered) == 2
    assert 0.0 not in filtered["band_gap"].values
    assert 3.0 not in filtered["band_gap"].values
