import pandas as pd

from matscreen.screening.filters import stability_filter, uncertainty_filter


def test_stability_filter(tiny_materials_df):
    filtered = stability_filter(tiny_materials_df, max_ehull=0.01)
    assert len(filtered) == 9
    assert "CdTe" not in filtered["formula"].values


def test_stability_filter_permissive(tiny_materials_df):
    filtered = stability_filter(tiny_materials_df, max_ehull=0.1)
    assert len(filtered) == 10


def test_uncertainty_filter(tiny_materials_df):
    filtered = uncertainty_filter(tiny_materials_df, std_col="bandgap_std", max_percentile=50)
    assert len(filtered) <= 6


def test_stability_filter_missing_column():
    df = pd.DataFrame({"formula": ["Si", "Ge"], "band_gap": [1.1, 0.7]})
    filtered = stability_filter(df, max_ehull=0.1)
    assert len(filtered) == 2
