import pandas as pd

from matscreen.features.composition import CompositionFeaturiser


def test_featurise_known_material():
    feat = CompositionFeaturiser()
    formulae = pd.Series(["Si", "GaAs"])
    ids = pd.Series(["mp-149", "mp-2534"])
    result = feat.featurise(formulae, ids, use_cache=False)
    assert len(result) == 2
    assert len(result.columns) == len(feat.feature_names())
    assert result.isna().sum().sum() == 0


def test_featurise_invalid_formula():
    feat = CompositionFeaturiser()
    formulae = pd.Series(["Si", "???invalid???", "GaAs"])
    ids = pd.Series(["a", "b", "c"])
    result = feat.featurise(formulae, ids, use_cache=False)
    assert len(result) == 2


def test_feature_names_length():
    feat = CompositionFeaturiser()
    names = feat.feature_names()
    assert len(names) == 132


def test_featurise_batch_consistency():
    feat = CompositionFeaturiser()
    formulae = pd.Series(["CdTe", "ZnO", "TiO2"])
    ids = pd.Series(["a", "b", "c"])
    r1 = feat.featurise(formulae, ids, use_cache=False)
    r2 = feat.featurise(formulae, ids, use_cache=False)
    pd.testing.assert_frame_equal(r1, r2)
