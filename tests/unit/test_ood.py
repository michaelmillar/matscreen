import numpy as np

from matscreen.uncertainty.ood import DomainDetector


def _make_train_data(n=500, d=50, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, d))


def test_mahalanobis_zero_for_mean():
    X = _make_train_data()
    det = DomainDetector(n_components=10)
    det.fit(X)
    centroid = X.mean(axis=0).reshape(1, -1)
    score = det.score(centroid)
    assert score[0] < 1.0


def test_mahalanobis_increases_with_distance():
    X = _make_train_data()
    det = DomainDetector(n_components=10)
    det.fit(X)
    near = X.mean(axis=0).reshape(1, -1) + 0.1
    far = X.mean(axis=0).reshape(1, -1) + 100.0
    assert det.score(far)[0] > det.score(near)[0]


def test_ood_flag_extreme_point():
    X = _make_train_data()
    det = DomainDetector(n_components=10, mahalanobis_percentile=95.0)
    stds = np.random.default_rng(0).exponential(0.1, len(X))
    det.fit(X, stds)
    extreme = np.ones((1, 50)) * 100
    assert det.is_ood(extreme)[0] is True or det.is_ood(extreme)[0]


def test_in_domain_for_training_point():
    X = _make_train_data()
    det = DomainDetector(n_components=10, mahalanobis_percentile=95.0)
    det.fit(X)
    mid = X[len(X) // 2].reshape(1, -1)
    ood_flags = det.is_ood(mid)
    assert not all(ood_flags)


def test_save_load_roundtrip(tmp_path):
    X = _make_train_data()
    det = DomainDetector(n_components=10)
    stds = np.ones(len(X)) * 0.1
    det.fit(X, stds)
    scores_before = det.score(X[:5])

    det.save(tmp_path / "ood")
    loaded = DomainDetector()
    loaded.load(tmp_path / "ood")
    scores_after = loaded.score(X[:5])

    np.testing.assert_array_almost_equal(scores_before, scores_after)
