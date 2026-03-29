import numpy as np
import pandas as pd

from matscreen.models.xgboost_ensemble import XGBoostEnsemble


def _make_data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(10)})
    y = X["f0"] * 2 + X["f1"] + rng.normal(0, 0.1, n)
    return X, y


def test_train_and_predict_shapes():
    X, y = _make_data()
    model = XGBoostEnsemble(n_models=3, seeds=[1, 2, 3])
    model.train(X, y)
    means, stds = model.predict(X)
    assert means.shape == (100,)
    assert stds.shape == (100,)


def test_ensemble_std_nonzero():
    X, y = _make_data()
    model = XGBoostEnsemble(n_models=3, seeds=[1, 2, 3])
    model.train(X, y)
    _, stds = model.predict(X)
    assert stds.sum() > 0


def test_predict_all_shape():
    X, y = _make_data()
    model = XGBoostEnsemble(n_models=5)
    model.train(X, y)
    all_preds = model.predict_all(X)
    assert all_preds.shape == (100, 5)


def test_save_and_load_roundtrip(tmp_path):
    X, y = _make_data()
    model = XGBoostEnsemble(n_models=3, seeds=[1, 2, 3])
    model.train(X, y)
    preds_before, _ = model.predict(X)

    model.save(tmp_path / "model")

    loaded = XGBoostEnsemble()
    loaded.load(tmp_path / "model")
    preds_after, _ = loaded.predict(X)

    np.testing.assert_array_almost_equal(preds_before, preds_after)


def test_val_metrics_returned():
    X, y = _make_data(200)
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]
    model = XGBoostEnsemble(n_models=2, seeds=[1, 2])
    metrics = model.train(X_train, y_train, X_val, y_val)
    assert "val_mae" in metrics
    assert "val_rmse" in metrics
    assert "train_mae" in metrics


def test_name_property():
    model = XGBoostEnsemble()
    assert model.name == "xgboost_ensemble"
