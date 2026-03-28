import tempfile
from pathlib import Path

import pandas as pd

from matscreen.data.cache import is_stale, load_parquet, save_parquet


def test_save_and_load_roundtrip():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        save_parquet(df, path)
        loaded = load_parquet(path)
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == ["a", "b"]


def test_load_nonexistent():
    result = load_parquet("/tmp/does_not_exist_12345.parquet")
    assert result is None


def test_is_stale_nonexistent():
    assert is_stale("/tmp/does_not_exist_12345.parquet") is True


def test_is_stale_fresh():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fresh.parquet"
        df = pd.DataFrame({"a": [1]})
        save_parquet(df, path)
        assert is_stale(path, max_age_days=30) is False
