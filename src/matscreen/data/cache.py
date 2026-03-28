from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def is_stale(path: str | Path, max_age_days: int = 30) -> bool:
    path = Path(path)
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(tz=timezone.utc) - mtime
    return age.days > max_age_days
