from __future__ import annotations

import pandas as pd


def stability_filter(df: pd.DataFrame, max_ehull: float = 0.1) -> pd.DataFrame:
    col = "energy_above_hull"
    if col not in df.columns:
        return df
    mask = df[col].isna() | (df[col] <= max_ehull)
    return df[mask].copy()


def element_filter(
    df: pd.DataFrame,
    allowed: list[str] | None = None,
    excluded: list[str] | None = None,
) -> pd.DataFrame:
    if not allowed and not excluded:
        return df

    if "elements" not in df.columns:
        return df

    mask = pd.Series(True, index=df.index)

    if excluded:
        excluded_set = set(excluded)
        mask &= df["elements"].apply(
            lambda elems: not bool(set(elems) & excluded_set) if elems else True
        )

    if allowed:
        allowed_set = set(allowed)
        mask &= df["elements"].apply(
            lambda elems: set(elems).issubset(allowed_set) if elems else False
        )

    return df[mask].copy()


def uncertainty_filter(
    df: pd.DataFrame,
    std_col: str = "bandgap_std",
    max_percentile: float = 95,
) -> pd.DataFrame:
    if std_col not in df.columns:
        return df
    threshold = df[std_col].quantile(max_percentile / 100)
    return df[df[std_col] <= threshold].copy()
