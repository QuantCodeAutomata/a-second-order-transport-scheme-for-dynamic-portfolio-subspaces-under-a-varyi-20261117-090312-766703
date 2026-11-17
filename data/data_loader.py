"""Daily value-weighted portfolio returns from the Kenneth R. French Data
Library (paper Sec. 4.6 "data_requirements"): the 48 industry portfolios and
the 100 portfolios formed on the intersections of 10 size and 10
book-to-market groups.

This is not OHLCV bar data, so it is outside the scope of the QuantX
Dataloader / Massive market-data path (which serves individual-instrument
price bars): the paper's experiment 4 and 3 inputs are the academic
French Data Library factor-portfolio files, fetched through
`pandas_datareader`, which downloads the library's published zip archives
directly from Dartmouth's data library host.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web

CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore

# Dataset names as published by pandas_datareader's FamaFrench reader.
INDUSTRY_DATASET = "48_Industry_Portfolios_daily"
SIZEVALUE_DATASET = "100_Portfolios_10x10_Daily"


class FrenchDataError(RuntimeError):
    """Raised when the French Data Library download fails or is empty."""


def _cache_path(dataset: str, start: str, end: str) -> Path:
    key = hashlib.sha256(f"{dataset}|{start}|{end}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"ff_{key}.csv.gz"


def load_french_portfolios(dataset: str, start: str, end: str,
                            use_cache: bool = True) -> pd.DataFrame:
    """Daily value-weighted percentage returns for a French Data Library
    portfolio dataset (table index 0: "Average Value Weighted Returns").

    Parameters
    ----------
    dataset: one of `INDUSTRY_DATASET`, `SIZEVALUE_DATASET`.
    start, end: inclusive calendar-date bounds, "YYYY-MM-DD".

    Returns a DataFrame indexed by trading date with one column per
    portfolio, returns expressed as decimals (library values are in
    percent and are divided by 100 here), with the -99.99 / -999 missing
    codes converted to NaN per the dataset's documented convention.
    """
    cache_file = _cache_path(dataset, start, end)
    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        raw = web.DataReader(dataset, "famafrench", start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed error below
        raise FrenchDataError(f"French Data Library download failed for "
                               f"{dataset} [{start}, {end}]: {exc}") from exc

    if 0 not in raw or raw[0].empty:
        raise FrenchDataError(f"Empty value-weighted table for {dataset} "
                               f"[{start}, {end}]")
    value_weighted = raw[0].copy()
    if isinstance(value_weighted.index, pd.PeriodIndex):
        value_weighted.index = value_weighted.index.to_timestamp()
    else:
        value_weighted.index = pd.to_datetime(value_weighted.index)
    value_weighted = value_weighted.replace([-99.99, -999], pd.NA)
    returns = value_weighted.astype(float) / 100.0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_csv(cache_file)
    return returns


def dataset_summary(returns: pd.DataFrame) -> dict:
    """Realized rows, date range and universe size of a loaded panel, logged
    so the executed sample can be checked against the paper's reported
    observation counts (Sec. 4.6)."""
    return {
        "n_obs": int(len(returns)),
        "n_assets": int(returns.shape[1]),
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "n_missing": int(returns.isna().sum().sum()),
    }


if __name__ == "__main__":
    industries = load_french_portfolios(
        INDUSTRY_DATASET, "1990-01-02", "2026-04-30", use_cache=False)
    print("industries:", json.dumps(dataset_summary(industries), indent=2))
    sizevalue = load_french_portfolios(
        SIZEVALUE_DATASET, "2009-07-01", "2024-06-28", use_cache=False)
    print("size/value:", json.dumps(dataset_summary(sizevalue), indent=2))
