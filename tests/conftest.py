"""Shared pytest fixtures.

Registers the `market_data` marker used by tests that load a reduced slice
of the Kenneth R. French Data Library through `data/data_loader.py`. These
tests depend on a live download of the library's published archives (no API
key is required for this data source, unlike the QuantX Dataloader / Massive
market-data path) and are skipped when that download is unavailable.
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test needs a live download of market data"
    )


@pytest.fixture(scope="session")
def french_industries_slice():
    """A short, cached slice of the 48-industry daily value-weighted
    portfolio returns, for tests that need real French Data Library data
    without downloading the full 1990-2026 sample."""
    from data.data_loader import INDUSTRY_DATASET, load_french_portfolios

    try:
        returns = load_french_portfolios(
            INDUSTRY_DATASET, "2015-01-02", "2017-12-29", use_cache=True
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"French Data Library unavailable: {exc}")
    if returns.empty:
        pytest.skip("French Data Library returned an empty panel")
    return returns.dropna(axis=1, how="any")
