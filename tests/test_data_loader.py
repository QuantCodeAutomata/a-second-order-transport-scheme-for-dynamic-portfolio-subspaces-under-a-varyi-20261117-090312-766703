"""Tests of data/data_loader.py against a reduced slice of the Kenneth R.
French Data Library (real data, no API key required for this source)."""
from __future__ import annotations

import pytest


@pytest.mark.market_data
def test_french_industries_slice_shape(french_industries_slice):
    returns = french_industries_slice
    assert returns.shape[0] > 100
    assert returns.shape[1] == 48 or returns.shape[1] > 0
    assert returns.abs().max().max() < 1.0  # decimal returns, not percent
