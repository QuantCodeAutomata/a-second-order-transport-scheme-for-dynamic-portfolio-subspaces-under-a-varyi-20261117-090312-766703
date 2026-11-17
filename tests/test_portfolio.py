"""Unit tests of src/portfolio.py on a small synthetic return panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import (
    active_direction,
    certainty_equivalent,
    decision_indices,
    radial_cap,
    run_backtest,
)
from src.covariance import investable_subspace_frame, ledoit_wolf_covariance


def _synthetic_returns(n_obs: int = 700, n_assets: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0002, 0.01, size=(n_obs, 1))
    loadings = rng.normal(0.5, 0.2, size=(1, n_assets))
    idio = rng.normal(0, 0.008, size=(n_obs, n_assets))
    values = factor @ loadings + idio
    dates = pd.bdate_range("2015-01-02", periods=n_obs)
    return pd.DataFrame(values, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_radial_cap_respects_gross_limit():
    w = np.array([1.5, -1.5, 0.3])
    capped = radial_cap(w, cap=2.0)
    assert np.abs(capped).sum() <= 2.0 + 1e-10
    w_small = np.array([0.1, -0.1])
    assert np.allclose(radial_cap(w_small, cap=2.0), w_small)


def test_active_direction_hits_vol_target():
    returns = _synthetic_returns()
    Q = ledoit_wolf_covariance(returns.values[:504], ridge=1e-10)
    U = investable_subspace_frame(Q, rank=3)
    s = np.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1, 0.2, -0.05])
    direction = active_direction(U, s, Q, vol_target=0.05, ann_days=252)
    realized_vol = np.sqrt(252 * direction @ Q @ direction)
    assert abs(realized_vol - 0.05) < 1e-6


def test_decision_indices_spacing_and_start():
    idx = decision_indices(n_obs=1000, cov_window=504, momentum_lookback=252,
                            momentum_gap=21, rebalance_freq=21)
    assert idx[0] == 504
    assert np.all(np.diff(idx) == 21)


def test_certainty_equivalent_penalizes_variance():
    r_low_vol = np.full(252, 0.0005)
    r_high_vol = 0.0005 + 0.02 * np.sin(np.arange(252))
    ce_low = certainty_equivalent(r_low_vol, 252, risk_aversion=3.0)
    ce_high = certainty_equivalent(r_high_vol, 252, risk_aversion=3.0)
    assert ce_low > ce_high


def test_run_backtest_geometric_mean_reduces_turnover_vs_independent_local():
    """Qualitative property emphasized by the paper (Sec. 4.6): carrying the
    previous target forward by a transport rule should not increase turnover
    relative to refitting it from scratch every period, on average."""
    returns = _synthetic_returns(n_obs=900, n_assets=8)
    common = dict(cov_window=252, ridge=1e-10, momentum_lookback=126,
                   momentum_gap=5, mad_k=4.0, vol_target=0.05, ann_days=252,
                   gross_cap=2.0, rebalance_freq=10, persistence=0.25,
                   cost_bps=10.0, sv_tol=1e-10)
    local = run_backtest(returns, rank=3, method="independent_local", **common)
    geo_mean = run_backtest(returns, rank=3, method="geometric_mean", **common)
    assert geo_mean["annualized_turnover"] < local["annualized_turnover"]
    assert geo_mean["weights"].shape == local["weights"].shape
    assert np.all(geo_mean["weights"].abs().sum(axis=1) <= 2.0 + 1e-8)


def test_run_backtest_rejects_unknown_method():
    returns = _synthetic_returns(n_obs=300, n_assets=4)
    try:
        run_backtest(returns, rank=2, method="bogus", cov_window=100, ridge=1e-10,
                     momentum_lookback=50, momentum_gap=5, mad_k=4.0,
                     vol_target=0.05, ann_days=252, gross_cap=2.0,
                     rebalance_freq=10, persistence=0.25, cost_bps=10.0,
                     sv_tol=1e-10)
        assert False, "expected ValueError"
    except ValueError:
        pass
