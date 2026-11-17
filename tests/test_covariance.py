"""Unit tests of src/covariance.py on small synthetic return panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.covariance import (
    investable_subspace_frame,
    ledoit_wolf_covariance,
    momentum_signal,
    rolling_covariances,
)


def _synthetic_returns(n_obs: int = 600, n_assets: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.01, size=(n_obs, 1))
    loadings = rng.normal(0.5, 0.2, size=(1, n_assets))
    idio = rng.normal(0, 0.008, size=(n_obs, n_assets))
    values = factor @ loadings + idio
    dates = pd.bdate_range("2015-01-02", periods=n_obs)
    return pd.DataFrame(values, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_ledoit_wolf_covariance_is_spd():
    returns = _synthetic_returns()
    Q = ledoit_wolf_covariance(returns.values[:504], ridge=1e-10)
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_rolling_covariances_no_lookahead():
    returns = _synthetic_returns()
    idx = np.array([504, 525])
    Qs = rolling_covariances(returns, window=504, ridge=1e-10, decision_idx=idx)
    # The covariance at t must only depend on rows [t-504, t); changing a row
    # at or after t must not change it.
    perturbed = returns.copy()
    perturbed.iloc[504:] = 0.0
    Qs_perturbed = rolling_covariances(perturbed, window=504, ridge=1e-10,
                                        decision_idx=idx)
    assert np.allclose(Qs[504], Qs_perturbed[504])


def test_investable_subspace_frame_is_q_orthonormal():
    returns = _synthetic_returns()
    Q = ledoit_wolf_covariance(returns.values[:504], ridge=1e-10)
    U = investable_subspace_frame(Q, rank=3)
    gram = U.T @ Q @ U
    assert np.allclose(gram, np.eye(3), atol=1e-8)


def test_momentum_signal_is_mean_zero_and_no_lookahead():
    returns = _synthetic_returns()
    s = momentum_signal(returns, decision_idx=300, lookback=252, gap=21, mad_k=4.0)
    assert abs(s.mean()) < 1e-10
    perturbed = returns.copy()
    perturbed.iloc[300 - 21:] = 10.0  # rows in the gap/after decision
    s_perturbed = momentum_signal(perturbed, decision_idx=300, lookback=252,
                                   gap=21, mad_k=4.0)
    assert np.allclose(s, s_perturbed)
