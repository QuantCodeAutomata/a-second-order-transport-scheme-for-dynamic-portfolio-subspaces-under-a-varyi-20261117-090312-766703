"""Rolling Ledoit-Wolf covariance, investable-subspace frame construction,
and the cross-sectional momentum signal shared by the historical
experiments (paper Appendix C.1).

# sklearn.covariance.LedoitWolf (confirmed as the library estimator the
# paper's Ledoit-Wolf covariance refers to; no portfolio-optimization
# library implements the minimum-variance-orthogonal frame M_t E_t (...)^-1/2
# of Appendix C.1, which is implemented from scratch below).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.geometry import _spd_inv_sqrt


def ledoit_wolf_covariance(window_returns: np.ndarray, ridge: float) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance of a (T, n) return window plus a
    diagonal ridge (Appendix C.1: "Q_t is the Ledoit-Wolf estimate of the
    preceding 504 observations ... plus 1e-10 I_n")."""
    lw = LedoitWolf().fit(window_returns)
    n = window_returns.shape[1]
    return lw.covariance_ + ridge * np.eye(n)


def rolling_covariances(returns: pd.DataFrame, window: int, ridge: float,
                         decision_idx: np.ndarray) -> dict[int, np.ndarray]:
    """Ledoit-Wolf covariance Q_t at each integer position in `decision_idx`,
    each estimated from the `window` observations immediately preceding
    that position (no look-ahead: row `decision_idx[k]` itself is excluded).
    """
    values = returns.values
    out = {}
    for t in decision_idx:
        assert t >= window, "decision index precedes the covariance window"
        out[int(t)] = ledoit_wolf_covariance(values[t - window:t], ridge)
    return out


def investable_subspace_frame(Q: np.ndarray, rank: int) -> np.ndarray:
    """Investable subspace frame U_t = M_t E_t (E_t^T M_t^T Q_t M_t E_t)^{-1/2}
    with M_t = Q_t^{-1} - (Q_t^{-1} 1 1^T Q_t^{-1}) / (1^T Q_t^{-1} 1)
    (Appendix C.1), E_t the leading `rank` eigenvectors of Q_t.

    M_t is the minimum-variance-orthogonal projector mapping raw asset
    coordinates to the hyperplane of fully-invested (1^T w = 1) portfolios,
    and the final whitening makes U_t Q-orthonormal (U_t^T Q_t U_t = I_rank).
    """
    n = Q.shape[0]
    ones = np.ones(n)
    Q_inv_ones = np.linalg.solve(Q, ones)
    M = np.linalg.inv(Q) - np.outer(Q_inv_ones, Q_inv_ones) / (ones @ Q_inv_ones)

    eigvals, eigvecs = np.linalg.eigh(Q)
    order = np.argsort(eigvals)[::-1]
    E = eigvecs[:, order[:rank]]

    ME = M @ E
    gram = E.T @ M.T @ Q @ M @ E
    gram = 0.5 * (gram + gram.T)
    return ME @ _spd_inv_sqrt(gram)


def momentum_signal(returns: pd.DataFrame, decision_idx: int, lookback: int,
                     gap: int, mad_k: float) -> np.ndarray:
    """Cross-sectional momentum signal at `decision_idx` (Appendix C.1):
    compound `lookback` daily returns ending `gap` observations before the
    decision, clip at the cross-sectional median +/- `mad_k` median absolute
    deviations, then remove the cross-sectional mean.
    """
    end = decision_idx - gap          # exclusive end of the compounding window
    start = end - lookback
    assert start >= 0, "momentum lookback precedes the start of the sample"
    window = returns.values[start:end]
    compounded = np.prod(1.0 + window, axis=0) - 1.0   # total compounded return

    median = np.median(compounded)
    mad = np.median(np.abs(compounded - median))
    if mad > 0:
        lower, upper = median - mad_k * mad, median + mad_k * mad
        compounded = np.clip(compounded, lower, upper)
    return compounded - compounded.mean()
