"""Historical portfolio implementation of the dynamic-subspace transport
scheme (paper Sec. 4.1, 4.6-4.7, Appendix C.1).

At each decision date the investable subspace frame U_t and covariance Q_t
are recomputed; the previous period's target weight vector is carried
forward into the new frame by one of six rules (independent local
refitting, ambient projection, Euclidean Procrustes, risk weighted
Procrustes, symmetric Procrustes, geometric mean update) and blended with
a freshly computed local momentum-tilted direction by the persistence
weight (Appendix C.1, step 7). No portfolio-optimization library
implements this subspace-transport carry-forward rule, so the backtest
loop below is a custom implementation of the paper's procedure; vectorized
helpers reuse `src.geometry`'s transport maps (Sec. 3.6) directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.covariance import investable_subspace_frame, momentum_signal, rolling_covariances
from src.geometry import (
    ambient_projection,
    euclidean_procrustes,
    geometric_mean_update,
    risk_weighted_procrustes,
    symmetric_procrustes_update,
)

METHODS = (
    "independent_local",
    "ambient_projection",
    "euclidean_procrustes",
    "risk_weighted_procrustes",
    "symmetric_procrustes",
    "geometric_mean",
)


def equal_weight_anchor(n: int) -> np.ndarray:
    """Fully-invested equal-weight anchor portfolio (methodology step 6)."""
    return np.ones(n) / n


def active_direction(U: np.ndarray, s: np.ndarray, Q: np.ndarray,
                      vol_target: float, ann_days: int) -> np.ndarray:
    """Local active direction U U^T s rescaled to `vol_target` annualized
    volatility under Q (methodology step 5)."""
    raw = U @ (U.T @ s)
    variance = raw @ Q @ raw
    ann_vol = np.sqrt(ann_days * max(variance, 0.0))
    if ann_vol < 1e-14:
        return raw
    return raw * (vol_target / ann_vol)


def radial_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Radial contraction enforcing a gross exposure cap: if the gross
    exposure sum|w| exceeds `cap`, rescale the whole vector down to the
    cap boundary (methodology step 6)."""
    gross = np.abs(w).sum()
    if gross <= cap or gross == 0.0:
        return w
    return w * (cap / gross)


def transport_target(method: str, Q0: np.ndarray, U0: np.ndarray, v0: np.ndarray,
                      Q1: np.ndarray, U1: np.ndarray, sv_tol: float) -> np.ndarray:
    """Carry the previous target `v0` (an asset-space vector represented in
    the old frame U0) into the new frame U1 by the named transport rule."""
    if method == "ambient_projection":
        return ambient_projection(U1, Q1, v0)
    c0 = U0.T @ Q0 @ v0  # frame coefficients: U0 is Q0-orthonormal
    if method == "euclidean_procrustes":
        R = euclidean_procrustes(U0, U1, sv_tol=sv_tol)
    elif method == "risk_weighted_procrustes":
        R = risk_weighted_procrustes(Q0, U0, Q1, U1, sv_tol=sv_tol)
    elif method == "symmetric_procrustes":
        R = symmetric_procrustes_update(Q0, U0, Q1, U1, sv_tol=sv_tol)
    elif method == "geometric_mean":
        R = geometric_mean_update(Q0, U0, Q1, U1, sv_tol=sv_tol)
    else:
        raise ValueError(f"Unknown transport method: {method}")
    return U1 @ (R @ c0)


def decision_indices(n_obs: int, cov_window: int, momentum_lookback: int,
                      momentum_gap: int, rebalance_freq: int) -> np.ndarray:
    """Integer row positions of rebalancing decisions: the first index has
    enough history for both the covariance window and the momentum
    compounding window, subsequent indices are spaced `rebalance_freq`
    observations apart (methodology step 8)."""
    start = max(cov_window, momentum_lookback + momentum_gap)
    return np.arange(start, n_obs, rebalance_freq)


def run_backtest(returns: pd.DataFrame, rank: int, method: str, *,
                  cov_window: int, ridge: float, momentum_lookback: int,
                  momentum_gap: int, mad_k: float, vol_target: float,
                  ann_days: int, gross_cap: float, rebalance_freq: int,
                  persistence: float, cost_bps: float, sv_tol: float) -> dict:
    """Run the full historical implementation of one transport method.

    Returns a dict with:
      'decision_idx': array of decision row positions,
      'weights': DataFrame (decision dates x assets) of target weights,
      'gross_returns': Series of daily portfolio returns before costs,
      'net_returns': Series of daily portfolio returns after proportional
          one-way costs charged on rebalancing days,
      'turnover': array of one-way turnover sum(|w_t - w_{t-1}|) per
          rebalance (first entry is turnover from the anchor-only start),
      'annualized_turnover': scalar,
      'ce_net': certainty-equivalent of the net daily return series
          (methodology step 10; CE = ann_days*mean(r) - 1.5*ann_days*var(r)
          with the 1.5 coefficient equal to risk_aversion/2 for
          risk_aversion = 3).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown transport method: {method}")
    values = returns.values
    n_obs, n_assets = values.shape
    idx = decision_indices(n_obs, cov_window, momentum_lookback, momentum_gap,
                            rebalance_freq)
    assert len(idx) >= 2, "sample too short for the requested windows"

    Qs = rolling_covariances(returns, cov_window, ridge, idx)
    anchor = equal_weight_anchor(n_assets)

    weights_rows = []
    turnovers = []
    prev_weight = anchor.copy()
    prev_Q, prev_U, prev_target = None, None, None

    for k, t in enumerate(idx):
        Q = Qs[int(t)]
        U = investable_subspace_frame(Q, rank)
        s = momentum_signal(returns, int(t), momentum_lookback, momentum_gap, mad_k)
        v_local = active_direction(U, s, Q, vol_target, ann_days)

        if method == "independent_local" or prev_target is None:
            target = v_local
        else:
            try:
                v_transported = transport_target(method, prev_Q, prev_U,
                                                   prev_target, Q, U, sv_tol)
            except ValueError:
                v_transported = v_local  # rejected singular overlap: hold local
            target = persistence * v_local + (1.0 - persistence) * v_transported

        weight = radial_cap(anchor + target, gross_cap)
        turnovers.append(float(np.abs(weight - prev_weight).sum()))
        weights_rows.append(weight)

        prev_weight, prev_Q, prev_U, prev_target = weight, Q, U, target

    weights = pd.DataFrame(weights_rows, index=returns.index[idx],
                            columns=returns.columns)
    turnover = np.array(turnovers)

    gross_returns = np.zeros(n_obs)
    cost_rate = cost_bps / 1e4
    wealth_cost_log = np.zeros(len(idx))
    for k, t in enumerate(idx):
        w = weights.values[k]
        end = idx[k + 1] if k + 1 < len(idx) else n_obs
        for day in range(t, min(end, n_obs)):
            gross_returns[day] = w @ values[day]
        wealth_cost_log[k] = np.log(max(1.0 - cost_rate * turnover[k], 1e-12))

    gross_series = pd.Series(gross_returns, index=returns.index)
    valid = gross_series.index >= returns.index[idx[0]]
    gross_valid = gross_series[valid]

    net_returns = gross_valid.copy()
    cost_days = returns.index[idx]
    for k, day in enumerate(cost_days):
        # Proportional costs deducted multiplicatively from wealth before the
        # next holding-period return (methodology step 9).
        net_returns.loc[day] = (1.0 + net_returns.loc[day]) * np.exp(wealth_cost_log[k]) - 1.0

    annualized_to = turnover[1:].mean() * (ann_days / rebalance_freq) if len(turnover) > 1 else np.nan

    return {
        "decision_idx": idx,
        "weights": weights,
        "gross_returns": gross_valid,
        "net_returns": net_returns,
        "turnover": turnover,
        "annualized_turnover": float(annualized_to),
        "ce_net": certainty_equivalent(net_returns.values, ann_days, risk_aversion=3.0),
        "ce_gross": certainty_equivalent(gross_valid.values, ann_days, risk_aversion=3.0),
    }


def certainty_equivalent(r: np.ndarray, ann_days: int, risk_aversion: float) -> float:
    """CE = ann_days * mean(r) - (risk_aversion / 2) * ann_days * var(r)
    (methodology step 10; Table 4 caption)."""
    r = np.asarray(r, dtype=float)
    return float(ann_days * r.mean() - 0.5 * risk_aversion * ann_days * r.var())


def composition_discrepancy(returns: pd.DataFrame, rank: int, method: str, *,
                             cov_window: int, ridge: float, step: int,
                             horizon: int, sv_tol: float,
                             start_sep: int = 65,
                             n_samples: int = 1000, seed: int = 0) -> np.ndarray:
    """Relative Frobenius discrepancy between the direct coefficient map over
    `horizon` observations and the composition of successive `step`-observation
    maps, with a shorter final sub-interval when `horizon` is not a multiple
    of `step` (Sec. 4.7, Appendix C: "composed at five observation intervals,
    including a shorter final interval where required").

    Starting dates are deterministically spaced `start_sep` (>= horizon)
    observations apart so that comparison windows are disjoint at both the
    21- and 63-day horizons (Sec. 4.7), capped at `n_samples` windows.
    """
    n_full = horizon // step
    remainder = horizon - n_full * step
    sub_steps = [step] * n_full + ([remainder] if remainder else [])
    n_obs = len(returns)
    low = cov_window
    high = n_obs - horizon - 1
    assert high > low, "sample too short for the requested horizon"
    starts = np.arange(low, high, start_sep)[:n_samples]

    def frame_at(t: int):
        Q = rolling_covariances(returns, cov_window, ridge, np.array([t]))[t]
        return Q, investable_subspace_frame(Q, rank)

    def coefficient_map(Q0, U0, Q1, U1):
        if method == "ambient_projection":
            # Compare the equivalent operator on frame coefficients:
            # c1 = U1^T Q1 (U1 U1^T Q1 U0) = (U1^T Q1 U1 U1^T Q1 U0); since
            # U1 is Q1-orthonormal this reduces to U1^T Q1 U0.
            return U1.T @ Q1 @ U0
        if method == "euclidean_procrustes":
            return euclidean_procrustes(U0, U1, sv_tol=sv_tol)
        if method == "risk_weighted_procrustes":
            return risk_weighted_procrustes(Q0, U0, Q1, U1, sv_tol=sv_tol)
        if method == "symmetric_procrustes":
            return symmetric_procrustes_update(Q0, U0, Q1, U1, sv_tol=sv_tol)
        if method == "geometric_mean":
            return geometric_mean_update(Q0, U0, Q1, U1, sv_tol=sv_tol)
        raise ValueError(f"Unknown transport method: {method}")

    discrepancies = []
    for start in starts:
        offsets = np.concatenate(([0], np.cumsum(sub_steps)))
        try:
            frames = [frame_at(int(start + off)) for off in offsets]
        except ValueError:
            continue
        direct = coefficient_map(frames[0][0], frames[0][1], frames[-1][0], frames[-1][1])
        composed = np.eye(rank)
        for k in range(len(sub_steps)):
            R = coefficient_map(frames[k][0], frames[k][1], frames[k + 1][0], frames[k + 1][1])
            composed = R @ composed
        denom = np.linalg.norm(direct, "fro")
        if denom < 1e-14:
            continue
        discrepancies.append(np.linalg.norm(composed - direct, "fro") / denom)
    return np.array(discrepancies)
