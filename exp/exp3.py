"""Experiment 3: Historical Geometry Stress Test (paper Sec. 4.5).

Diagnostic run on the 48-industry daily panel checking whether the local
defect coefficient D(t) of Theorem 3.7 (Eq. 3.32) predicts the observed
separation between the risk weighted Procrustes, geometric mean and
symmetric Procrustes endpoint rules over 5-day ("5 observation") states.

For every 5-day state we estimate Qdot and Udot at the state's decision
time by OLS linear regression (slope) of the 5 preceding daily frames and
the frame at the decision time itself (6 points, gauge-aligned to a common
reference frame before differencing), compute the local defect coefficient
D(t) from these finite-difference derivatives, and separately compute the
three endpoint coefficient maps between the decision time and 5 days later
to measure how much they actually disagree. We then test whether ||D(t)||
(ex ante) predicts the observed endpoint-rule separation (ex post).

Run: python -m exp.exp3
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config as cfg
from src import geometry as geo
from src.covariance import ledoit_wolf_covariance, investable_subspace_frame
from data.data_loader import INDUSTRY_DATASET, dataset_summary, load_french_portfolios

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_daily_frames(returns: pd.DataFrame, cov_window: int, ridge: float,
                          rank: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Daily rolling Ledoit-Wolf covariance Q_t and investable subspace
    frame U_t for every valid day index t in [cov_window, len(returns)-1].

    Returns (Qs, Us, t_start) with Qs[i], Us[i] corresponding to day
    `t_start + i`, so that the frame at day `t` is `Qs[t - t_start]`.
    """
    values = returns.values
    n_days = len(values)
    t_start = cov_window
    Qs = np.empty((n_days - t_start, values.shape[1], values.shape[1]))
    Us = np.empty((n_days - t_start, values.shape[1], rank))
    for i, t in enumerate(range(t_start, n_days)):
        Q = ledoit_wolf_covariance(values[t - cov_window:t], ridge)
        Qs[i] = Q
        Us[i] = investable_subspace_frame(Q, rank)
    return Qs, Us, t_start


def build_states(n_days: int, cov_window: int, state_len: int) -> list[int]:
    """Non-overlapping 5-day state boundaries (decision times) with a full
    preceding 5-day window of daily frames and a subsequent 5-day window
    both available (Sec. 4.5)."""
    t0 = cov_window + state_len
    boundaries = []
    t = t0
    while t <= n_days - state_len - 1:
        boundaries.append(t)
        t += state_len
    return boundaries


def _ols_slope(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares slope of Y (shape (m, ...)) regressed against x (shape
    (m,)) along the first axis, equivalent to a per-entry
    np.polyfit(x, y, 1)[0] but vectorized over the trailing matrix axes."""
    xc = x - x.mean()
    denom = float(xc @ xc)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    return np.tensordot(xc, Yc, axes=(0, 0)) / denom


def state_defect_and_separation(t: int, t_start: int, Qs: np.ndarray, Us: np.ndarray,
                                 state_len: int, sv_tol: float = 1e-10) -> dict | None:
    """Ex ante defect coefficient norm and ex post endpoint-rule separation
    for the 5-day state with decision time `t` (Sec. 4.5).

    Returns None (and the state should be counted as rejected) if any
    Procrustes/geometric-mean endpoint call raises ValueError (singular
    endpoint overlap, Appendix C.1's rejection rule).
    """
    try:
        Q_t, U_t = Qs[t - t_start], Us[t - t_start]

        rel_times = np.arange(-state_len, 1, dtype=float)
        Q_window = np.empty((state_len + 1,) + Q_t.shape)
        U_window = np.empty((state_len + 1,) + U_t.shape)
        for k, i in enumerate(range(t - state_len, t)):
            Q_i, U_i = Qs[i - t_start], Us[i - t_start]
            R_i = geo.euclidean_procrustes(U_i, U_t, sv_tol=sv_tol)
            Q_window[k] = Q_i
            U_window[k] = U_i @ R_i
        Q_window[-1] = Q_t
        U_window[-1] = U_t

        Qdot = _ols_slope(rel_times, Q_window)
        Udot = _ols_slope(rel_times, U_window)
        D_t = geo.defect_coefficient(Q_t, Qdot, U_t, Udot)
        norm_D_t = float(np.linalg.norm(D_t, 2))

        t5 = t + state_len
        Q_t5, U_t5 = Qs[t5 - t_start], Us[t5 - t_start]
        R_RP = geo.risk_weighted_procrustes(Q_t, U_t, Q_t5, U_t5, sv_tol=sv_tol)
        R_geo = geo.geometric_mean_update(Q_t, U_t, Q_t5, U_t5, sv_tol=sv_tol)
        R_sym = geo.symmetric_procrustes_update(Q_t, U_t, Q_t5, U_t5, sv_tol=sv_tol)

        scale = float(state_len ** 2)
        sep_rp_geo = float(np.linalg.norm(R_RP - R_geo, 2)) / scale
        sep_rp_sym = float(np.linalg.norm(R_RP - R_sym, 2)) / scale
        sep_geo_sym = float(np.linalg.norm(R_geo - R_sym, 2)) / scale
    except ValueError:
        return None

    return {
        "t": t, "norm_D_t": norm_D_t,
        "sep_rp_geo": sep_rp_geo, "sep_rp_sym": sep_rp_sym, "sep_geo_sym": sep_geo_sym,
    }


def moving_block_bootstrap_corr(x: np.ndarray, y: np.ndarray, block: int, reps: int,
                                 seed: int = 0) -> tuple[float, float]:
    """Moving block bootstrap 95% interval for the Spearman correlation of
    paired series (x, y), resampling blocks of consecutive states."""
    n = len(x)
    if n < block + 1:
        corr = float(spearmanr(x, y).correlation)
        return corr, corr
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    boot_corrs = np.empty(reps)
    for b in range(reps):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boot_corrs[b] = spearmanr(x[idx], y[idx]).correlation
    return float(np.percentile(boot_corrs, 2.5)), float(np.percentile(boot_corrs, 97.5))


def quartile_analysis(norm_D: np.ndarray, sep_rp_geo: np.ndarray) -> tuple[list[float], float]:
    """Mean sep_rp_geo in each quartile of norm_D_t (low to high), and the
    ratio of the top quartile's mean to the bottom quartile's mean."""
    order = np.argsort(norm_D)
    chunks = np.array_split(order, 4)
    means = [float(sep_rp_geo[chunk].mean()) for chunk in chunks]
    ratio = means[-1] / means[0] if means[0] != 0 else float("inf")
    return means, ratio


def run_experiment(returns: pd.DataFrame, summary: dict) -> dict:
    t0 = time.time()
    print("Computing daily rolling frames (Step 1)...")
    Qs, Us, t_start = compute_daily_frames(
        returns, cfg.EXP3_COV_WINDOW, cfg.EXP3_RIDGE, cfg.EXP3_RANK)
    print(f"  {len(Qs)} daily frames in {time.time() - t0:.1f}s")

    boundaries = build_states(len(returns), cfg.EXP3_COV_WINDOW, cfg.EXP3_STATE_LEN)
    n_states = len(boundaries)
    print(f"Built {n_states} state boundaries "
          f"(paper reports {cfg.EXP3_N_COMPARISONS_PAPER})")

    norm_D, sep_rp_geo, sep_rp_sym, sep_geo_sym = [], [], [], []
    n_rejected = 0
    for t in boundaries:
        res = state_defect_and_separation(t, t_start, Qs, Us, cfg.EXP3_STATE_LEN)
        if res is None:
            n_rejected += 1
            continue
        norm_D.append(res["norm_D_t"])
        sep_rp_geo.append(res["sep_rp_geo"])
        sep_rp_sym.append(res["sep_rp_sym"])
        sep_geo_sym.append(res["sep_geo_sym"])

    norm_D = np.array(norm_D)
    sep_rp_geo = np.array(sep_rp_geo)
    sep_rp_sym = np.array(sep_rp_sym)
    sep_geo_sym = np.array(sep_geo_sym)
    n_valid = len(norm_D)
    print(f"Valid states: {n_valid}, rejected: {n_rejected}")

    spearman_corr = float(spearmanr(norm_D, sep_rp_geo).correlation)
    ci_low, ci_high = moving_block_bootstrap_corr(
        norm_D, sep_rp_geo, cfg.EXP3_BOOTSTRAP_BLOCK, cfg.EXP3_BOOTSTRAP_REPS, seed=0)

    quartile_means, quartile_ratio = quartile_analysis(norm_D, sep_rp_geo)
    median_ratio_sep_over_D = float(np.median(sep_rp_geo / norm_D))
    median_sep_geo_sym = float(np.median(sep_geo_sym))
    median_sep_rp_geo = float(np.median(sep_rp_geo))
    ratio_geo_sym_closer = median_sep_rp_geo / median_sep_geo_sym

    results = {
        "n_states": n_states,
        "n_valid_states": n_valid,
        "n_rejected_states": n_rejected,
        "spearman_corr": spearman_corr,
        "spearman_ci_low": ci_low,
        "spearman_ci_high": ci_high,
        "quartile_means": quartile_means,
        "quartile_ratio_top_over_bottom": quartile_ratio,
        "median_ratio_sep_over_D": median_ratio_sep_over_D,
        "median_sep_geo_sym": median_sep_geo_sym,
        "median_sep_rp_geo": median_sep_rp_geo,
        "ratio_geo_sym_closer": ratio_geo_sym_closer,
        "data_summary": summary,
    }
    return results, norm_D, sep_rp_geo


def plot_scatter(norm_D: np.ndarray, sep_rp_geo: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(norm_D, sep_rp_geo, s=8, alpha=0.4, color="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("||D(t)||_2 (ex ante local defect coefficient)")
    ax.set_ylabel("sep(R_RP, R_geo) / 25 (ex post endpoint separation)")
    ax.set_title("Exp 3: Historical Geometry Stress Test (Sec. 4.5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Market data: Kenneth R. French Data Library 48-industry panel "
          "(paper Sec. 4.5) via pandas_datareader.")

    returns = load_french_portfolios(
        INDUSTRY_DATASET, cfg.EXP4_INDUSTRIES_START, cfg.EXP4_INDUSTRIES_END)
    returns = returns.dropna(axis=1, how="any")
    summary = dataset_summary(returns)
    print("data_summary:", json.dumps(summary, indent=2))

    results, norm_D, sep_rp_geo = run_experiment(returns, summary)

    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    plot_scatter(norm_D, sep_rp_geo, RESULTS_DIR / "exp3_scatter.png")

    print(json.dumps(results, indent=2, default=float))
    print("Saved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
