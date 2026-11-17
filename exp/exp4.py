"""Experiment 4: Historical Portfolio Implementation and Performance
Evaluation (paper Sec. 4.1, 4.6-4.7, Appendix C).

Runs the full historical implementation of the dynamic-subspace transport
scheme on the 48 industry portfolios and the 100 size/book-to-market
portfolios of the Kenneth R. French Data Library, comparing independent
local refitting, ambient projection, Euclidean Procrustes, risk weighted
Procrustes, symmetric Procrustes and the geometric mean update on
annualized turnover and certainty equivalent (Table 4), their sensitivity
to the persistence weight and the rebalancing frequency (Sec. 4.6), and the
direct-versus-composed coefficient-map discrepancy over 21- and 63-day
horizons (Sec. 4.7).

Run: python -m exp.exp4
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

from src import config as cfg
from src.portfolio import METHODS, composition_discrepancy, run_backtest
from data.data_loader import (
    FrenchDataError,
    INDUSTRY_DATASET,
    SIZEVALUE_DATASET,
    dataset_summary,
    load_french_portfolios,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def moving_block_bootstrap_ci(series_a: np.ndarray, series_b: np.ndarray,
                               block: int, reps: int, seed: int) -> dict:
    """Moving block bootstrap 95% interval for the mean paired difference
    series_a - series_b (Sec. 4.6/Appendix C.3's statistical-inference
    protocol applied to the rebalance-indexed turnover and return series).
    """
    diff = np.asarray(series_a) - np.asarray(series_b)
    n = len(diff)
    if n < block + 1:
        return {"mean": float(diff.mean()), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_pool = np.arange(0, n - block + 1)
    boot_means = np.empty(reps)
    for b in range(reps):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        sample = np.concatenate([diff[s:s + block] for s in starts])[:n]
        boot_means[b] = sample.mean()
    return {
        "mean": float(diff.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


def run_panel(name: str, returns: pd.DataFrame, rank: int, cost_bps: float,
              summary: dict) -> dict:
    """Primary comparison of all six transport rules (methodology steps 1-10)
    at the primary persistence weight and 21-day rebalancing frequency."""
    common = dict(
        cov_window=cfg.EXP4_COV_WINDOW, ridge=cfg.EXP4_RIDGE,
        momentum_lookback=cfg.EXP4_MOMENTUM_LOOKBACK,
        momentum_gap=cfg.EXP4_MOMENTUM_GAP, mad_k=cfg.EXP4_MOMENTUM_MAD_K,
        vol_target=cfg.EXP4_ACTIVE_VOL_TARGET, ann_days=cfg.EXP4_ANNUALIZATION_DAYS,
        gross_cap=cfg.EXP4_GROSS_CAP, rebalance_freq=cfg.EXP4_REBALANCE_FREQ_PRIMARY,
        persistence=cfg.EXP4_PERSISTENCE_PRIMARY, cost_bps=cost_bps,
        sv_tol=cfg.EXP4_ENDPOINT_SV_TOL,
    )
    runs = {}
    for method in METHODS:
        t0 = time.time()
        res = run_backtest(returns, rank, method, **common)
        runs[method] = res
        print(f"[{name}] {method}: ann_turnover={res['annualized_turnover']:.4f} "
              f"ce_net(bps/yr)={1e4 * res['ce_net']:.2f} "
              f"({time.time() - t0:.1f}s)")

    table = pd.DataFrame({
        m: {
            "annualized_turnover": runs[m]["annualized_turnover"],
            "ce_net_annualized": runs[m]["ce_net"],
            "ce_gross_annualized": runs[m]["ce_gross"],
            "mean_net_return_ann": float(runs[m]["net_returns"].mean() * cfg.EXP4_ANNUALIZATION_DAYS),
            "vol_net_ann": float(runs[m]["net_returns"].std() * np.sqrt(cfg.EXP4_ANNUALIZATION_DAYS)),
        }
        for m in METHODS
    }).T
    table["turnover_reduction_vs_local_pct"] = 100.0 * (
        1.0 - table["annualized_turnover"] / table.loc["independent_local", "annualized_turnover"]
    )
    table["ce_diff_vs_geo_bps"] = 1e4 * (table["ce_net_annualized"] - table.loc["geometric_mean", "ce_net_annualized"])

    boot_turnover = moving_block_bootstrap_ci(
        runs["independent_local"]["turnover"], runs["geometric_mean"]["turnover"],
        cfg.EXP4_BOOTSTRAP_BLOCK_DECISIONS, cfg.EXP4_BOOTSTRAP_REPS, seed=0)
    net_local = runs["independent_local"]["net_returns"].values
    net_geo = runs["geometric_mean"]["net_returns"].values
    n_common = min(len(net_local), len(net_geo))
    boot_return = moving_block_bootstrap_ci(
        net_local[:n_common], net_geo[:n_common],
        cfg.EXP4_BOOTSTRAP_BLOCK_RETURNS, cfg.EXP4_BOOTSTRAP_REPS, seed=1)

    return {
        "panel": name,
        "data_summary": summary,
        "table": table,
        "runs": runs,
        "bootstrap_turnover_local_minus_geo": boot_turnover,
        "bootstrap_daily_return_local_minus_geo": boot_return,
    }


def run_sensitivity(name: str, returns: pd.DataFrame, rank: int, cost_bps: float) -> pd.DataFrame:
    """Persistence-weight and rebalancing-frequency sensitivity grid
    (methodology step 7-8/research objective 3) for independent local
    refitting versus the geometric mean update."""
    rows = []
    for persistence in cfg.EXP4_PERSISTENCE_GRID:
        for method in ("independent_local", "risk_weighted_procrustes", "geometric_mean"):
            res = run_backtest(
                returns, rank, method, cov_window=cfg.EXP4_COV_WINDOW, ridge=cfg.EXP4_RIDGE,
                momentum_lookback=cfg.EXP4_MOMENTUM_LOOKBACK, momentum_gap=cfg.EXP4_MOMENTUM_GAP,
                mad_k=cfg.EXP4_MOMENTUM_MAD_K, vol_target=cfg.EXP4_ACTIVE_VOL_TARGET,
                ann_days=cfg.EXP4_ANNUALIZATION_DAYS, gross_cap=cfg.EXP4_GROSS_CAP,
                rebalance_freq=cfg.EXP4_REBALANCE_FREQ_PRIMARY, persistence=persistence,
                cost_bps=cost_bps, sv_tol=cfg.EXP4_ENDPOINT_SV_TOL)
            rows.append({"panel": name, "grid": "persistence", "value": persistence,
                         "method": method, "annualized_turnover": res["annualized_turnover"],
                         "ce_net": res["ce_net"]})
    for freq in cfg.EXP4_REBALANCE_FREQ_SENSITIVITY:
        for method in ("independent_local", "risk_weighted_procrustes", "geometric_mean"):
            res = run_backtest(
                returns, rank, method, cov_window=cfg.EXP4_COV_WINDOW, ridge=cfg.EXP4_RIDGE,
                momentum_lookback=cfg.EXP4_MOMENTUM_LOOKBACK, momentum_gap=cfg.EXP4_MOMENTUM_GAP,
                mad_k=cfg.EXP4_MOMENTUM_MAD_K, vol_target=cfg.EXP4_ACTIVE_VOL_TARGET,
                ann_days=cfg.EXP4_ANNUALIZATION_DAYS, gross_cap=cfg.EXP4_GROSS_CAP,
                rebalance_freq=freq, persistence=cfg.EXP4_PERSISTENCE_PRIMARY,
                cost_bps=cost_bps, sv_tol=cfg.EXP4_ENDPOINT_SV_TOL)
            rows.append({"panel": name, "grid": "rebalance_freq", "value": freq,
                         "method": method, "annualized_turnover": res["annualized_turnover"],
                         "ce_net": res["ce_net"]})
    return pd.DataFrame(rows)


def run_composition(name: str, returns: pd.DataFrame, rank: int, n_samples: int) -> pd.DataFrame:
    """Direct-versus-composed coefficient-map discrepancy (Sec. 4.7) for
    ambient projection, risk weighted Procrustes, symmetric Procrustes and
    the geometric mean update, at 21- and 63-day horizons."""
    rows = []
    for horizon in cfg.EXP4_COMPOSITION_HORIZONS:
        for method in ("ambient_projection", "risk_weighted_procrustes",
                       "symmetric_procrustes", "geometric_mean"):
            disc = composition_discrepancy(
                returns, rank, method, cov_window=cfg.EXP4_COV_WINDOW, ridge=cfg.EXP4_RIDGE,
                step=cfg.EXP4_COMPOSITION_SUBINTERVAL, horizon=horizon,
                sv_tol=cfg.EXP4_ENDPOINT_SV_TOL, start_sep=cfg.EXP4_COMPOSITION_START_SEP,
                n_samples=n_samples, seed=0)
            rows.append({
                "panel": name, "horizon": horizon, "method": method,
                "n_samples": len(disc),
                "median_discrepancy_pct": 100.0 * float(np.median(disc)) if len(disc) else float("nan"),
                "mean_discrepancy_pct": 100.0 * float(np.mean(disc)) if len(disc) else float("nan"),
            })
    return pd.DataFrame(rows)


def plot_panel(name: str, runs: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, res in runs.items():
        wealth = (1.0 + res["net_returns"]).cumprod()
        axes[0].plot(wealth.index, wealth.values, label=method, linewidth=1.1)
    axes[0].set_title(f"{name}: net cumulative wealth")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("wealth (start = 1)")
    axes[0].legend(fontsize=7)

    methods = list(runs.keys())
    turnovers = [runs[m]["annualized_turnover"] for m in methods]
    axes[1].bar(methods, turnovers, color="steelblue")
    axes[1].set_title(f"{name}: annualized turnover")
    axes[1].tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Market data: Kenneth R. French Data Library (paper Sec. 4.6) via pandas_datareader.")

    industries = load_french_portfolios(
        INDUSTRY_DATASET, cfg.EXP4_INDUSTRIES_START, cfg.EXP4_INDUSTRIES_END)
    industries = industries.dropna(axis=0, how="any")
    ind_summary = dataset_summary(industries)
    print("industries:", json.dumps(ind_summary, indent=2))

    sizevalue = load_french_portfolios(
        SIZEVALUE_DATASET, cfg.EXP4_SIZEVALUE_START, cfg.EXP4_SIZEVALUE_END)
    sizevalue = sizevalue.dropna(axis=0, how="any")
    sv_summary = dataset_summary(sizevalue)
    print("size/value:", json.dumps(sv_summary, indent=2))

    panels = [
        ("industries", industries, cfg.EXP4_RANK_INDUSTRIES, cfg.EXP4_COST_INDUSTRIES_BPS, ind_summary),
        ("size_value", sizevalue, cfg.EXP4_RANK_SIZEVALUE, cfg.EXP4_COST_SIZEVALUE_BPS, sv_summary),
    ]

    all_tables = []
    bootstrap_rows = []
    sensitivity_frames = []
    composition_frames = []
    for name, returns, rank, cost_bps, summary in panels:
        result = run_panel(name, returns, rank, cost_bps, summary)
        table = result["table"].reset_index().rename(columns={"index": "method"})
        table.insert(0, "panel", name)
        all_tables.append(table)
        plot_panel(name, result["runs"], RESULTS_DIR / f"exp4_{name}_wealth_turnover.png")

        bootstrap_rows.append({
            "panel": name,
            "turnover_diff_local_minus_geo_mean": result["bootstrap_turnover_local_minus_geo"]["mean"],
            "turnover_diff_ci_low": result["bootstrap_turnover_local_minus_geo"]["ci_low"],
            "turnover_diff_ci_high": result["bootstrap_turnover_local_minus_geo"]["ci_high"],
            "daily_return_diff_local_minus_geo_mean": result["bootstrap_daily_return_local_minus_geo"]["mean"],
            "daily_return_diff_ci_low": result["bootstrap_daily_return_local_minus_geo"]["ci_low"],
            "daily_return_diff_ci_high": result["bootstrap_daily_return_local_minus_geo"]["ci_high"],
        })

        sensitivity_frames.append(run_sensitivity(name, returns, rank, cost_bps))
        composition_frames.append(
            run_composition(name, returns, rank, cfg.EXP4_COMPOSITION_REPS_RUN))

    table_all = pd.concat(all_tables, ignore_index=True)
    table_all.to_csv(RESULTS_DIR / "exp4_table4_turnover_ce.csv", index=False)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(RESULTS_DIR / "exp4_bootstrap_ci.csv", index=False)

    sensitivity_df = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity_df.to_csv(RESULTS_DIR / "exp4_sensitivity.csv", index=False)

    composition_df = pd.concat(composition_frames, ignore_index=True)
    composition_df.to_csv(RESULTS_DIR / "exp4_composition_discrepancy.csv", index=False)

    summary = {
        "data": {"industries": ind_summary, "size_value": sv_summary},
        "config": {
            "cov_window": cfg.EXP4_COV_WINDOW, "ridge": cfg.EXP4_RIDGE,
            "rank_industries": cfg.EXP4_RANK_INDUSTRIES, "rank_sizevalue": cfg.EXP4_RANK_SIZEVALUE,
            "persistence_primary": cfg.EXP4_PERSISTENCE_PRIMARY,
            "rebalance_freq_primary": cfg.EXP4_REBALANCE_FREQ_PRIMARY,
            "composition_reps_run": cfg.EXP4_COMPOSITION_REPS_RUN,
            "composition_reps_paper": cfg.EXP4_COMPOSITION_REPS_PAPER,
        },
        "table4": table_all.to_dict(orient="records"),
        "bootstrap": bootstrap_rows,
    }
    (RESULTS_DIR / "exp4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(table_all.to_string(index=False))
    print("Saved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
