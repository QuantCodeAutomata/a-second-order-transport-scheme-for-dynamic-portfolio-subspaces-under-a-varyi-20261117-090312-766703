# Repository guide

This repository reproduces four experiments from *A Second Order Transport
Scheme for Dynamic Portfolio Subspaces under a Varying Risk Metric*
(A. Rodriguez Dominguez): a manufactured connection/gauge/motion check
(Sec. 4.2), a manufactured convergence-order and local-error-coefficient
check of four discrete transport rules (Sec. 4.3-4.4), a historical geometry
stress test on the 48 industry portfolios (Sec. 4.5), and a historical
portfolio implementation comparing transport rules on turnover, certainty
equivalent and coefficient-map discrepancy (Sec. 4.6-4.7).

## Layout

- `src/` — flat library modules: `geometry.py` (connection form, transport
  rules, manufactured problems), `covariance.py` (Ledoit-Wolf estimation,
  investable subspace frame, momentum signal), `portfolio.py` (backtest
  engine, certainty equivalent, composition discrepancy), `config.py` (every
  experiment parameter as a module-level `UPPER_CASE` variable with a
  comment naming its paper section or stating that the paper does not
  specify it).
- `exp/` — one script per experiment (`exp1.py`..`exp4.py`), each runnable as
  `python -m exp.expN` from the repository root, calling only the `src/`
  functions above.
- `data/` — `data_loader.py` downloads the Kenneth R. French Data Library
  48-industry and 100 size/book-to-market daily value-weighted return series
  through `pandas_datareader` and caches them under `data/cache/` (not
  committed). This is not OHLCV bar data, so it is outside the QuantX
  Dataloader / Massive market-data path, which serves per-instrument price
  bars; experiments 1 and 2 use no market data at all (purely synthetic
  manufactured paths).
- `tests/` — pytest unit tests for `src/` (synthetic fixtures) and
  integration tests for `data/data_loader.py`, marked `market_data` and
  skipped automatically when the French Data Library cannot be reached.
- `results/` — every table, JSON metrics file and plot produced by `exp/`,
  plus `RESULTS.md`, the paper comparison. Rerunning an experiment
  overwrites its files here; the numbers and verdicts in `RESULTS.md` must
  then be checked against the new files and updated by hand.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # unit/regression tests, no network
python -m pytest                        # full suite, downloads the French library tables
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
./reproduce.sh                          # tests + all four experiments, in order
```

Experiment 4 already runs the paper's full universe (48 + 100 portfolios)
and full sample period by default; there is no reduced `--full` variant
because no reduction was needed.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  constant with an inline comment giving its meaning, unit and source
  (paper section/equation, or "not specified in the paper"). Do not
  hardcode parameters inside `exp/` or `src/` function bodies.
- Market data (when used) is loaded only through `data/data_loader.py`.
  No other module performs a network request.
- `tests/` contains every pytest file; synthetic fixtures for geometry/
  covariance tests are generated inline or in `tests/conftest.py`.
- No API keys are required by this repository: experiments 1-2 are fully
  synthetic, and the French Data Library used by experiments 3-4 is a
  public, unauthenticated download.
- After any rerun of an experiment script, the files under `results/` are
  overwritten; `results/RESULTS.md` is not regenerated automatically and
  must be checked and updated against the new files by hand.
