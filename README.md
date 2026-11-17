# A Second Order Transport Scheme for Dynamic Portfolio Subspaces under a Varying Risk Metric

Reproduction of four experiments from *A Second Order Transport Scheme for
Dynamic Portfolio Subspaces under a Varying Risk Metric* (A. Rodriguez
Dominguez).

The paper studies the transport of a low-rank investable subspace through
time when the risk metric (covariance) defining its inner product changes.
It derives a continuous parallel-transport connection for the subspace
frame, a first-order (risk weighted Procrustes) and several second-order
(principal square root map, geometric mean update, symmetric Procrustes)
discrete endpoint transport rules, proves their convergence orders and the
local error coefficient of the first-order rule, and evaluates the economic
effect of these rules on real portfolio implementations.

## Repository layout

```
src/            connection/transport geometry, covariance estimation, portfolio backtest engine, config
exp/            one script per experiment, run as `python -m exp.expN`
data/           data/data_loader.py: Kenneth R. French Data Library loader (used by exp_3, exp_4)
tests/          pytest unit and regression tests
results/        all result tables, JSON metrics and plots written by the experiment scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API key is required: experiments 1 and 2 use purely synthetic manufactured
paths, and experiments 3 and 4 download the 48 industry and the 100
size/book-to-market value-weighted daily portfolio return series from the
public Kenneth R. French Data Library through `pandas_datareader` (cached
under `data/cache/`, never committed).

## Running the experiments

```bash
python -m exp.exp1   # Manufactured connection, gauge and motion verification (Sec. 4.2)
python -m exp.exp2   # Manufactured convergence order and local error coefficient (Sec. 4.3-4.4)
python -m exp.exp3   # Historical geometry stress test, 48 industries (Sec. 4.5)
python -m exp.exp4   # Historical portfolio implementation, 48 industries + 100 size-value (Sec. 4.6-4.7)
```

Each script writes its outputs directly into `results/` (JSON metrics, CSV
tables, PNG figures) and overwrites them on a rerun; `results/RESULTS.md`
must be checked against the new files and updated by hand after a rerun.

## Tests

```bash
python -m pytest -m "not market_data"   # unit/regression tests, no network access
python -m pytest                        # also runs the French-library data tests
```

## Reproduce everything

```bash
./reproduce.sh
```

runs the test suite and all four experiments in sequence.

## Notes on experiment 4 scale

Experiment 4 uses the full paper universe (48 industry portfolios and 100
size/book-to-market portfolios) and the full paper sample period in every
run: the French Data Library portfolio files are two small, already
aggregated CSV-like tables (not per-instrument OHLCV downloads), so no
universe or period reduction was necessary to stay within the time
available for this replication.
