#!/usr/bin/env bash
# Runs the test suite and every experiment of the reproduction, in order.
set -uo pipefail

cd "$(dirname "$0")"

echo "== tests (no network) =="
python -m pytest -m "not market_data" || exit 1

echo "== exp1: manufactured connection, gauge, motion =="
python -m exp.exp1 || exit 1

echo "== exp2: manufactured convergence order and local error coefficient =="
python -m exp.exp2 || exit 1

echo "== exp3: historical geometry stress test (48 industries) =="
python -m exp.exp3 || exit 1

echo "== exp4: historical portfolio implementation (industries + size-value) =="
python -m exp.exp4 || exit 1

echo "Done. See results/ for all generated tables, metrics and plots."
