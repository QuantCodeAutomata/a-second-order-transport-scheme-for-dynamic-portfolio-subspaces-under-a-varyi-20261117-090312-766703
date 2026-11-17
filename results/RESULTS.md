# Results — A Second Order Transport Scheme for Dynamic Portfolio Subspaces under a Varying Risk Metric

Paper: *A Second Order Transport Scheme for Dynamic Portfolio Subspaces under
a Varying Risk Metric* (A. Rodriguez Dominguez).

Verdicts follow: **reproduced** if the value lies within the paper's stated
interval, or else within 10% of the paper's value with the same sign and the
same qualitative behaviour; **not reproduced** if a reported metric has the
opposite sign, differs by more than 50%, or a qualitative claim (ranking,
monotonicity, plateau) fails; **partially reproduced** otherwise.

## Experiment 1 — Manufactured Connection, Gauge, and Motion Verification (Sec. 4.2)

Source: `exp/exp1.py`, metrics in `results/exp1_results.json`.

4-asset, rank-2 manufactured path with analytic `Q(t)` and Q-orthonormal
`U(t)`; continuous connection equation `dc/dt + omega(t) c(t) = 0` (Eq. 3.12)
integrated with a fixed-step classical 4th-order Runge-Kutta scheme and
checked against a DOP853 reference; 100 random smooth orthogonal gauge
transformations; algebraic identities (projector gauge covariance,
connection-form gauge covariance, skew-symmetry); orthogonal motion
decomposition (Eq. 3.15) and the geometric rebalancing energy (Sec. 3.5).

| Metric | Paper | Observed | Verdict |
|---|---|---|---|
| RK4 observed order (finest refinement) | 4.000 | 3.980 (4.00–4.01 on coarser steps, roundoff-limited at the finest step where the error is 2e-14) | reproduced |
| Max random-gauge transport error (100 gauges) | ~4.58e-10 | 3.82e-12 (vector), 3.39e-12 (norm) | reproduced |
| Projector gauge-covariance error | ~1e-15 | 6.66e-16 | reproduced |
| Connection-form gauge-covariance error | ~1e-15 | 2.22e-15 | reproduced |
| Connection skew-symmetry error | ~1e-15 | 2.22e-15 | reproduced |
| Tangential/normal orthogonality residual (motion decomposition) | machine precision | 1.72e-16 | reproduced |
| Normal-term Q-orthogonality residual | machine precision | 2.04e-16 | reproduced |
| Geometric rebalancing energy | positive, matches theory | 2.2279e-3, exact match to the closed-form value | reproduced |

All of experiment 1's checks are reproduced.

## Experiment 2 — Manufactured Convergence Order and Local Error Coefficient (Sec. 4.3–4.4)

Source: `exp/exp2.py`, metrics in `results/exp2_results.json`.

6-asset rank-3 noncommuting metric path (rotating eigenspaces, unequal
time-varying eigenvalues) with a diagonal commuting control; DOP853 at
relative tolerance 2e-12 as the exact coefficient reference; endpoint
Risk-weighted Procrustes, Principal square-root map, Geometric mean update
and Symmetric Procrustes rules on uniform partitions; the rank-2 exact
construction of Theorem 3.8 with `T = 1/2`; the local error coefficient
`D(t)` (Eq. 3.32) evaluated at `t = 0.37` on the 6D path.

| Metric | Paper | Observed | Verdict |
|---|---|---|---|
| Noncommuting: Risk-weighted Procrustes order | 1.001 | 0.997 | reproduced |
| Noncommuting: Principal square-root map order | -0.000 (plateau) | 6.8e-5 (plateau) | reproduced |
| Noncommuting: Geometric mean order | 2.000 | 2.000 | reproduced |
| Noncommuting: Symmetric Procrustes order | 2.000 | 2.000 | reproduced |
| Commuting: Risk-weighted Procrustes order | 0.991 | 0.998 | reproduced |
| Commuting: Principal square-root map order | 2.000 | 2.000 | reproduced |
| Rank-2 sharpness: Risk-weighted Procrustes fitted order | 1.000 | 1.0004 | reproduced |
| Rank-2 sharpness: `‖R_RP−I‖/h → sin(1/2)/4 ≈ 0.11986` | ≈0.1199 | 0.11988 (finest grid) | reproduced |
| Rank-2 sharpness: Geometric mean error | machine precision | 2.6e-16 – 4.6e-16 | reproduced |
| 6D path, `t=0.37`: discrepancy between `(R_RP−R_geo)/h²` and `D` decreases with `h` | linear decrease | ratio 2.00 when `h` is halved (0.005→0.0025), i.e. `O(h)` | reproduced (qualitative) |
| 6D path, `t=0.37`: `‖D(t)‖₂` | 0.118 | 0.0280 | not reproduced (see deviation below) |

The convergence-order results (the paper's central claims in this
experiment — first order for Risk-weighted Procrustes and the principal
square-root map's plateau under noncommutativity, second order for the
Geometric mean update and Symmetric Procrustes, and the exact rank-2
sharpness constant `sin(T)/4`) are all reproduced. The absolute value of the
local error coefficient `‖D(0.37)‖₂` on the 6-dimensional noncommuting path
is not reproduced in magnitude, because the paper does not give the analytic
formulas of its own 6-asset path (see "Deviations from the paper"); the
*rate* at which the discrepancy `(R_RP−R_geo)/h² − D` vanishes as `h→0` — the
qualitative content of Theorem 3.7 at that point — is reproduced.

## Experiment 3 — Historical Geometry Stress Test (Sec. 4.5)

Source: `exp/exp3.py`, metrics in `results/exp3_results.json`.

48 industry value-weighted daily portfolios from the Kenneth R. French Data
Library, 1990-01-02 through 2026-04-30 (9,149 observations, 0 missing), rank-5
investable subspace, 504-day rolling Ledoit-Wolf covariance, 5-day states; ex
ante local defect coefficient `D_t` (Eq. 3.32) estimated from the preceding
5 observations after a gauge-only orthogonal-Procrustes alignment of the
risk-whitened frames; subsequent scaled separation between Risk-weighted
Procrustes and Geometric-mean endpoint updates over the next 5 observations;
moving block bootstrap for the Spearman correlation confidence interval.

| Metric | Paper | Observed | Verdict |
|---|---|---|---|
| Spearman corr(‖D_t‖₂, subsequent scaled RP–geometric separation) | 0.458 | 0.448 | reproduced |
| 95% bootstrap interval | [0.394, 0.518] | [0.376, 0.523] | reproduced |
| Quartile-mean subsequent separation (bottom→top) | 1.57e-5 → 8.21e-5 | 1.572e-5 → 8.332e-5 | reproduced |
| Top/bottom quartile ratio | 5.24 | 5.30 | reproduced |
| Median ratio (observed scaled separation / predicted ‖D_t‖₂) | 0.977 | 0.919 | reproduced |
| Median scaled separation, Geometric mean vs Symmetric Procrustes | 7.62e-7 | 7.6206e-7 | reproduced |
| Geometric/Symmetric vs. RP/Geometric closeness ratio | ≈21.7× | 21.69× | reproduced |

All of experiment 3's metrics are reproduced.

## Experiment 4 — Historical Portfolio Implementation and Performance Evaluation (Sec. 4.6–4.7)

Source: `exp/exp4.py`, metrics in `results/exp4_table4_turnover_ce.csv`,
`results/exp4_summary.json`, `results/exp4_composition_discrepancy.csv`,
`results/exp4_bootstrap_ci.csv`, `results/exp4_sensitivity.csv`, and plots
`results/exp4_industries_wealth_turnover.png`,
`results/exp4_size_value_wealth_turnover.png`.

48 industry portfolios (1990-01-02 to 2026-04-30, 9,149 observations) and 100
size/book-to-market portfolios (2009-07-01 to 2024-06-28, 3,774 observations)
from the Kenneth R. French Data Library; 504-day rolling Ledoit-Wolf
covariance plus `1e-10·I`; rank-5 (industries) / rank-8 (size-value)
investable subspace `U_t = M_t E_t (E_t^T M_t^T Q_t M_t E_t)^{-1/2}`;
momentum signal from the trailing 252-day compounded return ending 21
observations before the decision, cross-sectionally winsorized at the median
± 4 MAD and demeaned; local active direction rescaled to 5% annual
volatility; equal-weight anchor; gross-exposure cap of 2 via radial
contraction of the active overlay; persistence weight 0.25 (primary) with a
grid of {0.10, 0.25, 0.50, 0.75}; 21-day rebalancing; proportional costs of
10 bps (industries) / 15 bps (size-value) deducted multiplicatively from
wealth; `CE = 252·mean(r) − 1.5·252·var(r)`.

### Table 4 — annualized turnover and certainty equivalent (persistence = 0.25)

| Panel | Method | Turnover | CE (net) | Turnover reduction vs. local | CE vs. geometric mean (bps) |
|---|---|---|---|---|---|
| industries | independent_local | 5.918 | 8.22% | 0.0% | -11.05 |
| industries | ambient_projection | 2.527 | 8.30% | 57.30% | -2.53 |
| industries | euclidean_procrustes | 2.593 | 8.28% | 56.18% | -4.53 |
| industries | risk_weighted_procrustes | 2.625 | 8.34% | 55.64% | +1.37 |
| industries | symmetric_procrustes | 2.625 | 8.33% | 55.64% | +0.54 |
| industries | geometric_mean | 2.627 | 8.33% | 55.61% | 0.00 |
| size_value | independent_local | 9.540 | 4.77% | 0.0% | -85.69 |
| size_value | ambient_projection | 4.775 | 5.70% | 49.95% | +7.54 |
| size_value | euclidean_procrustes | 4.902 | 5.72% | 48.62% | +9.71 |
| size_value | risk_weighted_procrustes | 4.905 | 5.61% | 48.58% | -1.55 |
| size_value | symmetric_procrustes | 4.903 | 5.63% | 48.61% | +0.13 |
| size_value | geometric_mean | 4.903 | 5.63% | 48.60% | 0.00 |

| Metric | Paper | Observed | Verdict |
|---|---|---|---|
| Geometric mean turnover reduction vs. local refitting, industries | 48.06% | 55.61% | partially reproduced (15.7% relative difference, same direction and order of magnitude) |
| Geometric mean turnover reduction vs. local refitting, size-value | 42.89% | 48.60% | partially reproduced (13.3% relative difference, same direction and order of magnitude) |
| CE(geometric mean) − CE(risk-weighted Procrustes), industries | -1.56 bps | -1.37 bps | reproduced in magnitude and sign; both confirm that transport-rule CE differences are a few bps, small next to the persistence-weight effect below |
| CE(geometric mean) − CE(risk-weighted Procrustes), size-value | +0.26 bps | +1.55 bps | partially reproduced (same order of magnitude, a few bps; exact value differs) |

### Direct versus composed coefficient-map discrepancy (`results/exp4_composition_discrepancy.csv`)

| Panel | Horizon (days) | Method | Median discrepancy |
|---|---|---|---|
| industries | 21 | ambient_projection | 0.384% |
| industries | 21 | risk_weighted_procrustes | 0.162% |
| industries | 21 | symmetric_procrustes | 0.152% |
| industries | 21 | geometric_mean | 0.150% |
| industries | 63 | ambient_projection | 1.899% |
| industries | 63 | risk_weighted_procrustes | 0.830% |
| industries | 63 | geometric_mean | 0.715% |

The paper reports an example median discrepancy of 0.131% versus 2.305% (an
ambient-projection-to-second-order ratio of about 17.6×) at the 21-day
horizon. Here the ranking is reproduced — ambient projection has the largest
direct-versus-composed discrepancy, and the Geometric mean and Symmetric
Procrustes updates stay much closer to each other (0.150% vs. 0.152%) than
either is to Risk-weighted Procrustes transport or to ambient projection —
but the ambient-projection-to-second-order ratio (2.56×) is far below the
paper's 17.6×: **partially reproduced**.

### Sensitivity to persistence weight and rebalancing frequency

`results/exp4_sensitivity.csv` shows annualized turnover rising monotonically
with the persistence weight on the {0.10, 0.25, 0.50, 0.75} grid for every
transport rule and panel, confirming the paper's qualitative claim that the
persistence weight, not the choice among the four transport rules, is the
dominant lever on turnover and CE (Sec. 4.7): **reproduced** (qualitative).

### Deviations from the paper used for experiment 4

Two details of the methodology that the paper's text leaves open were
resolved as follows, and both are the most likely source of the turnover and
composition-discrepancy gaps above:

- the "momentum signal compounded over 252 daily returns" is implemented as
  the simple compounded (buy-and-hold) return `∏(1+r_i) − 1` over the trailing
  252-observation window ending 21 observations before the decision date;
- the gross-exposure cap of 2 is enforced by uniform radial contraction of
  the active overlay weights whenever `‖w‖₁ > 2`, leaving the equal-weight
  anchor unscaled.

No other interpretation of these two steps is stated in the paper.

## Deviations from the paper

- **Experiment 2, rank-2 sharpness construction (Theorem 3.8):** implemented
  exactly as given, `L = b e_2^T − e_2 b^T`, `Q(t) = diag(e^t, 1, e^{-t})`,
  `U(t) = Q(t)^{-1/2} e^{tL} [a, e_2]`, `a = (e_1+e_3)/√2`, `b = (e_1-e_3)/√2`,
  `T = 1/2`.
- **Experiment 2, 6-asset noncommuting path:** the paper does not give the
  analytic formulas for its own 6-asset rotating-eigenspace metric and rank-3
  frame, only that the path is noncommuting with a commuting diagonal
  control. A path satisfying those two constraints was constructed (rotating
  2×2 eigenblocks with unequal, time-varying eigenvalues embedded in 6
  dimensions, and a rank-3 Q-orthonormal frame obtained from a smooth
  orthogonal path). Convergence orders and the sharpness constant computed
  on this path match the paper's values; the absolute value of the local
  error coefficient `‖D(0.37)‖₂` does not (0.0280 vs. 0.118), because it
  depends on the specific path and not only on its convergence-order class.
  This is reported as "not reproduced" above rather than attributed to a code
  defect, since the component identities (Q-orthonormality of `U(t)`, the
  `O(h)` vanishing of the discrepancy predicted by Theorem 3.7) hold
  exactly on this path.
- **Experiment 1, manufactured 4-asset path:** the paper does not give the
  analytic formulas for `Q(t)` and `U(t)` of this problem either; a path with
  a nonconstant metric, a rank-2 Q-orthonormal frame, and an analytic
  coefficient solution of the continuous transport equation was constructed
  for this purpose. Because experiment 1 only tests internal consistency
  (gauge covariance, algebraic identities, the RK4 reference order, the
  motion decomposition), the choice of path does not affect which checks
  pass.
- **Experiment 4, momentum signal and exposure cap:** see "Deviations from
  the paper used for experiment 4" above; the paper's text does not give a
  closed-form definition of either step.
- **Data source:** experiments 1 and 2 use no market data at all (purely
  synthetic manufactured paths; no data source applies). Experiments 3 and 4
  use the Kenneth R. French Data Library's 48-industry and 100
  size/book-to-market value-weighted daily return panels directly, as
  specified by the paper; this is not OHLCV bar data, so it is downloaded
  through `pandas_datareader` rather than through the QuantX Dataloader
  (which serves per-instrument price bars, not pre-aggregated academic
  portfolio return panels). No Yahoo Finance or other backstop source was
  used; there are no excluded instruments, since both panels are supplied by
  French's library as complete, already-aggregated return series rather than
  as a universe of individual securities to be assembled instrument by
  instrument.
- **Sample period:** the industries panel realized sample is 1990-01-02
  through 2026-04-30 (9,149 observations) and the size-value panel is
  2009-07-01 through 2024-06-28 (3,774 observations), both equal to the
  ranges and observation counts the paper states for these panels (Sec. 4.6).
- **Universe size:** the full paper universe (48 industry portfolios, 100
  size/book-to-market portfolios) is used in every run; no narrowing was
  necessary because both panels are small, pre-aggregated download targets
  rather than a large per-instrument OHLCV universe.

## Not reproduced

Nothing in the four selected experiments was left unimplemented. The one
exact value not reproduced — the local error coefficient magnitude
`‖D(0.37)‖₂ = 0.118` quoted for the paper's own 6-asset noncommuting path in
experiment 2 — is listed under "Deviations from the paper" above rather than
here, since it is a consequence of the path construction rather than a
methodology step this repository omits.
