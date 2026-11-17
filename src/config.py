"""Experiment parameters for the replication of
'A Second Order Transport Scheme for Dynamic Portfolio Subspaces under a
Varying Risk Metric' (A. Rodriguez Dominguez).

Every module-level constant names its meaning, unit, and the paper section
or equation it is taken from. Constants with no stated paper value are
marked 'not specified in the paper' and a reasonable choice is recorded.
"""

# ---------------------------------------------------------------------------
# Experiment 1: manufactured connection, gauge and motion verification
# (paper Sec. 4.2, Appendix C "first manufactured problem")
# ---------------------------------------------------------------------------
EXP1_N_ASSETS = 4          # ambient asset-coordinate dimension n (Sec. 4.2)
EXP1_RANK = 2              # investable subspace rank p (Sec. 4.2)
EXP1_T0 = 0.0              # path start time, not specified in the paper
EXP1_T1 = 1.0              # path end time, not specified in the paper
EXP1_RK4_REFINEMENTS = [10, 20, 40, 80, 160, 320]  # RK4 step counts used to
# measure the observed order of the reference integrator against the exact
# closed-form rotation solution; grid not specified in the paper, chosen fine
# enough for a clean order-4 fit while staying above the float64 roundoff
# floor (640 steps drives the error below ~1e-15, where the fitted order
# degrades to noise rather than reflecting truncation error)
EXP1_N_GAUGES = 100        # number of random smooth gauge transformations
# tested for transport invariance (Sec. 4.2, "100 random smooth orthogonal
# gauge transformations")
EXP1_GAUGE_STEPS = 400     # RK4 steps used for each gauge-transport check,
# not specified in the paper; chosen fine enough for near machine-precision
# agreement
EXP1_SEED = 20260001       # random seed for the 100 gauge draws, not
# specified in the paper

# ---------------------------------------------------------------------------
# Experiment 2: manufactured convergence order and local error coefficient
# (paper Sec. 4.3-4.4, Theorem 3.7, Theorem/Proposition 3.8)
# ---------------------------------------------------------------------------
EXP2_N_ASSETS = 6          # ambient dimension of the noncommuting path (Sec. 4.3)
EXP2_RANK = 3              # subspace rank of the noncommuting path (Sec. 4.3)
EXP2_T0 = 0.0              # path start time, not specified in the paper
EXP2_T1 = 1.0              # path end time, not specified in the paper
EXP2_DOP853_RTOL = 2e-12   # DOP853 relative tolerance for the continuous
# coefficient reference (Sec. 4.3: "DOP853 ... at relative tolerance 2e-12")
EXP2_DOP853_ATOL = 1e-14   # absolute tolerance, not specified in the paper
EXP2_STEP_COUNTS = [20, 40, 80, 160, 320, 640]  # uniform partitions used to
# fit the global convergence order of each endpoint rule; refinement values
# not exhaustively listed in the paper beyond those used in Sec. 4.4

EXP2_RANK2_T = 0.5         # T = 1/2 in Proposition 3.8's exact rank-2
# construction (Sec. 4.4: "We set T = 1/2")
EXP2_RANK2_STEP_COUNTS = [32, 64, 128, 256]  # step counts of Table 3 (Sec. 4.4)

EXP2_LOCAL_TIME = 0.37     # evaluation time of D(t) on the 6D path (Sec. 4.4:
# "At time 0.37 on the 6D path")
EXP2_LOCAL_STEPS = [0.005, 0.0025]  # step sizes used for the local
# discrepancy check (Sec. 4.4: "from ... step 0.005 to ... step 0.0025")

# ---------------------------------------------------------------------------
# Experiment 3: historical geometry stress test
# (paper Sec. 4.5, Eq. 3.32, 48-industry panel)
# ---------------------------------------------------------------------------
EXP3_RANK = 5                 # representation rank, same as the industry
# portfolio experiment (Sec. 4.5: "same ... rank five representation")
EXP3_COV_WINDOW = 504         # Ledoit-Wolf covariance window length, trading
# days (Sec. 4.6/Appendix C.1, reused for the Sec. 4.5 diagnostic)
EXP3_RIDGE = 1e-10            # diagonal ridge added to the covariance matrix
# (Appendix C.1: "Q_t is the Ledoit-Wolf estimate ... plus 1e-10 I_n")
EXP3_STATE_LEN = 5            # length in trading days of each rolling state
# used to estimate Qdot, Udot and to evaluate the subsequent separation
# (Sec. 4.5: "each five day state", "preceding five observations")
EXP3_BOOTSTRAP_REPS = 2000    # moving block bootstrap replications for the
# Spearman correlation interval; count not specified in the paper, chosen
# for a stable 95% interval
EXP3_BOOTSTRAP_BLOCK = 20     # block length (states) for the moving block
# bootstrap; not specified in the paper
EXP3_N_COMPARISONS_PAPER = 1727  # number of historical comparisons reported
# in the paper (Sec. 4.5: "Across 1,727 historical comparisons")

# ---------------------------------------------------------------------------
# Experiment 4: historical portfolio implementation
# (paper Sec. 4.1, 4.6-4.7, Appendix C)
# ---------------------------------------------------------------------------
# Data source and sample (Sec. 4.6, data_requirements)
FF_INDUSTRY_DATASET = "48_Industry_Portfolios_daily"    # Kenneth French
# Data Library dataset name for pandas_datareader (Sec. 4.6)
FF_SIZEVALUE_DATASET = "100_Portfolios_10x10_Daily"     # size/book-to-market
# intersections dataset name (Sec. 4.6)
EXP4_INDUSTRIES_START = "1990-01-02"   # sample start (Sec. 4.6: "2 Jan 1990")
EXP4_INDUSTRIES_END = "2026-04-30"     # sample end (Sec. 4.6: "30 Apr 2026")
EXP4_INDUSTRIES_N_OBS_PAPER = 9149     # reported observation count (Sec. 4.6)
EXP4_SIZEVALUE_START = "2009-07-01"    # sample start (Sec. 4.6: "1 Jul 2009")
EXP4_SIZEVALUE_END = "2024-06-28"      # sample end (Sec. 4.6: "28 Jun 2024")
EXP4_SIZEVALUE_N_OBS_PAPER = 3774      # reported observation count (Sec. 4.6)

# Universe size actually processed in the executed run (paper uses the full
# panels; both panels are small index portfolios, not individual equities,
# so no universe reduction for download time is required here).
EXP4_UNIVERSE_SIZE_PAPER_INDUSTRIES = 48
EXP4_UNIVERSE_SIZE_RUN_INDUSTRIES = 48
EXP4_UNIVERSE_SIZE_PAPER_SIZEVALUE = 100
EXP4_UNIVERSE_SIZE_RUN_SIZEVALUE = 100

# Covariance and subspace construction (Appendix C.1)
EXP4_COV_WINDOW = 504          # Ledoit-Wolf covariance window, trading days
EXP4_RIDGE = 1e-10             # diagonal ridge 1e-10 I_n (Appendix C.1)
EXP4_RANK_INDUSTRIES = 5       # leading covariance eigenvectors, industries
# (Appendix C.1: "leading five covariance eigenvectors for industries")
EXP4_RANK_SIZEVALUE = 8        # leading covariance eigenvectors, size-value
# (Appendix C.1: "eight for size and value")

# Momentum signal (Appendix C.1)
EXP4_MOMENTUM_LOOKBACK = 252   # compounding window, trading days
EXP4_MOMENTUM_GAP = 21         # observations between signal end and decision
EXP4_MOMENTUM_MAD_K = 4.0      # clip at cross-sectional median +/- 4 MAD
EXP4_ACTIVE_VOL_TARGET = 0.05  # annualized active volatility target, 5%/yr
EXP4_ANNUALIZATION_DAYS = 252  # trading days per year used for annualization
EXP4_GROSS_CAP = 2.0           # gross exposure cap on the active component

# Rebalancing and persistence (Sec. 4.1, 4.6)
EXP4_REBALANCE_FREQ_PRIMARY = 21     # primary rebalancing frequency, days
EXP4_REBALANCE_FREQ_SENSITIVITY = [5, 21, 63]  # weekly/monthly/quarterly
EXP4_PERSISTENCE_PRIMARY = 0.25      # primary weight on the new local target
EXP4_PERSISTENCE_GRID = [0.10, 0.25, 0.50, 0.75]  # sensitivity grid

# Transaction costs (Sec. 4.1, 4.6)
EXP4_COST_INDUSTRIES_BPS = 10.0      # proportional cost, basis points
EXP4_COST_SIZEVALUE_BPS = 15.0       # proportional cost, basis points

# Certainty equivalent (Table 4 caption, Appendix C.3)
EXP4_RISK_AVERSION = 3.0             # risk-aversion coefficient in CE

# Endpoint overlap rejection rule (Appendix C.1)
EXP4_ENDPOINT_SV_TOL = 1e-10         # reject if sigma_min < tol * sigma_max

# Statistical inference (Sec. 4.6, Appendix C.3)
EXP4_BOOTSTRAP_REPS = 500            # paired moving block bootstrap draws
EXP4_BOOTSTRAP_BLOCK_RETURNS = 3     # block length, rebalancing intervals
EXP4_BOOTSTRAP_BLOCK_DECISIONS = 6   # block length, decisions

# Composition diagnostic (Sec. 4.7, Appendix C.3)
EXP4_COMPOSITION_REPS_PAPER = 1000   # replications used in the paper
EXP4_COMPOSITION_REPS_RUN = 1000     # replications executed in this run
EXP4_COMPOSITION_START_SEP = 65      # separation between starting dates
EXP4_COMPOSITION_SUBINTERVAL = 5     # composition sub-interval, observations
EXP4_COMPOSITION_HORIZONS = [21, 63]  # comparison horizons, trading days

# Numerical tolerances shared across experiments
GEOMETRY_EPS = 1e-12           # small ridge used only inside matrix power
# computations to guard against roundoff-induced negative eigenvalues
