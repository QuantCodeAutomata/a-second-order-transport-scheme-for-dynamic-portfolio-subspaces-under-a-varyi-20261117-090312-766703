"""Unit tests of exp/exp3.py (paper Sec. 4.5, Historical Geometry Stress
Test) on a small slice of the 48-industry daily panel.

These tests need a live download of the Kenneth R. French Data Library
(no API key required) and are marked `market_data` so they can be skipped
when that download is unavailable, following the pattern of
`tests/conftest.py`'s `french_industries_slice` fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import geometry as geo
from exp.exp3 import (
    build_states,
    compute_daily_frames,
    state_defect_and_separation,
)

COV_WINDOW = 60
RANK = 3
RIDGE = 1e-10
STATE_LEN = 5


@pytest.fixture(scope="module")
def small_frames(french_industries_slice):
    returns = french_industries_slice
    Qs, Us, t_start = compute_daily_frames(returns, COV_WINDOW, RIDGE, RANK)
    return returns, Qs, Us, t_start


@pytest.mark.market_data
def test_compute_daily_frames_shapes_and_q_orthonormality(small_frames):
    returns, Qs, Us, t_start = small_frames
    n_assets = returns.shape[1]
    assert Qs.shape[1:] == (n_assets, n_assets)
    assert Us.shape[1:] == (n_assets, RANK)
    assert len(Qs) == len(returns) - COV_WINDOW
    for i in (0, len(Qs) // 2, len(Qs) - 1):
        gram = Us[i].T @ Qs[i] @ Us[i]
        assert np.allclose(gram, np.eye(RANK), atol=1e-6)


@pytest.mark.market_data
def test_build_states_boundaries_are_valid(small_frames):
    returns, Qs, Us, t_start = small_frames
    boundaries = build_states(len(returns), COV_WINDOW, STATE_LEN)
    assert len(boundaries) > 10
    for t in boundaries:
        assert t - STATE_LEN >= t_start
        assert t + STATE_LEN <= len(returns) - 1


@pytest.mark.market_data
def test_euclidean_procrustes_output_is_orthogonal(small_frames):
    returns, Qs, Us, t_start = small_frames
    U0, U1 = Us[0], Us[10]
    R = geo.euclidean_procrustes(U0, U1)
    assert np.allclose(R.T @ R, np.eye(RANK), atol=1e-8)


@pytest.mark.market_data
def test_gauge_alignment_preserves_subspace(small_frames):
    returns, Qs, Us, t_start = small_frames
    for i, j in ((0, 10), (5, 20)):
        U_i, Q_i = Us[i], Qs[i]
        U_ref = Us[j]
        R = geo.euclidean_procrustes(U_i, U_ref)
        U_i_aligned = U_i @ R
        projected_before = U_i @ U_i.T @ Q_i
        projected_after = U_i_aligned @ U_i_aligned.T @ Q_i
        assert np.allclose(projected_before, projected_after, atol=1e-8)


@pytest.mark.market_data
def test_state_defect_coefficient_is_skew_symmetric(small_frames):
    returns, Qs, Us, t_start = small_frames
    boundaries = build_states(len(returns), COV_WINDOW, STATE_LEN)
    n_checked = 0
    for t in boundaries[:15]:
        res = state_defect_and_separation(t, t_start, Qs, Us, STATE_LEN)
        if res is None:
            continue
        # Recompute D directly to check skew-symmetry explicitly.
        Q_t, U_t = Qs[t - t_start], Us[t - t_start]
        rel_times = np.arange(-STATE_LEN, 1, dtype=float)
        Q_window = np.empty((STATE_LEN + 1,) + Q_t.shape)
        U_window = np.empty((STATE_LEN + 1,) + U_t.shape)
        for k, i in enumerate(range(t - STATE_LEN, t)):
            Q_i, U_i = Qs[i - t_start], Us[i - t_start]
            R_i = geo.euclidean_procrustes(U_i, U_t)
            Q_window[k] = Q_i
            U_window[k] = U_i @ R_i
        Q_window[-1] = Q_t
        U_window[-1] = U_t
        from exp.exp3 import _ols_slope
        Qdot = _ols_slope(rel_times, Q_window)
        Udot = _ols_slope(rel_times, U_window)
        D = geo.defect_coefficient(Q_t, Qdot, U_t, Udot)
        assert np.max(np.abs(D + D.T)) < 1e-6
        n_checked += 1
    assert n_checked >= 5


@pytest.mark.market_data
def test_separations_nonnegative_and_full_state_pipeline_finite(small_frames):
    returns, Qs, Us, t_start = small_frames
    boundaries = build_states(len(returns), COV_WINDOW, STATE_LEN)
    assert len(boundaries) >= 10

    n_valid = 0
    for t in boundaries:
        res = state_defect_and_separation(t, t_start, Qs, Us, STATE_LEN)
        if res is None:
            continue
        assert np.isfinite(res["norm_D_t"])
        assert np.isfinite(res["sep_rp_geo"])
        assert np.isfinite(res["sep_rp_sym"])
        assert np.isfinite(res["sep_geo_sym"])
        assert res["sep_rp_geo"] >= 0.0
        assert res["sep_rp_sym"] >= 0.0
        assert res["sep_geo_sym"] >= 0.0
        assert res["norm_D_t"] >= 0.0
        n_valid += 1
        if n_valid >= 10:
            break
    assert n_valid >= 10
