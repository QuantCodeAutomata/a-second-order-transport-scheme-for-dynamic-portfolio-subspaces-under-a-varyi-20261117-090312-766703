"""Unit tests for exp/exp2.py and the geometry primitives it exercises
(Sec. 4.3-4.4). Tests check structural properties on small, fast ad hoc
grids rather than reproducing the full convergence study.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src import config as cfg
from src import geometry as geo
from exp import exp2


def test_connection_form_skew_symmetric_noncommuting_and_commuting():
    for t in (0.0, 0.37, 0.9):
        for commuting in (False, True):
            Q, dQ, U, dU = geo.manufactured_problem_2(t, commuting=commuting)
            omega = geo.connection_form(Q, dQ, U, dU)
            assert np.allclose(omega, -omega.T, atol=1e-9)


def test_dop853_propagator_is_orthogonal():
    t0, t1 = 0.0, 0.2
    Psi = exp2.exact_propagator(t0, t1, commuting=False, rank=cfg.EXP2_RANK)
    assert np.allclose(Psi.T @ Psi, np.eye(cfg.EXP2_RANK), atol=1e-8)


def test_rank2_theorem38_path_is_q_orthonormal():
    for t in (0.1, 0.5, 1.0, 1.4):
        Q, U = geo.rank2_theorem38_path(t)
        gram = U.T @ Q @ U
        assert np.allclose(gram, np.eye(2), atol=1e-8)


def test_geometric_mean_update_sharp_on_rank2_construction():
    T = 0.5
    t_nodes = np.linspace(0.0, T, 4)
    qu_nodes = [geo.rank2_theorem38_path(t) for t in t_nodes]
    R_total = exp2.compose_path_rule(geo.geometric_mean_update, qu_nodes)
    exact = geo.rank2_theorem38_exact_propagator()
    assert np.linalg.norm(exact - R_total, 2) < 1e-6


def test_risk_weighted_procrustes_error_over_h_is_small_positive():
    T = 0.5
    t_nodes = np.array([0.0, T])
    qu_nodes = [geo.rank2_theorem38_path(t) for t in t_nodes]
    R_total = exp2.compose_path_rule(geo.risk_weighted_procrustes, qu_nodes)
    exact = geo.rank2_theorem38_exact_propagator()
    err_over_h = np.linalg.norm(exact - R_total, 2) / T
    assert 0.05 < err_over_h < 0.3
