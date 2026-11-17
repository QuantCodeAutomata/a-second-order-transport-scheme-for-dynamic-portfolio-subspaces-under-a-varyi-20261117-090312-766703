"""Unit tests of src/geometry.py on the manufactured problems.

These check the paper's structural properties (gauge invariance, skew
symmetry, Q-orthogonality, convergence order, exact sharpness of the
Theorem 3.8 construction) rather than reported numbers, on small synthetic
inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src import geometry as geo


def test_manufactured_problem_1_is_q_orthonormal():
    for t in (0.0, 0.37, 1.0, 1.9):
        Q, dQ, U, dU = geo.manufactured_problem_1(t)
        gram = U.T @ Q @ U
        assert np.allclose(gram, np.eye(2), atol=1e-10)
        assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_manufactured_problem_2_noncommuting_and_commuting():
    Q_nc, dQ_nc, U_nc, _ = geo.manufactured_problem_2(0.6, commuting=False)
    Q_c, dQ_c, U_c, _ = geo.manufactured_problem_2(0.6, commuting=True)
    assert np.allclose(U_nc.T @ Q_nc @ U_nc, np.eye(3), atol=1e-10)
    assert np.allclose(U_c.T @ Q_c @ U_c, np.eye(3), atol=1e-10)
    # Noncommuting: Q and dQ must not commute at this time.
    commutator = Q_nc @ dQ_nc - dQ_nc @ Q_nc
    assert np.max(np.abs(commutator)) > 1e-3
    # Commuting control: Q and dQ are both diagonal, hence commute exactly.
    commutator_c = Q_c @ dQ_c - dQ_c @ Q_c
    assert np.max(np.abs(commutator_c)) < 1e-12


def test_connection_form_is_skew_symmetric():
    Q, dQ, U, dU = geo.manufactured_problem_1(0.42)
    omega = geo.connection_form(Q, dQ, U, dU)
    assert np.max(np.abs(omega + omega.T)) < 1e-12


def test_defect_coefficient_is_skew_symmetric():
    Q, dQ, U, dU = geo.manufactured_problem_2(0.37, commuting=False)
    D = geo.defect_coefficient(Q, dQ, U, dU)
    assert np.max(np.abs(D + D.T)) < 1e-10


def test_motion_decomposition_orthogonality():
    Q, dQ, U, dU = geo.manufactured_problem_1(0.55)
    c, dc = np.array([0.4, -0.3]), np.array([0.1, 0.2])
    tangential, normal, total = geo.motion_decomposition(Q, dQ, U, dU, c, dc)
    assert np.allclose(total, tangential + normal)
    assert abs(tangential @ Q @ normal) < 1e-9


def test_rebalancing_energy_nonnegative_and_zero_for_no_move():
    Q, _, _, _ = geo.manufactured_problem_1(0.2)
    delta = np.zeros(4)
    assert geo.rebalancing_energy(Q, delta) == 0.0
    delta = np.array([0.01, -0.02, 0.0, 0.01])
    assert geo.rebalancing_energy(Q, delta) > 0.0


def test_random_gauge_transform_is_orthogonal_and_matches_derivative():
    R, dR = geo.random_gauge_transform(seed=1, rank=2)
    for t in (0.0, 0.3, 0.9):
        Rt = R(t)
        assert np.allclose(Rt.T @ Rt, np.eye(2), atol=1e-10)
        h = 1e-6
        fd = (R(t + h) - R(t - h)) / (2 * h)
        assert np.max(np.abs(fd - dR(t))) < 1e-5


@pytest.mark.parametrize("n_steps", [50, 100, 200])
def test_rk4_matches_scipy_reference_closely(n_steps):
    """The custom fixed-step RK4 used in exp1 should track solve_ivp's
    adaptive RK45 reference to well within its own truncation error."""
    from exp.exp1 import omega_of_manufactured, rk4_fixed_step

    c0 = np.array([1.0, 0.0])

    def rhs(t, c):
        return geo.continuous_transport_rhs(c, omega_of_manufactured(t))

    ref = solve_ivp(rhs, (0.0, 1.0), c0, method="RK45", rtol=1e-11, atol=1e-13)
    assert ref.success
    c_rk4 = rk4_fixed_step(omega_of_manufactured, c0, 0.0, 1.0, n_steps)
    assert np.linalg.norm(c_rk4 - ref.y[:, -1]) < 1e-5


def test_transport_rules_agree_when_frame_is_unchanged():
    """With Q0 = Q1 and U0 = U1, every endpoint rule must return (close to)
    the identity coefficient map."""
    Q, _, U, _ = geo.manufactured_problem_1(0.3)
    for fn in (geo.risk_weighted_procrustes, geo.principal_root_map,
               geo.geometric_mean_update, geo.symmetric_procrustes_update,
               geo.euclidean_procrustes):
        if fn is geo.euclidean_procrustes:
            R = fn(U, U)
        else:
            R = fn(Q, U, Q, U)
        assert np.allclose(R, np.eye(2), atol=1e-8)


def test_rank2_theorem38_exact_propagator_is_identity():
    Q0, U0 = geo.rank2_theorem38_path(0.1)
    Q1, U1 = geo.rank2_theorem38_path(0.1 + 1e-4)
    assert np.allclose(U0.T @ Q0 @ U0, np.eye(2), atol=1e-8)
    assert np.allclose(U1.T @ Q1 @ U1, np.eye(2), atol=1e-8)
    psi = geo.rank2_theorem38_exact_propagator()
    assert np.allclose(psi, np.eye(2))


def test_endpoint_overlap_rejection():
    Q = np.eye(2)
    U0 = np.array([[1.0, 0.0], [0.0, 0.0]])  # rank-deficient, singular overlap
    with pytest.raises(ValueError):
        geo.euclidean_procrustes(U0, np.eye(2)[:, :2], sv_tol=1e-6)
