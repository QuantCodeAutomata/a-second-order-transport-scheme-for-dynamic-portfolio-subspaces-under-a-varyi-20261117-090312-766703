"""Experiment 1: Manufactured Connection, Gauge, and Motion Verification
(paper Sec. 4.2).

Runs on the 4-asset, rank-2 manufactured path `src.geometry.manufactured_problem_1`:
  1. Convergence order of a classical fixed-step 4th-order Runge-Kutta
     integrator of the continuous connection equation dc/dt + omega(t)c(t)=0
     (Eq. 3.12), checked against a tight-tolerance DOP853 reference.
  2. Gauge covariance: 100 random smooth orthogonal ambient transformations
     R(t) applied to (Q, U); the terminal transported asset-space vector and
     its risk norm must agree with the untransformed computation.
  3. Algebraic identities checked in closed form (projector gauge covariance,
     connection-form gauge transformation law, skew-symmetry of omega).
  4. Orthogonal motion decomposition (Eq. 3.15) and Q-orthogonality of its
     tangential/normal parts, and the geometric rebalancing energy (Sec. 3.5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp

from src import config as cfg
from src import geometry as geo

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

J2 = np.array([[0.0, 1.0], [-1.0, 0.0]])  # the only (up to scale) 2x2 skew-symmetric generator


def rk4_fixed_step(omega_fn, c0: np.ndarray, t0: float, t1: float, n_steps: int) -> np.ndarray:
    """Classical fixed-step 4th-order Runge-Kutta integration of
    dc/dt = -omega(t) c(t), used as the scheme under test for the
    convergence-order check (object (1) of Sec. 4.2)."""
    h = (t1 - t0) / n_steps
    c = c0.copy()
    t = t0
    for _ in range(n_steps):
        k1 = geo.continuous_transport_rhs(c, omega_fn(t))
        k2 = geo.continuous_transport_rhs(c + 0.5 * h * k1, omega_fn(t + 0.5 * h))
        k3 = geo.continuous_transport_rhs(c + 0.5 * h * k2, omega_fn(t + 0.5 * h))
        k4 = geo.continuous_transport_rhs(c + h * k3, omega_fn(t + h))
        c = c + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return c


def omega_of_manufactured(t: float) -> np.ndarray:
    Q, dQ, U, dU = geo.manufactured_problem_1(t)
    return geo.connection_form(Q, dQ, U, dU)


def exact_coefficient_solution(t0: float, t1: float, c0: np.ndarray) -> np.ndarray:
    """Exact analytic solution of dc/dt + omega(t) c(t) = 0 for the rank-2
    manufactured problem.

    Any real 2x2 skew-symmetric matrix is a scalar multiple of the fixed
    generator J = [[0, 1], [-1, 0]], so omega(t) = omega_12(t) * J for a
    scalar function omega_12. Since J commutes with itself at every time,
    the linear ODE has the exact closed-form rotation solution
    c(t1) = expm(-F(t1-t0) J) c0 = cos(F) c0 - sin(F) (J c0), where
    F = integral of omega_12 over [t0, t1]. F is evaluated with adaptive
    quadrature (`scipy.integrate.quad`), giving a reference accurate to
    quadrature tolerance rather than to another fixed-step integrator's
    truncation error, matching the "analytic coefficient solution" of the
    manufactured problem (Sec. 4.2).
    """
    def omega_12(t: float) -> float:
        return float(omega_of_manufactured(t)[0, 1])

    F, quad_err = quad(omega_12, t0, t1, epsabs=1e-14, epsrel=1e-13, limit=200)
    assert quad_err < 1e-10, "quadrature reference not accurate enough"
    return np.cos(F) * c0 - np.sin(F) * (J2 @ c0)


def run_rk4_order_check() -> dict:
    """Fits the empirical convergence order of `rk4_fixed_step` against the
    exact closed-form rotation solution available for the rank-2 problem
    (see `exact_coefficient_solution`), independent of the fixed-step
    scheme under test."""
    t0, t1 = cfg.EXP1_T0, cfg.EXP1_T1
    c0 = np.array([1.0, 0.0])

    c_ref = exact_coefficient_solution(t0, t1, c0)

    errors = []
    for n in cfg.EXP1_RK4_REFINEMENTS:
        c_n = rk4_fixed_step(omega_of_manufactured, c0, t0, t1, n)
        errors.append(np.linalg.norm(c_n - c_ref))

    orders = [
        np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)
    ]
    return {
        "step_counts": cfg.EXP1_RK4_REFINEMENTS,
        "errors": errors,
        "observed_orders": orders,
        "finest_order": orders[-1],
    }


def run_gauge_covariance_check() -> dict:
    """100 random smooth orthogonal gauge transforms of the Q-orthonormal
    frame bundle, U(t) -> U(t) R(t) with Q left untouched, verifying that
    the terminal transported asset-space vector and its risk norm are
    gauge invariant (Sec. 3.3, Sec. 4.2).

    Under this gauge, the connection form transforms as
    omega -> R^T omega R + R^T dR/dt (since U^T Q U = I_rank), and the
    coefficient equation dc/dt + omega c = 0 has the exact solution
    c_gauge(t) = R(t)^T c(t) for every t, so the physical asset-space
    vector U(t) R(t) c_gauge(t) = U(t) c(t) is identically gauge invariant.
    """
    t0, t1 = cfg.EXP1_T0, cfg.EXP1_T1
    rank = cfg.EXP1_RANK
    c0 = np.array([0.6, -0.8])

    def base_rhs(t, c):
        return geo.continuous_transport_rhs(c, omega_of_manufactured(t))

    base = solve_ivp(base_rhs, (t0, t1), c0, method="RK45", rtol=1e-11,
                      atol=1e-13, dense_output=True)
    assert base.success
    Q1, _, U1, _ = geo.manufactured_problem_1(t1)
    c1 = base.y[:, -1]
    v_base = U1 @ c1
    norm_base = np.sqrt(max(v_base @ Q1 @ v_base, 0.0))

    vector_errors = []
    norm_errors = []
    projector_gauge_errors = []
    connection_covariance_errors = []
    connection_skew_errors = []
    coefficient_identity_errors = []

    for i in range(cfg.EXP1_N_GAUGES):
        R, dR = geo.random_gauge_transform(seed=cfg.EXP1_SEED + i, rank=rank)

        def gauge_rhs(t, c, R=R, dR=dR):
            Q, dQ, U, dU = geo.manufactured_problem_1(t)
            Rt, dRt = R(t), dR(t)
            Ug = U @ Rt
            dUg = dU @ Rt + U @ dRt
            omega_g = geo.connection_form(Q, dQ, Ug, dUg)
            return geo.continuous_transport_rhs(c, omega_g)

        cg0 = R(t0).T @ c0
        sol = solve_ivp(gauge_rhs, (t0, t1), cg0, method="RK45", rtol=1e-11,
                         atol=1e-13)
        assert sol.success
        Rt1 = R(t1)
        Ug1 = U1 @ Rt1
        cg1 = sol.y[:, -1]
        v_from_gauge = Ug1 @ cg1  # Q is untouched: v_gauge should equal v_base

        vector_errors.append(np.linalg.norm(v_from_gauge - v_base))
        norm_g = np.sqrt(max(v_from_gauge @ Q1 @ v_from_gauge, 0.0))
        norm_errors.append(abs(norm_g - norm_base))
        coefficient_identity_errors.append(np.linalg.norm(cg1 - Rt1.T @ c1))

        # Algebraic identities, checked at a sample of interior times.
        for t_s in (0.0, 0.3, 0.71, 1.0):
            Q, dQ, U, dU = geo.manufactured_problem_1(t_s)
            Rt, dRt = R(t_s), dR(t_s)
            Ug = U @ Rt
            dUg = dU @ Rt + U @ dRt

            P = geo.projector(Q, U)
            Pg = geo.projector(Q, Ug)
            projector_gauge_errors.append(np.max(np.abs(Pg - P)))

            omega = geo.connection_form(Q, dQ, U, dU)
            omega_g = geo.connection_form(Q, dQ, Ug, dUg)
            # Gauge transformation law of the connection form, Sec. 3.3:
            # omega_g = R^T omega R + R^T dR/dt (U^T Q U = I_rank).
            predicted = Rt.T @ omega @ Rt + Rt.T @ dRt
            connection_covariance_errors.append(np.max(np.abs(omega_g - predicted)))
            connection_skew_errors.append(np.max(np.abs(omega_g + omega_g.T)))

    return {
        "n_gauges": cfg.EXP1_N_GAUGES,
        "max_vector_error": max(vector_errors),
        "max_norm_error": max(norm_errors),
        "max_coefficient_identity_error": max(coefficient_identity_errors),
        "max_projector_gauge_error": max(projector_gauge_errors),
        "max_connection_covariance_error": max(connection_covariance_errors),
        "max_connection_skew_error": max(connection_skew_errors),
    }


def run_motion_decomposition_check() -> dict:
    """Verifies the orthogonal motion decomposition (Eq. 3.15) and the
    Q-orthogonality of its tangential and normal parts, plus the geometric
    rebalancing energy (Sec. 3.5)."""
    t = 0.42
    h = 1e-6
    Q, dQ, U, dU = geo.manufactured_problem_1(t)
    c = np.array([0.3, 0.9])
    dc = np.array([-0.2, 0.15])

    tangential, normal, total = geo.motion_decomposition(Q, dQ, U, dU, c, dc)
    assert np.allclose(total, tangential + normal)

    orthogonality_residual = abs(tangential @ Q @ normal)
    normal_q_orthogonal_residual = np.max(np.abs(U.T @ Q @ (
        geo.normal_term_N(Q, dQ, U, dU))))

    # Geometric rebalancing energy for a finite-horizon rebalancing step
    # (Sec. 3.5): the vector connecting consecutive target portfolios.
    Q_next, _, U_next, _ = geo.manufactured_problem_1(t + 0.05)
    v_t = U @ c
    c_next = c + dc * 0.05
    v_next = U_next @ c_next
    delta = v_next - v_t
    energy = geo.rebalancing_energy(Q_next, delta)
    theoretical_energy = float(delta @ Q_next @ delta)

    return {
        "decomposition_sum_ok": bool(np.allclose(total, tangential + normal, atol=1e-13)),
        "tangential_normal_orthogonality_residual": orthogonality_residual,
        "normal_q_orthogonality_residual": normal_q_orthogonal_residual,
        "rebalancing_energy": energy,
        "rebalancing_energy_theoretical": theoretical_energy,
        "rebalancing_energy_matches": abs(energy - theoretical_energy) < 1e-12,
        "rebalancing_energy_positive": energy >= 0.0,
    }


def main() -> dict:
    results = {
        "rk4_order_check": run_rk4_order_check(),
        "gauge_covariance_check": run_gauge_covariance_check(),
        "motion_decomposition_check": run_motion_decomposition_check(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
