"""Integration tests for exp/exp1.py (paper Sec. 4.2 manufactured problem)."""
from __future__ import annotations

import numpy as np

from exp import exp1


def test_rk4_order_check_is_near_four():
    """The fixed-step RK4 reference integrator must show 4th-order global
    convergence on the manufactured coefficient equation (Eq. 3.12)."""
    result = exp1.run_rk4_order_check()
    orders = result["observed_orders"]
    assert len(orders) >= 2
    # Use a coarse order (far from the float64 roundoff floor) rather than the
    # finest, which degrades toward noise once truncation error is sub-1e-13.
    assert 3.9 < orders[0] < 4.1
    assert result["errors"][0] > result["errors"][-1]


def test_gauge_covariance_near_machine_precision():
    """Random smooth orthogonal gauge transformations must leave the
    transported terminal vector and its risk norm invariant (Sec. 3.3)."""
    result = exp1.run_gauge_covariance_check()
    assert result["max_vector_error"] < 1e-6
    assert result["max_norm_error"] < 1e-6
    assert result["max_projector_gauge_error"] < 1e-10
    assert result["max_connection_skew_error"] < 1e-10


def test_motion_decomposition_orthogonality_and_positive_energy():
    """The orthogonal motion decomposition (Eq. 3.15) must be Q-orthogonal
    and the geometric rebalancing energy must be positive (Sec. 3.5)."""
    result = exp1.run_motion_decomposition_check()
    assert result["decomposition_sum_ok"]
    assert result["tangential_normal_orthogonality_residual"] < 1e-8
    assert result["normal_q_orthogonality_residual"] < 1e-8
    assert result["rebalancing_energy_positive"]
    assert np.isclose(
        result["rebalancing_energy"], result["rebalancing_energy_theoretical"], atol=1e-8
    )
