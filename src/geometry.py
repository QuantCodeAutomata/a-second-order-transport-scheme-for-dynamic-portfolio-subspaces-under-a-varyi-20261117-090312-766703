"""Core differential-geometric transport scheme for dynamic portfolio
subspaces under a varying risk metric.

Implements the selected ambient connection, its induced connection form,
continuous parallel transport, the orthogonal motion decomposition, and the
four endpoint transport rules compared in the paper (risk weighted
Procrustes, principal square root whitening, geometric mean alignment,
symmetric Procrustes), together with the local error (defect) coefficient
D(t) of Theorem 3.7.

All matrix power and polar-decomposition primitives are the custom
formulas of the paper; `scipy.linalg.polar` and `scipy.linalg.sqrtm`
(confirmed as the appropriate library primitives, no higher-level package
implements this transport scheme) are used for the underlying linear
algebra. (scipy.linalg polar/sqrtm)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import polar, sqrtm, fractional_matrix_power


def projector(Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Oblique projector P = U U^T Q onto the represented subspace (Sec. 3.3)."""
    return U @ U.T @ Q


def q_orthonormalize(Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Return a Q-orthonormal frame spanning the same subspace as U.

    Rescales columns so that U^T Q U = I_p, following the normalization
    used throughout Sec. 3 (e.g. the risk weighted Procrustes whitening
    U0(U0^T Q1 U0)^{-1/2} in Eq. 3.19).
    """
    gram = U.T @ Q @ U
    gram = 0.5 * (gram + gram.T)
    inv_sqrt = _spd_inv_sqrt(gram)
    return U @ inv_sqrt


def _spd_inv_sqrt(A: np.ndarray) -> np.ndarray:
    """Inverse square root of a symmetric positive definite matrix via its
    eigendecomposition (stable for the small p x p Gram matrices used here)."""
    w, v = np.linalg.eigh(0.5 * (A + A.T))
    assert np.all(w > 0), "Gram matrix must be positive definite"
    return (v * (1.0 / np.sqrt(w))) @ v.T


def _spd_sqrt(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (A + A.T))
    assert np.all(w > -1e-10), "matrix must be positive semidefinite"
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def _spd_frechet_derivative(A: np.ndarray, dA: np.ndarray, f, fprime) -> np.ndarray:
    """Fréchet derivative of a scalar matrix function f applied to a
    symmetric matrix A, in the direction dA, via the Daleckii-Krein /
    Loewner divided-difference formula built from A's eigendecomposition.

    Used to differentiate the Q-orthonormalizing inverse square root of a
    manufactured problem's Gram matrix analytically (no finite differences),
    so the manufactured derivatives dQ/dt, dU/dt feeding the convergence
    studies of Sec. 4.2-4.4 are exact to machine precision rather than
    limited by a finite-difference truncation/roundoff floor.
    """
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    fw = f(w)
    denom = w[:, None] - w[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        loewner = np.where(np.abs(denom) < 1e-10,
                            fprime(0.5 * (w[:, None] + w[None, :])),
                            (fw[:, None] - fw[None, :]) / np.where(denom == 0, 1.0, denom))
    dA_tilde = V.T @ dA @ V
    return V @ (loewner * dA_tilde) @ V.T


def _d_inv_sqrt(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """Analytic derivative of A^{-1/2} for symmetric positive definite A."""
    return _spd_frechet_derivative(A, dA, f=lambda x: x ** -0.5,
                                    fprime=lambda x: -0.5 * x ** -1.5)


def connection_form(Q: np.ndarray, dQ: np.ndarray, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
    """Induced connection form omega_a = U^T Q dU + 0.5 U^T dQ U (Eq. 3.10).

    `dQ` and `dU` are directional derivatives of Q and U (e.g. along a
    curve parameter t). The returned matrix is skew-symmetric by
    Proposition 3.2.
    """
    omega = U.T @ Q @ dU + 0.5 * U.T @ dQ @ U
    return omega


def ambient_min_norm_coefficient(Q: np.ndarray, dQ: np.ndarray) -> np.ndarray:
    """Ambient minimum-norm connection coefficient A^min = 0.5 Q^-1 dQ/dt.

    This is the constant-asset-coordinate selector minimizing the
    instantaneous coefficient norm (Sec. 3.2/Proposition 3.5's geodesic
    derivation), used in the normal covariant motion term N (Eq. 3.31).
    """
    return 0.5 * np.linalg.solve(Q, dQ)


def normal_term_N(Q: np.ndarray, dQ: np.ndarray, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
    """Normal differential of the frame, N_a = (I - P) D_a^min U (Eq. 3.14),
    equivalently N = (I - P)(dU/dt + A^min U) (Eq. 3.31)."""
    P = projector(Q, U)
    n = Q.shape[0]
    A_min = ambient_min_norm_coefficient(Q, dQ)
    return (np.eye(n) - P) @ (dU + A_min @ U)


def normal_term_B(Q: np.ndarray, dQ: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Normal metric variation, B = (I - P) Q^-1 dQ/dt U (Eq. 3.31)."""
    P = projector(Q, U)
    n = Q.shape[0]
    return (np.eye(n) - P) @ np.linalg.solve(Q, dQ) @ U


def defect_coefficient(Q: np.ndarray, dQ: np.ndarray, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
    """Local error (defect) coefficient D = 1/4 (N^T Q B - B^T Q N) (Eq. 3.32).

    D is skew-symmetric: it is the leading h^2 term separating risk
    weighted Procrustes from the exact continuous transport and from the
    geometric mean update (Theorem 3.7, Eqs. 3.33-3.35).
    """
    N = normal_term_N(Q, dQ, U, dU)
    B = normal_term_B(Q, dQ, U)
    return 0.25 * (N.T @ Q @ B - B.T @ Q @ N)


def motion_decomposition(Q: np.ndarray, dQ: np.ndarray, U: np.ndarray, dU: np.ndarray,
                          c: np.ndarray, dc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthogonal motion decomposition of D_t^min v for v = U c (Eq. 3.15).

    Returns (tangential, normal, total) asset-space vectors with
    total = tangential + normal and tangential Q-orthogonal to normal.
    """
    omega = connection_form(Q, dQ, U, dU)
    tangential = U @ (dc + omega @ c)
    N = normal_term_N(Q, dQ, U, dU)
    normal = N @ c
    return tangential, normal, tangential + normal


def continuous_transport_rhs(c: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Right-hand side of dc/dt + omega(t) c(t) = 0 (Eq. 3.12)."""
    return -omega @ c


# ---------------------------------------------------------------------------
# Endpoint transport rules (Sec. 3.6)
# ---------------------------------------------------------------------------

def risk_weighted_procrustes(Q0: np.ndarray, U0: np.ndarray, Q1: np.ndarray, U1: np.ndarray,
                              sv_tol: float = 1e-10) -> np.ndarray:
    """Risk weighted Procrustes coefficient map R_RP (Eq. 3.19).

    U0_tilde = U0 (U0^T Q1 U0)^{-1/2};  R_RP = polar(U1^T Q1 U0_tilde).
    """
    gram = U0.T @ Q1 @ U0
    gram = 0.5 * (gram + gram.T)
    U0_tilde = U0 @ _spd_inv_sqrt(gram)
    overlap = U1.T @ Q1 @ U0_tilde
    _check_overlap(overlap, sv_tol)
    R, _ = polar(overlap)
    return R


def principal_root_map(Q0: np.ndarray, U0: np.ndarray, Q1: np.ndarray, U1: np.ndarray,
                        sv_tol: float = 1e-10) -> np.ndarray:
    """Principal square root whitening coefficient map (Eq. 3.20).

    T_root = Q1^{-1/2} Q0^{1/2}; coefficient update uses the same endpoint
    overlap pattern as the other rules, R_root = polar(U1^T Q1 T_root U0).
    """
    Q1_sqrt = _spd_sqrt(Q1)
    Q1_inv_sqrt = _spd_inv_sqrt(Q1)
    Q0_sqrt = _spd_sqrt(Q0)
    T_root = Q1_inv_sqrt @ Q0_sqrt
    overlap = U1.T @ Q1 @ T_root @ U0
    _check_overlap(overlap, sv_tol)
    R, _ = polar(overlap)
    return R


def matrix_geometric_mean(Q0: np.ndarray, Q1: np.ndarray) -> np.ndarray:
    """Standard matrix geometric mean Q0 # Q1 (Proposition 3.5).

    Q0 # Q1 = Q0^{1/2} (Q0^{-1/2} Q1 Q0^{-1/2})^{1/2} Q0^{1/2}.
    """
    Q0_sqrt = _spd_sqrt(Q0)
    Q0_inv_sqrt = _spd_inv_sqrt(Q0)
    G = Q0_inv_sqrt @ Q1 @ Q0_inv_sqrt
    G_sqrt = _spd_sqrt(G)
    return Q0_sqrt @ G_sqrt @ Q0_sqrt


def geometric_mean_update(Q0: np.ndarray, U0: np.ndarray, Q1: np.ndarray, U1: np.ndarray,
                           sv_tol: float = 1e-10) -> np.ndarray:
    """Geometric mean coefficient map R_geo (Eq. 3.27-3.28).

    R_geo = polar(U1^T (Q0 # Q1) U0).
    """
    QG = matrix_geometric_mean(Q0, Q1)
    overlap = U1.T @ QG @ U0
    _check_overlap(overlap, sv_tol)
    R, _ = polar(overlap)
    return R


def symmetric_procrustes_update(Q0: np.ndarray, U0: np.ndarray, Q1: np.ndarray, U1: np.ndarray,
                                 sv_tol: float = 1e-10) -> np.ndarray:
    """Symmetric Procrustes coefficient map R_sym (Eq. 3.38).

    R_sym = polar((F + B^T) / 2), with F = R_RP(t1, t0) the forward update
    and B = R_RP(t0, t1) the backward update.
    """
    F = risk_weighted_procrustes(Q0, U0, Q1, U1, sv_tol=sv_tol)
    B = risk_weighted_procrustes(Q1, U1, Q0, U0, sv_tol=sv_tol)
    M = 0.5 * (F + B.T)
    _check_overlap(M, sv_tol)
    R, _ = polar(M)
    return R


def ambient_projection(U1: np.ndarray, Q1: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Ambient projection of an asset-space vector, v_amb = U1 U1^T Q1 v0 (Eq. C.2)."""
    return U1 @ (U1.T @ Q1 @ v0)


def euclidean_procrustes(U0: np.ndarray, U1: np.ndarray, sv_tol: float = 1e-10) -> np.ndarray:
    """Euclidean (metric-free) Procrustes coefficient map, R_E = polar(U1^T U0) (Eq. C.2)."""
    overlap = U1.T @ U0
    _check_overlap(overlap, sv_tol)
    R, _ = polar(overlap)
    return R


def _check_overlap(overlap: np.ndarray, sv_tol: float) -> None:
    """Reject endpoint overlaps whose smallest singular value is below
    `sv_tol` times the largest (Appendix C.1)."""
    svals = np.linalg.svd(overlap, compute_uv=False)
    if svals[-1] < sv_tol * svals[0]:
        raise ValueError("Endpoint overlap singular; rejected per Appendix C.1 rule")


def integrate_continuous_transport(omega_fn, c0: np.ndarray, t0: float, t1: float,
                                    rtol: float = 1e-10, atol: float = 1e-12):
    """Reference solution of dc/dt + omega(t) c(t) = 0 (Eq. 3.12).

    `omega_fn(t)` returns the connection form at time t. Uses
    `scipy.integrate.solve_ivp` with an explicit Runge-Kutta scheme
    (RK45/DOP853 share the classical 4th/8th order RK family; RK45 reduces
    to classical 4th order accuracy) as an independent reference
    integrator, never the endpoint transport rules under test.
    (scipy.integrate.solve_ivp)
    """
    from scipy.integrate import solve_ivp

    def rhs(t, c):
        return continuous_transport_rhs(c, omega_fn(t))

    sol = solve_ivp(rhs, (t0, t1), c0, method="RK45", rtol=rtol, atol=atol,
                     dense_output=True)
    if not sol.success:
        raise RuntimeError(f"RK4/5 integration failed: {sol.message}")
    return sol


def random_gauge_transform(seed: int, rank: int = 2, t_scale: float = 1.0):
    """Random smooth orthogonal gauge path R(t) = expm(f(t) * K) for a fixed
    random skew-symmetric generator K (size `rank x rank`) and a smooth
    scalar time function f(t).

    Returns a callable R(t) and its exact derivative dR(t), used to test
    gauge covariance of the frame bundle of Q-orthonormal rank-`rank`
    frames (Sec. 3.3). A gauge transformation changes the frame by right
    multiplication, U -> U(t) R(t), leaving Q and the represented subspace
    untouched; the induced connection form then transforms as
    omega -> R^T omega R + R^T dR/dt (since U^T Q U = I_rank), and the
    coefficient equation dc/dt + omega c = 0 satisfies the exact identity
    c_gauge(t) = R(t)^T c(t), so the transported asset-space vector
    U(t) R(t) c_gauge(t) = U(t) c(t) and its Q-risk norm are gauge
    invariant.
    """
    rng = np.random.default_rng(seed)
    n = rank
    raw = rng.standard_normal((n, n))
    K = raw - raw.T  # skew-symmetric generator
    freq = t_scale * (0.5 + rng.uniform(0, 1.0))
    phase = rng.uniform(0, 2 * np.pi)

    def f(t):
        return np.sin(freq * t + phase)

    def df(t):
        return freq * np.cos(freq * t + phase)

    def R(t):
        return _expm_skew(f(t) * K)

    def dR(t):
        # d/dt expm(f(t) K) = f'(t) K expm(f(t) K) since K is constant and
        # commutes with itself.
        return df(t) * (K @ R(t))

    return R, dR


def rebalancing_energy(Q: np.ndarray, delta: np.ndarray) -> float:
    """Geometric rebalancing energy ||delta||_Q^2 = delta^T Q delta for an
    asset-space rebalancing vector delta (Sec. 3.5 quadratic transaction
    cost / Riemannian trading-cost proxy)."""
    value = float(delta @ Q @ delta)
    assert value >= -1e-12, "rebalancing energy must be nonnegative (Q is SPD)"
    return max(value, 0.0)


# ---------------------------------------------------------------------------
# Manufactured problem 1: 4-asset, rank-2, nonconstant metric (Sec. 4.2)
# ---------------------------------------------------------------------------

def manufactured_problem_1(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic 4-asset, rank-2 manufactured path (Q(t), U(t)) with
    U(t)^T Q(t) U(t) = I_2, together with their exact time derivatives.

    The paper does not give explicit formulas for this manufactured
    problem; the construction below is designed from scratch to satisfy
    the stated constraints: a nonconstant, uniformly positive definite
    4x4 metric and a smooth rank-2 Q-orthonormal frame, both given by
    closed-form analytic expressions so that exact derivatives are
    available.

    Q(t) = I_4 + 0.3*sin(t) * S1 + 0.2*cos(2t) * S2 for fixed symmetric
    S1, S2 chosen to keep Q uniformly positive definite on [0, 2].
    The raw frame is a smooth rotation-like path, Q-orthonormalized
    analytically through its Gram matrix (itself a smooth function of t,
    so its inverse square root is smooth and differentiated via the
    autograd-free finite-difference-free closed form below).
    """
    S1 = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    S2 = np.array([[1, 0, 0, 1], [0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -1]], dtype=float)
    Q = np.eye(4) + 0.3 * np.sin(t) * S1 + 0.2 * np.cos(2 * t) * S2
    dQ = 0.3 * np.cos(t) * S1 - 0.4 * np.sin(2 * t) * S2

    raw = np.array([
        [np.cos(t), -np.sin(0.5 * t)],
        [np.sin(t), np.cos(0.5 * t)],
        [0.5 * np.cos(0.3 * t), 0.4 * np.sin(t)],
        [0.3 * np.sin(0.7 * t), 0.6 * np.cos(0.4 * t)],
    ])
    d_raw = np.array([
        [-np.sin(t), -0.5 * np.cos(0.5 * t)],
        [np.cos(t), -0.5 * np.sin(0.5 * t)],
        [-0.15 * np.sin(0.3 * t), 0.4 * np.cos(t)],
        [0.21 * np.cos(0.7 * t), -0.24 * np.sin(0.4 * t)],
    ])

    gram = raw.T @ Q @ raw
    gram = 0.5 * (gram + gram.T)
    inv_sqrt = _spd_inv_sqrt(gram)
    U = raw @ inv_sqrt

    # Analytic derivative of U = raw @ gram^{-1/2} via the product rule and
    # the Fréchet derivative of the matrix inverse square root (exact to
    # machine precision, avoiding a finite-difference truncation floor that
    # would otherwise mask the Runge-Kutta convergence order under
    # refinement, Sec. 4.2).
    d_gram = d_raw.T @ Q @ raw + raw.T @ dQ @ raw + raw.T @ Q @ d_raw
    d_inv_sqrt = _d_inv_sqrt(gram, d_gram)
    dU = d_raw @ inv_sqrt + raw @ d_inv_sqrt

    return Q, dQ, U, dU


# ---------------------------------------------------------------------------
# Manufactured problem 2: 6-asset, rank-3 noncommuting path (Sec. 4.3)
# ---------------------------------------------------------------------------

def _skew(n: int, t: float, seed_matrix: np.ndarray) -> np.ndarray:
    return t * seed_matrix


def manufactured_problem_2(t: float, commuting: bool = False):
    """6-asset, rank-3 manufactured path with rotating eigenspaces and
    unequal, time-varying eigenvalues (noncommuting case), or a diagonal
    path providing the commuting control (Sec. 4.3).

    As with manufactured_problem_1, the paper specifies the qualitative
    requirements (6 asset coordinates, rank-3 subspace, rotating
    eigenspaces with unequal varying eigenvalues for the noncommuting
    path, a diagonal path for the commuting control) but not explicit
    formulas; the construction below satisfies them with closed-form,
    analytically differentiable expressions.
    """
    n = 6
    eigs = np.array([1.0 + 0.4 * np.sin(t + i) for i in range(n)]) + 0.1 * np.arange(n)
    deigs = np.array([0.4 * np.cos(t + i) for i in range(n)])

    if commuting:
        Q = np.diag(eigs)
        dQ = np.diag(deigs)
    else:
        # Rotating eigenbasis: product of two commuting-axis Givens-like
        # rotations whose angle grows with t, giving a smooth orthogonal
        # V(t) with nontrivial dV/dt (hence [Q, dQ] != 0 in general).
        A = np.zeros((n, n))
        A[0, 1], A[1, 0] = 1.0, -1.0
        A[2, 3], A[3, 2] = 0.7, -0.7
        A[4, 5], A[5, 4] = 0.5, -0.5
        V = _expm_skew(A * t)
        dV = A @ V
        Q = V @ np.diag(eigs) @ V.T
        dQ = dV @ np.diag(eigs) @ V.T + V @ np.diag(deigs) @ V.T + V @ np.diag(eigs) @ dV.T

    def raw_frame(tt):
        # Smooth diagonally-dominant frame mixing all six coordinates
        # (each column has a distinct dominant pair of rows plus small
        # off-block couplings at other frequencies), so the represented
        # subspace is not confined to the rotation-invariant blocks of the
        # eigenbasis and the Gram matrix stays well conditioned.
        return np.array([
            [np.cos(tt), -np.sin(0.5 * tt), 0.3 * np.cos(0.2 * tt)],
            [np.sin(tt), np.cos(0.5 * tt), 0.2 * np.sin(0.9 * tt)],
            [0.5 * np.cos(0.3 * tt), 0.4 * np.sin(tt), np.cos(0.6 * tt)],
            [0.3 * np.sin(0.7 * tt), 0.6 * np.cos(0.4 * tt), -np.sin(0.6 * tt)],
            [0.2 * np.cos(0.9 * tt), 0.3 * np.sin(0.6 * tt), 0.5 * np.cos(tt)],
            [0.4 * np.sin(0.2 * tt), 0.5 * np.cos(0.8 * tt), 0.4 * np.sin(0.5 * tt)],
        ])

    def d_raw_frame(tt):
        # Exact analytic derivative of raw_frame, term by term.
        return np.array([
            [-np.sin(tt), -0.5 * np.cos(0.5 * tt), -0.06 * np.sin(0.2 * tt)],
            [np.cos(tt), -0.5 * np.sin(0.5 * tt), 0.18 * np.cos(0.9 * tt)],
            [-0.15 * np.sin(0.3 * tt), 0.4 * np.cos(tt), -0.6 * np.sin(0.6 * tt)],
            [0.21 * np.cos(0.7 * tt), -0.24 * np.sin(0.4 * tt), -0.6 * np.cos(0.6 * tt)],
            [-0.18 * np.sin(0.9 * tt), 0.18 * np.cos(0.6 * tt), -0.5 * np.sin(tt)],
            [0.08 * np.cos(0.2 * tt), -0.4 * np.sin(0.8 * tt), 0.2 * np.cos(0.5 * tt)],
        ])

    raw = raw_frame(t)
    d_raw = d_raw_frame(t)
    gram = raw.T @ Q @ raw
    gram = 0.5 * (gram + gram.T)
    inv_sqrt = _spd_inv_sqrt(gram)
    U = raw @ inv_sqrt

    # Analytic derivative (product rule + Fréchet derivative of the inverse
    # square root), avoiding the finite-difference floor that would mask
    # the convergence orders under refinement (Sec. 4.3-4.4).
    d_gram = d_raw.T @ Q @ raw + raw.T @ dQ @ raw + raw.T @ Q @ d_raw
    d_inv_sqrt = _d_inv_sqrt(gram, d_gram)
    dU = d_raw @ inv_sqrt + raw @ d_inv_sqrt

    return Q, dQ, U, dU


def _expm_skew(A: np.ndarray) -> np.ndarray:
    """Matrix exponential of a skew-symmetric matrix via eigendecomposition
    of iA (Hermitian), returning a real orthogonal matrix."""
    from scipy.linalg import expm
    return expm(A)


# ---------------------------------------------------------------------------
# Theorem/Proposition 3.8: exact rank-2 construction (Sec. 4.4, Eq. 3.36)
# ---------------------------------------------------------------------------

def rank2_theorem38_path(t: float):
    """Exact rank-2 construction of Proposition 3.8 (Eq. 3.36):

    a = (e1+e3)/sqrt(2), b = (e1-e3)/sqrt(2),
    L = b e2^T - e2 b^T, Q(t) = diag(e^t, 1, e^-t),
    U(t) = Q(t)^{-1/2} e^{tL} [a, e2].

    Returns Q(t), U(t) and the exact coefficient propagator, which is the
    identity matrix for all t in (0, pi/2) per Proposition 3.8.
    """
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])
    a = (e1 + e3) / np.sqrt(2)
    b = (e1 - e3) / np.sqrt(2)
    L = np.outer(b, e2) - np.outer(e2, b)

    Q = np.diag([np.exp(t), 1.0, np.exp(-t)])
    Q_inv_sqrt = np.diag([np.exp(-t / 2), 1.0, np.exp(t / 2)])
    expo = _expm_skew(t * L)
    U = Q_inv_sqrt @ expo @ np.column_stack([a, e2])
    return Q, U


def rank2_theorem38_exact_propagator() -> np.ndarray:
    """Exact continuous coefficient propagator for the Theorem 3.8 path:
    Psi(t+h, t) = I_2 for all t in (0, pi/2) (Proposition 3.8 proof)."""
    return np.eye(2)
