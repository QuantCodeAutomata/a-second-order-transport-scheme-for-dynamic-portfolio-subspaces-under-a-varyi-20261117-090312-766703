"""Experiment 2: Manufactured Convergence Order and Local Error Coefficient
Verification (paper Sec. 4.3-4.4, Theorem 3.7, Theorem/Proposition 3.8).

Part A -- global convergence order of the four endpoint transport rules
  (risk weighted Procrustes, principal root map, geometric mean update,
  symmetric Procrustes update) on the 6-asset rank-3 noncommuting path and
  its diagonal commuting control, against a DOP853 reference propagator of
  the continuous connection ODE (Eq. 3.10, 3.12).
Part B -- exact sharpness of the rank-2 Theorem/Proposition 3.8 construction:
  risk weighted Procrustes error/h approaches sin(T)/4 while the geometric
  mean update stays exact (machine precision) for every mesh.
Part C -- local error (defect) coefficient D(t) (Eq. 3.32, Theorem 3.7): the
  leading h^2 discrepancy between risk weighted Procrustes and the geometric
  mean update converges to D(t) as the step size shrinks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from src import config as cfg
from src import geometry as geo

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RULES = {
    "risk_weighted_procrustes": geo.risk_weighted_procrustes,
    "principal_root_map": geo.principal_root_map,
    "geometric_mean_update": geo.geometric_mean_update,
    "symmetric_procrustes_update": geo.symmetric_procrustes_update,
}


def exact_propagator(t0: float, t1: float, commuting: bool, rank: int) -> np.ndarray:
    """Reference coefficient propagator Psi(t1), solving the matrix ODE
    dPsi/dt = -omega(t) Psi(t), Psi(t0) = I_rank (Eq. 3.10, 3.12), by
    integrating the flattened 9-vector ODE with DOP853 at tight tolerance."""
    def rhs(t, y):
        Psi = y.reshape(rank, rank)
        Q, dQ, U, dU = geo.manufactured_problem_2(t, commuting=commuting)
        omega = geo.connection_form(Q, dQ, U, dU)
        return (-omega @ Psi).ravel()

    sol = solve_ivp(rhs, (t0, t1), np.eye(rank).ravel(), method="DOP853",
                     rtol=cfg.EXP2_DOP853_RTOL, atol=cfg.EXP2_DOP853_ATOL)
    assert sol.success
    return sol.y[:, -1].reshape(rank, rank)


def compose_path_rule(rule, qu_nodes: list) -> np.ndarray:
    """Compose an endpoint rule step by step over a sequence of (Q, U) node
    values, R_total = R_N @ ... @ R_1 where R_k = rule(Q_{k-1}, U_{k-1}, Q_k, U_k)."""
    rank = qu_nodes[0][1].shape[1]
    R_total = np.eye(rank)
    for (Q0, U0), (Q1, U1) in zip(qu_nodes[:-1], qu_nodes[1:]):
        R_step = rule(Q0, U0, Q1, U1)
        R_total = R_step @ R_total
    return R_total


def fit_order(step_counts: list, errors: list, span: float) -> float:
    """Fit the observed convergence order from log(error) vs log(h) by
    linear regression on the three finest step counts, h = span / N."""
    hs = np.array([span / n for n in step_counts[-3:]])
    errs = np.array(errors[-3:])
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
    return float(slope)


def run_part_a(commuting: bool, rule_names: list) -> dict:
    t0, t1 = cfg.EXP2_T0, cfg.EXP2_T1
    span = t1 - t0
    Psi_exact = exact_propagator(t0, t1, commuting, cfg.EXP2_RANK)

    rules_out = {}
    for name in rule_names:
        rule = RULES[name]
        errors = []
        for n in cfg.EXP2_STEP_COUNTS:
            t_nodes = np.linspace(t0, t1, n + 1)
            qu_nodes = [geo.manufactured_problem_2(t, commuting=commuting)[0::2]
                        for t in t_nodes]
            R_total = compose_path_rule(rule, qu_nodes)
            errors.append(float(np.linalg.norm(Psi_exact - R_total, 2)))
        rules_out[name] = {
            "errors": errors,
            "order": fit_order(cfg.EXP2_STEP_COUNTS, errors, span),
        }
    return {"step_counts": cfg.EXP2_STEP_COUNTS, "rules": rules_out}


def run_part_b() -> dict:
    T = cfg.EXP2_RANK2_T
    step_counts = cfg.EXP2_RANK2_STEP_COUNTS
    exact = geo.rank2_theorem38_exact_propagator()

    rp_errors, rp_errors_over_h, geo_errors = [], [], []
    for n in step_counts:
        t_nodes = np.linspace(0.0, T, n + 1)
        qu_nodes = [geo.rank2_theorem38_path(t) for t in t_nodes]
        h = T / n

        R_rp = compose_path_rule(geo.risk_weighted_procrustes, qu_nodes)
        R_geo = compose_path_rule(geo.geometric_mean_update, qu_nodes)

        err_rp = float(np.linalg.norm(exact - R_rp, 2))
        err_geo = float(np.linalg.norm(exact - R_geo, 2))
        rp_errors.append(err_rp)
        rp_errors_over_h.append(err_rp / h)
        geo_errors.append(err_geo)

    return {
        "step_counts": step_counts,
        "risk_weighted_procrustes_error": rp_errors,
        "risk_weighted_procrustes_error_over_h": rp_errors_over_h,
        "geometric_mean_error": geo_errors,
        "risk_weighted_procrustes_order": fit_order(step_counts, rp_errors, T),
        "expected_sharpness_constant_sin_T_over_4": float(np.sin(T) / 4.0),
    }


def run_part_c() -> dict:
    t = cfg.EXP2_LOCAL_TIME
    Q0, dQ0, U0, dU0 = geo.manufactured_problem_2(t, commuting=False)
    D = geo.defect_coefficient(Q0, dQ0, U0, dU0)
    D_norm = float(np.linalg.norm(D, 2))

    discrepancies = []
    for h in cfg.EXP2_LOCAL_STEPS:
        Q1, _, U1, _ = geo.manufactured_problem_2(t + h, commuting=False)
        R_rp = geo.risk_weighted_procrustes(Q0, U0, Q1, U1)
        R_geo = geo.geometric_mean_update(Q0, U0, Q1, U1)
        discrepancy = np.linalg.norm((R_rp - R_geo) / h ** 2 - D, 2)
        discrepancies.append(float(discrepancy))

    ratio = discrepancies[0] / discrepancies[1] if discrepancies[1] else float("inf")
    return {
        "t": t,
        "defect_coefficient_norm": D_norm,
        "steps": cfg.EXP2_LOCAL_STEPS,
        "discrepancies": discrepancies,
        "discrepancy_ratio": float(ratio),
        "decreases_with_h": bool(discrepancies[-1] < discrepancies[0]),
    }


def print_summary(results: dict) -> None:
    print("=== Experiment 2: convergence order & local error coefficient ===")
    for key in ("part_a_noncommuting", "part_a_commuting"):
        label = "noncommuting" if key.endswith("noncommuting") else "commuting"
        print(f"\n-- Part A ({label}) -- step counts {results[key]['step_counts']}")
        for name, out in results[key]["rules"].items():
            print(f"  {name:28s} order ~ {out['order']: .3f}  "
                  f"errors[-3:] = {[f'{e:.3e}' for e in out['errors'][-3:]]}")

    b = results["part_b_sharpness"]
    print(f"\n-- Part B (rank-2 Theorem 3.8, T={cfg.EXP2_RANK2_T}) --")
    print(f"  risk_weighted_procrustes order ~ {b['risk_weighted_procrustes_order']: .3f}")
    print(f"  error/h at finest grids: {[f'{e:.5f}' for e in b['risk_weighted_procrustes_error_over_h'][-2:]]}"
          f"  (expected ~ {b['expected_sharpness_constant_sin_T_over_4']:.5f})")
    print(f"  geometric_mean raw error at finest grids: "
          f"{[f'{e:.2e}' for e in b['geometric_mean_error'][-2:]]} (expect ~1e-10 or smaller)")

    c = results["part_c_local_defect"]
    print(f"\n-- Part C (local defect at t={c['t']}) --")
    print(f"  ||D||_2 = {c['defect_coefficient_norm']:.6e}")
    print(f"  discrepancies for h={c['steps']}: {c['discrepancies']}")
    print(f"  ratio (h0/h1 discrepancy) = {c['discrepancy_ratio']:.4f}, "
          f"decreases with h: {c['decreases_with_h']}")


def main() -> dict:
    results = {
        "part_a_noncommuting": run_part_a(commuting=False, rule_names=list(RULES)),
        "part_a_commuting": run_part_a(
            commuting=True,
            rule_names=["risk_weighted_procrustes", "principal_root_map"],
        ),
        "part_b_sharpness": run_part_b(),
        "part_c_local_defect": run_part_c(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "exp2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print_summary(results)
    return results


if __name__ == "__main__":
    main()
