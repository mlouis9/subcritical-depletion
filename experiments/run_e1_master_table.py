#!/usr/bin/env python3
"""E1 -- the master scaling table (theory note Sec. 6.3, 7.4).

Runs one Ensemble-I (ordinary closed-loop) exposure-mode depletion
replica to a prescribed exposure window, and records every quantity
needed to populate the master table: the exposure-indexed inventory and
its clock, the endpoint multiplication and a normalized-rate diagnostic
(for the endpoint-local absolute rate row), and -- if ``--t-star`` is
given -- the calendar-indexed inventory delivered by the Sec. 4.4
interpolation algorithm at a prescribed date.

Run once per (case, k_target, replica); aggregate the resulting
summary CSV across replicas with ordinary replica statistics
(``scm_depletion.common.replica_stats``) in a separate, lightweight
post-processing step, exactly as the parent paper's static-case study
aggregates ``bg_summary_*.csv`` files after the fact.
"""

from __future__ import annotations

import numpy as np

from _expcommon import (base_arg_parser, case_and_defaults, replica_cache_dir,
                         append_summary_row, build_operator_and_model,
                         resolve_alpha, transport_workdir)
from scm_depletion.integrators import ExposureIntegrator
from scm_depletion.trajectory import SCMTrajectory


def _total_fission_rate(res) -> float:
    """Best-effort extraction of a normalized total fission rate from an
    SCMOperatorResult, for the R(tau*) diagnostic. This depends on the
    exact ReactionRates indexing of the installed OpenMC build; adjust
    the indexing below to match if it does not apply directly.
    """
    try:
        idx = res.rates.index_rx["fission"]
        return float(sum(mat_rates[:, idx].sum() for mat_rates in res.rates))
    except Exception:
        return float("nan")


def main():
    parser = base_arg_parser(__doc__)
    parser.add_argument("--t-star", type=float, default=None,
                         help="Optional prescribed calendar date at which to "
                              "additionally deliver the interpolated inventory "
                              "(Sec. 4.4). Requires --tau-max large enough to "
                              "cover it; the script extends the run once if not.")
    args = parser.parse_args()

    case, n_particles, n_generations = case_and_defaults(args)
    cache_dir = replica_cache_dir(args.cache_root, "e1_master_table", args)
    workdir = transport_workdir(args.cache_root, "e1_master_table", args)

    tau_max = args.tau_max
    if tau_max is None:
        raise SystemExit("--tau-max is required for E1 (no case-level default "
                          "exposure window is assumed).")

    dtau_grid = [tau_max / args.n_steps] * args.n_steps

    trajectory = SCMTrajectory(cache_dir, metadata={
        "case": args.case, "k_target": args.k_target, "alpha": resolve_alpha(args),
        "replica": args.replica,
        # no single "seed" here: build_operator_and_model derives a
        # resume-dependent seed per process (see its docstring), so
        # there isn't one fixed value describing this whole trajectory.
        "n_particles": n_particles, "n_generations": n_generations,
        "source_strength": case.source_strength,
    })

    with build_operator_and_model(
            case, args, n_particles, n_generations, cache_dir, workdir) as \
            (model, operator, initial_vec):

        integrator = ExposureIntegrator(
            operator, initial_vec, case.source_strength,
            # Flat 40 (doubled from the original 20), not a k-dependent
            # schedule: the theory paper's Sec. 4.3/Appendix C shows
            # r(Gs) -> c < 1 as k -> 1 for this benchmark family (Remark 2),
            # so the required inactive-cycle count for a given source bias
            # is k-independent in the first place -- a flat count was
            # always the theoretically-motivated choice, not a
            # simplification pending a schedule. The bump to 40 is purely
            # to shrink the per-step k_std noise floor identified by
            # diagnose_R_variance.py, not because 20 was insufficient for
            # source convergence.
            n_inactive_schedule=lambda _step: 40,
            diagnostic_fn=lambda res0, res_mid: {"r_fission": _total_fission_rate(res_mid)})

        integrator.run(trajectory, dtau_grid, resume=True)

        if args.t_star is not None:
            while True:
                try:
                    info = trajectory.interpolate_at_time(args.t_star)
                    break
                except Exception:
                    # Window too short (Sec. 4.4, step 4): extend by one more
                    # copy of the original window and continue.
                    integrator.extend(trajectory, dtau_grid)

    last = trajectory.records[-1]
    N_end = last.vec[0]
    t_end = trajectory.t[-1]
    k_end = last.k_mid.k
    M_end = last.k_mid.M
    r_end = last.diagnostics.get("r_fission", float("nan"))
    R_end = case.source_strength * M_end * r_end

    row = {
        "case": args.case, "k_target": args.k_target, "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "n_steps": args.n_steps, "tau_max": tau_max,
        "N_end_sum": float(np.sum(N_end)),
        "t_end": t_end, "k_end": k_end, "M_end": M_end,
        "r_fission_end": r_end, "R_fission_end": R_end,
    }

    if args.t_star is not None:
        row["N_at_t_star_sum"] = float(np.sum(info["N"]))
        row["tau_hat_at_t_star"] = info["tau_hat"]
        row["k_hat_at_t_star"] = info["k_hat"]

    append_summary_row(args.summary_csv, row)
    print(f"[E1] case={args.case} k_target={args.k_target} replica={args.replica} "
          f"-> N_end={row['N_end_sum']:.6e} t_end={t_end:.6e} M_end={M_end:.4f}")


if __name__ == "__main__":
    main()
