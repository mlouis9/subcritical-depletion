#!/usr/bin/env python3
"""E9B -- Method B (branching fixed-source) analog of E9: a fixed-
calendar-duration sweep, for comparing variance scaling directly
against the SCM results (E9).

Structurally identical to E9 -- same fixed --t-max shared across the
whole k-sweep, same reported quantities -- but transport is ordinary
branching fixed-source Monte Carlo (BranchingFixedSourceOperator, no
population control, no rescaling), stepped with
BranchingCalendarIntegrator instead of CalendarIntegrator. The theory
note's Method B: reaction rates are tallied directly in absolute units
(reactions/second), with no S0*M rescaling anywhere -- see
matrix_utils.branching_matrix. k/kq/ks/M are STILL read off the same
C-API call SCM uses (model.settings.calculate_subcritical_k = True
makes this valid outside SUBCRITICAL_MULTIPLICATION mode -- see
scm_transport._fetch_k_factors' docstring), but here they are a SIDE
quantity for comparison against Method A/C, not something the
transmutation update depends on at all.

No n_inactive_schedule is passed here (unlike E1/E9): branching
fixed-source has no source-iteration bank to warm up at all -- each of
--n-particles starting histories is followed through its ENTIRE
branching descendant tree within one batch, not carried forward
generation-to-generation. This is also why Method B is expected to cost
substantially more than the equivalent SCM run at the same
(n_particles, n_generations), increasingly so near criticality (each
starting history's own descendant tree grows with M, where SCM's total
work stays fixed) -- see BranchingFixedSourceOperator's own docstring.
PILOT A SINGLE NEAR-CRITICAL REPLICA (e.g. k=0.99 or 0.995) before
submitting a full array; this has not been run or timed in this
environment.
"""

from __future__ import annotations

import numpy as np

from _expcommon import (base_arg_parser, case_and_defaults, replica_cache_dir,
                         append_summary_row, build_operator_and_model,
                         resolve_alpha, transport_workdir)
from scm_depletion.integrators import BranchingCalendarIntegrator
from scm_depletion.scm_transport import BranchingFixedSourceOperator
from scm_depletion.trajectory import SCMTrajectory


def main():
    parser = base_arg_parser(__doc__)
    parser.add_argument("--t-max", type=float, required=True,
                         help="Calendar duration to step to, directly. MUST "
                              "be passed as the IDENTICAL literal value for "
                              "every k_target in a given sweep, and SHOULD "
                              "match the --t-max used for the corresponding "
                              "E9 run, so the two methods are compared at "
                              "the same calendar endpoint.")
    args = parser.parse_args()

    case, n_particles, n_generations = case_and_defaults(args)
    cache_dir = replica_cache_dir(args.cache_root, "e9b_fixed_duration", args)
    workdir = transport_workdir(args.cache_root, "e9b_fixed_duration", args)

    dt_grid = [args.t_max / args.n_steps] * args.n_steps

    trajectory = SCMTrajectory(cache_dir, metadata={
        "case": args.case, "k_target": args.k_target, "alpha": resolve_alpha(args),
        "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "source_strength": case.source_strength,
        "t_max": args.t_max, "method": "B",
    })

    with build_operator_and_model(
            case, args, n_particles, n_generations, cache_dir, workdir,
            run_mode="fixed source",
            operator_cls=BranchingFixedSourceOperator,
            calculate_subcritical_k=True) as \
            (model, operator, initial_vec):

        integrator = BranchingCalendarIntegrator(operator, initial_vec,
                                                   case.source_strength)
        integrator.run(trajectory, dt_grid, resume=True)

    last = trajectory.records[-1]
    N_end = last.vec[0]
    t_end = trajectory.t[-1]           # deterministic: == args.t_max, always
    tau_end = trajectory.tau[-1]       # stochastic side quantity (see module
                                        # docstring): this replica's own
                                        # accumulated exposure by t_max
    k_end = last.k_mid.k
    M_end = last.k_mid.M
    S0 = case.source_strength
    R_end = S0 * M_end                 # same convention as E9, not an
                                        # extracted rate diagnostic

    row = {
        "case": args.case, "k_target": args.k_target, "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "n_steps": args.n_steps, "t_max": args.t_max,
        "N_end_sum": float(np.sum(N_end)),
        "t_end": t_end, "tau_end": tau_end, "k_end": k_end, "M_end": M_end,
        "R_end": R_end,
    }
    append_summary_row(args.summary_csv, row)
    print(f"[E9B] case={args.case} k_target={args.k_target} replica={args.replica} "
          f"t_max={args.t_max:.4e} -> N_end={row['N_end_sum']:.6e} "
          f"tau_end={tau_end:.6e} M_end={M_end:.4f}")


if __name__ == "__main__":
    main()
