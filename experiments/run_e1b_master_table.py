#!/usr/bin/env python3
"""E1B -- Method B (branching fixed-source) analog of E1: an exposure-
indexed ("matched burnup") comparison point, for comparing variance
scaling directly against the SCM results (E1).

Method B has no native exposure coordinate to step in (see
matrix_utils.branching_matrix's docstring: it tallies absolute rates
directly, with no S0*M rescaling to remove from the depletion ordinate
in the first place -- there is nothing for an exposure reformulation to
buy here). So this experiment does NOT reimplement exposure stepping
for Method B; instead it steps in calendar time
(BranchingCalendarIntegrator, same as E9B) to a PER-k_target calendar
window chosen so that the resulting run lands close to the SAME
--tau-star used for E1's own --tau-max, using the CALIBRATED k_target
itself (M = 1/(1-k_target)) rather than a pilot run:

    t_max(k_target) = tau_star / (S0 * M(k_target))

This is a DELIBERATE APPROXIMATION, not an exact match to E1's own
delivery: E1's exposure grid lands on tau_max EXACTLY, by construction,
with zero interpolation error, because tau IS the independent variable
it steps in. Here, t_max is fixed in ADVANCE from the NOMINAL M implied
by k_target, and the realized accumulated tau_end (see tau_end in the
output row) is a stochastic quantity that will differ from tau_star by
however much this replica's own M estimate, averaged over the run,
differs from the nominal 1/(1-k_target) -- from calibration precision,
composition-driven reactivity drift over the depletion, or ordinary
statistical noise in the accumulated M estimate. Check tau_end against
--tau-star in the printed output (not just that nothing crashed) before
trusting a full array's worth of replicas to be "at the same burnup" in
the same sense E1's own tau_max is.

No n_inactive_schedule (see E9B's module docstring -- branching
fixed-source has no source-iteration bank to warm up). Expect this to
be substantially more expensive than the equivalent SCM run, especially
near criticality -- see BranchingFixedSourceOperator's own docstring.
PILOT A SINGLE NEAR-CRITICAL REPLICA before submitting a full array;
this has not been run or timed in this environment.
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
    parser.add_argument("--tau-star", type=float, required=True,
                         help="The exposure target to match -- pass the SAME "
                              "value used as E1's --tau-max, so the two "
                              "methods are compared at (approximately) the "
                              "same burnup. See module docstring for how "
                              "this is converted to a per-k_target t_max.")
    args = parser.parse_args()

    case, n_particles, n_generations = case_and_defaults(args)
    cache_dir = replica_cache_dir(args.cache_root, "e1b_master_table", args)
    workdir = transport_workdir(args.cache_root, "e1b_master_table", args)

    S0 = case.source_strength
    M_nominal = 1.0 / (1.0 - args.k_target)
    t_max = args.tau_star / (S0 * M_nominal)
    dt_grid = [t_max / args.n_steps] * args.n_steps

    trajectory = SCMTrajectory(cache_dir, metadata={
        "case": args.case, "k_target": args.k_target, "alpha": resolve_alpha(args),
        "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "source_strength": S0, "tau_star": args.tau_star, "t_max": t_max,
        "method": "B",
    })

    with build_operator_and_model(
            case, args, n_particles, n_generations, cache_dir, workdir,
            run_mode="fixed source",
            operator_cls=BranchingFixedSourceOperator,
            calculate_subcritical_k=True) as \
            (model, operator, initial_vec):

        integrator = BranchingCalendarIntegrator(operator, initial_vec, S0)
        integrator.run(trajectory, dt_grid, resume=True)

    last = trajectory.records[-1]
    N_end = last.vec[0]
    t_end = trajectory.t[-1]           # deterministic: == t_max, always
    tau_end = trajectory.tau[-1]       # stochastic: see module docstring for
                                        # how far this can land from tau_star
    k_end = last.k_mid.k
    M_end = last.k_mid.M
    R_end = S0 * M_end

    row = {
        "case": args.case, "k_target": args.k_target, "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "n_steps": args.n_steps, "tau_star": args.tau_star, "t_max": t_max,
        "N_end_sum": float(np.sum(N_end)),
        "t_end": t_end, "tau_end": tau_end, "k_end": k_end, "M_end": M_end,
        "R_end": R_end,
    }
    append_summary_row(args.summary_csv, row)
    rel_miss = abs(tau_end - args.tau_star) / args.tau_star
    print(f"[E1B] case={args.case} k_target={args.k_target} replica={args.replica} "
          f"t_max={t_max:.4e} -> N_end={row['N_end_sum']:.6e} "
          f"tau_end={tau_end:.6e} (target {args.tau_star:.4e}, "
          f"{rel_miss:.2%} off) M_end={M_end:.4f}")


if __name__ == "__main__":
    main()
