#!/usr/bin/env python3
"""E9 -- a properly-controlled fixed-calendar-duration sweep (theory
note Sec. 6.1, master table rows "N(t*)"/"R(t*), fixed calendar
duration").

E1's --t-star mechanism retrofits a fixed calendar target onto a
trajectory that was run to a fixed EXPOSURE window (tau_max held fixed
across the k-sweep) -- meaning the exposure grid, and hence how much of
each replica's own dynamic range t* actually represents, is wildly
different at the low-k and high-k ends of a wide sweep. That's not a
data problem, it's a mismatch between what was run and what the row
asks for, and no amount of re-plotting fixes it (see the prior several
turns' diagnosis).

E9 does this properly: it steps in CALENDAR time directly
(CalendarIntegrator, exactly what Method A already does), with ONE
--t-max value used identically for every task in the sweep, whatever
k_target. Every replica of every k_target lands on that same date by
construction -- no interpolation, no window-extension retry, no
mismatch. The clock (dtau accumulated) is the free/stochastic quantity
here, exactly mirroring how t_end was the free quantity for E1's
exposure integration (Sec. 6.1's dual clock structure).

R(t*) uses S0*M directly, not an extracted fission-rate diagnostic --
see scaling_table_figure.py's comment on why _total_fission_rate is not
trusted for this.

Run once per (case, k_target, replica), SAME --t-max for every task in
the sweep; aggregate with ordinary replica statistics afterward, same
as every other E-numbered experiment in this package. Per-nuclide
extraction is a separate step (extract_nuclide.py --experiment
e9_fixed_duration), not done here -- this only ever writes N_end_sum,
matching E1's own division of labor.
"""

from __future__ import annotations

import numpy as np

from _expcommon import (base_arg_parser, case_and_defaults, replica_cache_dir,
                         append_summary_row, build_operator_and_model,
                         resolve_alpha, transport_workdir)
from scm_depletion.integrators import CalendarIntegrator
from scm_depletion.trajectory import SCMTrajectory


def main():
    parser = base_arg_parser(__doc__)
    parser.add_argument("--t-max", type=float, required=True,
                         help="Calendar duration to step to, directly, via "
                              "CalendarIntegrator. MUST be passed as the "
                              "IDENTICAL literal value for every k_target in "
                              "a given sweep -- that shared value is what "
                              "makes this a fixed-calendar-duration "
                              "comparison at all. Pick it as some reasonable "
                              "fraction of the dynamic range you care about; "
                              "unlike E1's --t-star there is no window to run "
                              "out of, since calendar stepping just runs "
                              "however much exposure it takes to get there, "
                              "whatever that turns out to be for a given k.")
    args = parser.parse_args()

    case, n_particles, n_generations = case_and_defaults(args)
    cache_dir = replica_cache_dir(args.cache_root, "e9_fixed_duration", args)
    workdir = transport_workdir(args.cache_root, "e9_fixed_duration", args)

    dt_grid = [args.t_max / args.n_steps] * args.n_steps

    trajectory = SCMTrajectory(cache_dir, metadata={
        "case": args.case, "k_target": args.k_target, "alpha": resolve_alpha(args),
        "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "source_strength": case.source_strength,
        "t_max": args.t_max,
    })

    with build_operator_and_model(
            case, args, n_particles, n_generations, cache_dir, workdir) as \
            (model, operator, initial_vec):

        integrator = CalendarIntegrator(operator, initial_vec, case.source_strength)
        integrator.run(trajectory, dt_grid, resume=True)

    last = trajectory.records[-1]
    N_end = last.vec[0]
    t_end = trajectory.t[-1]           # deterministic: == args.t_max, always
    tau_end = trajectory.tau[-1]       # stochastic: this replica's own
                                        # accumulated exposure by t_max
    k_end = last.k_mid.k
    M_end = last.k_mid.M
    S0 = case.source_strength
    R_end = S0 * M_end                 # S0*M directly -- see module docstring

    row = {
        "case": args.case, "k_target": args.k_target, "replica": args.replica,
        "n_particles": n_particles, "n_generations": n_generations,
        "n_steps": args.n_steps, "t_max": args.t_max,
        "N_end_sum": float(np.sum(N_end)),
        "t_end": t_end, "tau_end": tau_end, "k_end": k_end, "M_end": M_end,
        "R_end": R_end,
    }
    append_summary_row(args.summary_csv, row)
    print(f"[E9] case={args.case} k_target={args.k_target} replica={args.replica} "
          f"t_max={args.t_max:.4e} -> N_end={row['N_end_sum']:.6e} "
          f"tau_end={tau_end:.6e} M_end={M_end:.4f}")


if __name__ == "__main__":
    main()
