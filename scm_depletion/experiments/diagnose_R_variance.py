#!/usr/bin/env python3
"""Decompose Var[R(t*)] = Var[S0*M_hat] into its two theoretically
distinct sources, per k_target, to check why the fitted calendar-indexed
R(t*) exponents in fig_variance_scaling.pdf can come out well below the
paper's M^4 (matched burnup) / M^6 (fixed duration) predictions.

Background (see trajectory.py's interpolate_at_time / k_at_tau):
R(t*) = S0 * M_hat(tau_hat), with M_hat = 1/(1 - k_hat) and k_hat a
LINEAR interpolation, at the replica's own tau_hat(t*), of the per-step
midpoint k estimate k_mid.k -- a single-transport-solve point estimate
with its own statistical noise (k_mid.k_std), independent of exposure.
Two distinct mechanisms can make R(t*) vary across replicas at the same
t*:

  (1) CLOCK SPREAD -- replicas reach t* at different accumulated
      exposure tau_hat (the paper's Eq. 17-ish: Var(tau_hat) ~ M^2 for a
      matched-time endpoint). Evaluating k(tau) at different tau_hat
      picks up different points along the (slowly, but nonzero-slope)
      evolving k(tau) trajectory: Var(k from this route) ~ (dk/dtau)^2
      * Var(tau_hat). This is the mechanism the paper's own longitudinal-
      displacement analysis is about, and it is what should carry the
      extra M^2 (matched time) / M^4 (duration) beyond the M^2 floor
      common to every R row.

  (2) LOCAL ESTIMATOR NOISE -- k_mid.k_std at whichever step tau_hat
      happens to land near, independent of tau_hat's OWN spread. This
      is the single-estimator-noise term present for every R row,
      including the exposure-indexed R(tau*) in panel (a).

If (2) dominates (1) within this benchmark's step count/particle count,
every R row collapses onto the SAME M^2 floor regardless of endpoint --
exactly the "R(t*) isn't scaling as expected" pattern -- and the fix is
to shrink (2) (more particles/generations, more inactive cycles near
criticality) or to isolate (1) directly (this script's own "predicted
clock contribution" numbers) rather than to distrust the theory.

This is a read-only diagnostic: it does not rerun transport or touch
the cache except to load already-checkpointed trajectory.pkl files.

Caveat: k_mid.k_std is the IN-RUN cycle-to-cycle estimator, known to be
biased low by source-iteration correlation especially for under-
converged (high dominance ratio / high k / insufficient n_inactive)
cases. Treat the "local estimator noise" numbers below as a lower bound
on mechanism (2), not an exact value; if mechanism (2) is already
exceeding mechanism (1) even using this likely-understated k_std, the
real gap is probably wider.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_common import load_replicas, safe_interpolate_at_time, M_of_k


def _local_dk_dtau(traj, tau_hat: float) -> float:
    """Finite-difference dk/dtau of this replica's own k_mid trace,
    evaluated at the step nearest tau_hat. Uses the trajectory's own
    (tau, k_mid.k) samples directly -- no interpolation library needed
    for a first-difference slope.
    """
    taus = np.asarray(traj.tau[1:])
    ks = np.asarray([r.k_mid.k for r in traj.records])
    j = int(np.clip(np.searchsorted(taus, tau_hat), 1, len(taus) - 1))
    dtau = taus[j] - taus[j - 1]
    if dtau <= 0:
        return float("nan")
    return (ks[j] - ks[j - 1]) / dtau


def _local_k_std(traj, tau_hat: float) -> float:
    taus = np.asarray(traj.tau[1:])
    j = int(np.clip(np.searchsorted(taus, tau_hat), 0, len(taus) - 1))
    return float(traj.records[j].k_mid.k_std)


def analyze_one_k(cache_root, case, k, n_steps, max_replicas, t_star=None):
    trajs = load_replicas(cache_root, "e1_master_table", case, k, n_steps,
                           range(max_replicas))
    if len(trajs) < 4:
        print(f"[diag] k={k}: only {len(trajs)} replicas, need >=4 -- skipping.")
        return None

    if t_star is None:
        t_star = min(t.t[-1] for t in trajs) * 0.9

    tau_hats, k_hats, dk_dtaus, k_stds = [], [], [], []
    for t in trajs:
        info, reason = safe_interpolate_at_time(t, t_star)
        if info is None:
            print(f"[diag] k={k}: replica skipped ({reason}).")
            continue
        tau_hats.append(info["tau_hat"])
        k_hats.append(info["k_hat"])
        dk_dtaus.append(_local_dk_dtau(t, info["tau_hat"]))
        k_stds.append(_local_k_std(t, info["tau_hat"]))

    if len(tau_hats) < 4:
        print(f"[diag] k={k}: fewer than 4 replicas survived interpolation at "
              f"t*={t_star:.4g} -- skipping.")
        return None

    tau_hats = np.array(tau_hats); k_hats = np.array(k_hats)
    dk_dtaus = np.array(dk_dtaus); k_stds = np.array(k_stds)

    M = M_of_k(k)
    var_tau_hat = float(np.var(tau_hats, ddof=1))
    mean_dk_dtau = float(np.nanmean(dk_dtaus))
    # mechanism (1): clock spread propagated through the local slope
    var_k_from_clock = mean_dk_dtau ** 2 * var_tau_hat
    # mechanism (2): local per-step estimator noise floor (mean of the
    # per-replica k_std^2 -- these are independent across replicas by
    # construction, common.seed_for)
    var_k_from_local = float(np.mean(k_stds ** 2))
    # actual, observed
    var_k_hat_actual = float(np.var(k_hats, ddof=1))

    M_hats = 1.0 / (1.0 - k_hats)
    var_M_actual = float(np.var(M_hats, ddof=1))
    rel_var_R_actual = var_M_actual / float(np.mean(M_hats)) ** 2

    dM_dk = M ** 2  # local sensitivity used throughout the paper
    rel_var_R_from_clock = (dM_dk ** 2 * var_k_from_clock) / M ** 2
    rel_var_R_from_local = (dM_dk ** 2 * var_k_from_local) / M ** 2

    return {
        "k_target": k, "M": M, "n_replicas": len(tau_hats), "t_star": t_star,
        "var_tau_hat": var_tau_hat, "mean_dk_dtau": mean_dk_dtau,
        "var_k_from_clock": var_k_from_clock, "var_k_from_local": var_k_from_local,
        "var_k_hat_actual": var_k_hat_actual,
        "rel_var_R_from_clock": rel_var_R_from_clock,
        "rel_var_R_from_local": rel_var_R_from_local,
        "rel_var_R_actual": rel_var_R_actual,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", type=lambda s: Path(s).resolve(), required=True)
    p.add_argument("--case", default="heu_sphere_nu")
    p.add_argument("--k-targets", type=float, nargs="+", required=True)
    p.add_argument("--n-steps", type=int, required=True)
    p.add_argument("--max-replicas", type=int, default=12)
    p.add_argument("--t-star", type=float, default=None,
                    help="Shared calendar target across all k_targets. If "
                         "omitted, each k_target uses its own "
                         "0.9*min(replica t_end) -- fine for this "
                         "diagnostic (unlike the matched-burnup figure "
                         "panel, nothing here is a headline number that "
                         "needs a cross-k-target-consistent target).")
    p.add_argument("--out-txt", type=Path, default=None)
    args = p.parse_args()

    lines = []
    header = (f"{'k_target':>9} {'M':>8} {'R':>4} {'t*':>12} | "
              f"{'Var(tau_hat)':>13} {'dk/dtau':>10} | "
              f"{'relVar(R) clock':>16} {'relVar(R) local':>16} {'relVar(R) actual':>17}")
    lines.append(header)
    lines.append("-" * len(header))
    for k in sorted(args.k_targets):
        res = analyze_one_k(args.cache_root, args.case, k, args.n_steps,
                             args.max_replicas, args.t_star)
        if res is None:
            continue
        lines.append(
            f"{res['k_target']:9.4f} {res['M']:8.2f} {res['n_replicas']:4d} "
            f"{res['t_star']:12.5g} | {res['var_tau_hat']:13.5g} "
            f"{res['mean_dk_dtau']:10.3g} | {res['rel_var_R_from_clock']:16.5g} "
            f"{res['rel_var_R_from_local']:16.5g} {res['rel_var_R_actual']:17.5g}")
    report = "\n".join(lines)
    print("\n" + report)
    print("\n[diag] Read this as: if 'relVar(R) local' tracks 'relVar(R) actual' "
          "across the whole k sweep while 'relVar(R) clock' stays far below both, "
          "the per-step k-estimator noise floor (mechanism 2) is what the fitted "
          "R(t*) exponent is actually measuring, not the accumulated exposure-"
          "clock displacement (mechanism 1) the M^4/M^6 theory rows are about. "
          "That points at increasing particles/generations or n_inactive (the "
          "k_std reduction fixes both this script's own likely-understated "
          "k_std and the underlying floor), not at the theory.")

    if args.out_txt is not None:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(report + "\n")
        print(f"[diag] wrote {args.out_txt}")


if __name__ == "__main__":
    main()
