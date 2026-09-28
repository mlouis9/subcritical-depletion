#!/usr/bin/env python3
"""The paper's three-panel "error accumulation" figure (its Fig. 2):
Xe-135 (or any single tracked nuclide) trajectories from E1's cached
replicas, shown three ways to make the longitudinal-displacement
mechanism visible directly, rather than only summarized as a fitted
exponent the way make_fig_variance_scaling.py does.

All three panels use the SAME depiction of replica-to-replica spread --
every individual replica drawn as its own thin, alpha-blended line, plus
one bold mean line per k_target on top for orientation. No panel uses a
summary band or per-point markers; the point of showing all three this
way is that the SAME underlying spread is directly comparable panel to
panel, not re-encoded differently each time.

  (a) N(tau) for every replica of every k_target, against the SHARED
      exposure grid (uniform and deterministic across the whole sweep
      by construction -- E1's own design). Per the paper: trajectories
      should coincide with an M-independent spread.
  (b) The SAME per-replica values, against each replica's OWN
      calendar-time grid (t, not tau). Per the paper: k families should
      separate horizontally (different k reach different t ranges for
      the same tau), and within a k family a horizontal spread should
      appear (replica-to-replica variation in the accumulated clock).
  (c) Each replica's own N(t) interpolated onto a COMMON calendar-time
      grid per k_target, spanning the SAME range panel (b) actually
      shows (from the earliest step any replica of that k_target
      records, up to that k_target's own min(t_end) across replicas --
      see the grid-construction comment below for why both bounds are
      safe interpolation, not extrapolation). Per the paper: the
      horizontal spread of (b) becomes the vertical, k-dependent spread
      of a date-indexed inventory here.

Needs only E1's cached trajectories (--cache-root), not E9 -- all three
panels reinterpret the SAME exposure-domain run.

Usage::

    python make_fig_trajectory_bands.py \\
      --cache-root cache --case heu_sphere_nu \\
      --k-targets 0.50 0.80 0.95 0.99 0.995 \\
      --n-steps 200 --nuclide Xe135 \\
      --out fig_trajectory_bands

A subset of k_targets, not the full 9-point sweep, is recommended for
this figure specifically: unlike the scaling-exponent figures, every
replica's full trajectory is drawn here, and both readability and
plotting time scale with (k_targets x replicas x n_steps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_common import nuclide_index, load_replicas, safe_interpolate_at_time
from make_fig_variance_scaling import _configure_latex


def _plot_spread(ax, x_by_replica, y_by_replica, color, label):
    """Common depiction used by all three panels: every replica as its
    own thin line, plus one bold mean on top. x_by_replica/y_by_replica
    are lists of per-replica (possibly different-length) arrays; the
    mean is computed only where every replica actually has a value (so
    ragged-length inputs -- e.g. panel (c)'s partially-interpolated
    replicas -- don't bias the mean toward whichever replicas happen to
    cover a given point).
    """
    for i, (x, y) in enumerate(zip(x_by_replica, y_by_replica)):
        ax.plot(x, y, color=color, alpha=0.35, linewidth=0.8)
    # Mean only over grid points where every input array shares the same
    # x -- true for panels (a)/(c) (shared grid by construction), and for
    # panel (b) no mean is drawn at all (see caller).
    if x_by_replica and all(len(x) == len(x_by_replica[0]) for x in x_by_replica) \
            and all(np.allclose(x, x_by_replica[0]) for x in x_by_replica[1:]):
        y_stack = np.array(y_by_replica)
        ax.plot(x_by_replica[0], np.nanmean(y_stack, axis=0), color=color,
                linewidth=2.0, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--case", default="heu_sphere_nu")
    parser.add_argument("--k-targets", type=float, nargs="+", required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--max-replicas", type=int, default=12)
    parser.add_argument("--n-grid-c", type=int, default=30,
                         help="Number of points on panel (c)'s common "
                              "calendar-time grid, per k_target.")
    parser.add_argument("--nuclide", required=True)
    parser.add_argument("--chain-file", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _configure_latex()

    chain_file = args.chain_file
    if chain_file is None:
        import openmc
        chain_file = openmc.config.get("chain_file")
        if chain_file is None:
            raise SystemExit("No --chain-file and openmc.config['chain_file'] unset.")
    idx = nuclide_index(chain_file, args.nuclide)

    k_targets = sorted(args.k_targets)
    colors = cm.viridis(np.linspace(0.05, 0.9, len(k_targets)))

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5))

    for k, color in zip(k_targets, colors):
        trajs = load_replicas(args.cache_root, "e1_master_table", args.case, k,
                               args.n_steps, range(args.max_replicas))
        if len(trajs) < 3:
            print(f"[fig] k={k}: only {len(trajs)} e1_master_table replicas, "
                  "need >=3 -- skipping this k_target entirely.")
            continue

        # ---- panel (a): N(tau), shared deterministic exposure grid ----
        tau_grid = np.asarray(trajs[0].tau[1:])  # same for every replica by
                                                   # construction (uniform,
                                                   # k_target-independent
                                                   # exposure grid)
        N_by_step = [np.array([rec.vec[0][idx] for rec in t.records])
                     for t in trajs]
        _plot_spread(ax_a, [tau_grid] * len(trajs), N_by_step, color,
                    rf"$k={k}$")

        # ---- panel (b): N(t), each replica's OWN calendar grid ----
        # No mean line here deliberately: different replicas' own t grids
        # don't align at a common x, so there is no single x to plot a
        # mean against without first interpolating -- which is exactly
        # what panel (c) does. A legend entry is still added via a
        # zero-alpha proxy so all three panels' legends list the same
        # k_targets consistently.
        for i, t in enumerate(trajs):
            t_grid = np.asarray(t.t[1:])
            N_seq = np.array([rec.vec[0][idx] for rec in t.records])
            ax_b.plot(t_grid, N_seq, color=color, alpha=0.35, linewidth=0.8)
        ax_b.plot([], [], color=color, linewidth=2.0, label=rf"$k={k}$")

        # ---- panel (c): N(t) interpolated onto a common calendar grid ----
        # Grid spans the SAME range panel (b) actually shows for this
        # k_target: lower bound = earliest step any replica records
        # (min(t.t[1]) -- capturing the early buildup/peak panel (b)
        # shows, which a grid starting later would miss entirely), upper
        # bound = min(t.t[-1]) across the full ensemble (not a reference-
        # half split the way _series_matched_burnup uses for the scaling
        # figure -- that split guards a variance ESTIMATE fed into a
        # fitted slope; here every individual replica is drawn and
        # visible on its own, so whichever replica is shortest simply
        # terminates at the grid's right edge in plain view). Both bounds
        # are genuine interpolation, not extrapolation: every trajectory's
        # own t[0] = 0 by construction (interpolate_at_time only checks
        # the UPPER bound, t_star > t[-1]; PchipInterpolator is built from
        # the full t array including its t=0 point), so any t_star in
        # [0, t[-1]] is within that replica's valid domain regardless of
        # where the grid's own lower bound happens to sit.
        t_lo = min(t.t[1] for t in trajs)
        t_hi = min(t.t[-1] for t in trajs) * 0.999
        if t_hi <= t_lo:
            print(f"[fig] k={k}: shortest replica's t_end doesn't reach "
                  "past the earliest recorded step across the ensemble -- "
                  "panel (c) skipped for this k_target.")
            continue
        t_grid_c = np.geomspace(t_lo, t_hi, args.n_grid_c)
        N_interp = np.full((len(trajs), args.n_grid_c), np.nan)
        for i, t in enumerate(trajs):
            for j, t_star in enumerate(t_grid_c):
                info, reason = safe_interpolate_at_time(t, t_star)
                if info is not None:
                    N_interp[i, j] = info["N"][idx]
        n_valid = np.sum(~np.isnan(N_interp), axis=0)
        ok = n_valid >= 3
        if not np.any(ok):
            print(f"[fig] k={k}: no panel-(c) grid point had >=3 valid "
                  "interpolated replicas -- panel (c) skipped for this "
                  "k_target.")
            continue
        x_by_replica, y_by_replica = [], []
        for i in range(N_interp.shape[0]):
            row_ok = ok & ~np.isnan(N_interp[i])
            if np.sum(row_ok) < 2:
                continue
            x_by_replica.append(t_grid_c[row_ok])
            y_by_replica.append(N_interp[i][row_ok])
        for x, y in zip(x_by_replica, y_by_replica):
            ax_c.plot(x, y, color=color, alpha=0.35, linewidth=0.8)
        # Mean over the grid points enough replicas actually reached
        mean_c = np.nanmean(N_interp, axis=0)
        ax_c.plot(t_grid_c[ok], mean_c[ok], color=color, linewidth=2.0,
                  label=rf"$k={k}$")

    ax_a.set_xlabel(r"$\tau$")
    ax_a.set_ylabel(rf"$N_{{{args.nuclide}}}$")
    ax_a.set_title(r"(a) against exposure $\tau$")
    ax_a.legend(fontsize=7, loc="best")

    ax_b.set_xlabel(r"$t$")
    ax_b.set_ylabel(rf"$N_{{{args.nuclide}}}$")
    ax_b.set_title(r"(b) against each trajectory's own $t(\tau)$")
    ax_b.set_xscale("log")
    ax_b.legend(fontsize=7, loc="best")

    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"$t$ (common grid, per $k_\text{target}$)")
    ax_c.set_ylabel(rf"$N_{{{args.nuclide}}}$")
    ax_c.set_title(r"(c) interpolated to a common date")
    ax_c.legend(fontsize=7, loc="best")

    fig.suptitle(rf"Accumulation of the longitudinal error, {args.case}, "
                 rf"$N_{{{args.nuclide}}}$")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"[fig] wrote {args.out.with_suffix('.pdf')} and "
          f"{args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
