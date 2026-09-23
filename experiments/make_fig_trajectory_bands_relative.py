#!/usr/bin/env python3
"""Trajectory-band figure (paper Fig. 2 layout), plus a --relative mode
that plots each replica as its FRACTIONAL deviation from that k_target's
own ensemble mean, N_i(x)/mean_k(x) - 1, instead of raw N.

Why this exists: make_fig_trajectory_bands.py plots raw N on a linear
axis, and for a saturating fission product (Xe-135, I-131) the
across-k_target range of the MEAN trajectory (set by the physical flux
level at each k, which varies by 1-2 orders of magnitude over a
k=0.5->0.995 sweep) dwarfs the within-k_target replica spread on the
same linear scale. Low-k bands get compressed to a hairline near y=0,
and any k-dependent widening of the replica spread (the thing the
M-scaling theory actually predicts) is invisible by construction,
regardless of whether it is present in the data. This is a plotting
artifact, not evidence the noise isn't there -- see the companion
written diagnosis for why U-235 looks "clean" in the original figure:
its total depletion range across 200 days is only ~2%, so it happens to
sit in a y-range narrow enough that replica spread stays visible on a
linear axis without any special handling, not because it is fitting the
predicted M-scaling from a place of privilege.

--relative fixes this by removing the across-k mean-level differences
entirely, leaving only the quantity the theory is actually a statement
about: how much a replica departs from its own k_target's mean, as a
fraction. All three panels take the SAME flag.

Usage: identical to make_fig_trajectory_bands.py, plus --relative.
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

try:
    from make_fig_variance_scaling import _configure_latex
except ImportError:
    def _configure_latex():
        plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})
        return False


def _to_relative(y_stack: np.ndarray) -> np.ndarray:
    """y_stack: (n_replicas, n_points). Returns fractional deviation from
    the cross-replica mean at each point, y_i/mean(y) - 1. NaN columns
    (no valid replicas at that point) stay NaN. Guards mean==0 (shouldn't
    happen for a physical concentration, but a placeholder/zeroed nuclide
    column would otherwise divide by zero silently).
    """
    mean = np.nanmean(y_stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = y_stack / mean - 1.0
    rel[:, mean == 0] = np.nan
    return rel


def _plot_spread(ax, x_by_replica, y_by_replica, color, label, plot_mean=True):
    for x, y in zip(x_by_replica, y_by_replica):
        ax.plot(x, y, color=color, alpha=0.35, linewidth=0.8)
    if plot_mean and x_by_replica and \
            all(len(x) == len(x_by_replica[0]) for x in x_by_replica) and \
            all(np.allclose(x, x_by_replica[0]) for x in x_by_replica[1:]):
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
    parser.add_argument("--n-grid-c", type=int, default=30)
    parser.add_argument("--nuclide", required=True)
    parser.add_argument("--chain-file", default=None)
    parser.add_argument("--relative", dest="relative", action="store_true", default=True,
                         help="Plot fractional deviation from each k_target's "
                              "own mean instead of raw N (default: on).")
    parser.add_argument("--no-relative", dest="relative", action="store_false")
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
            print(f"[fig] k={k}: only {len(trajs)} replicas, need >=3 -- skipping.")
            continue

        # ---- panel (a): N(tau) or relative deviation, shared exposure grid ----
        tau_grid = np.asarray(trajs[0].tau[1:])
        N_by_step = np.array([[rec.vec[0][idx] for rec in t.records] for t in trajs])
        Y_a = _to_relative(N_by_step) if args.relative else N_by_step
        _plot_spread(ax_a, [tau_grid] * len(trajs), list(Y_a), color, rf"$k={k}$")

        # ---- panel (b): against each replica's own calendar grid ----
        # relative mode needs the SAME per-k mean as panel (a), evaluated on
        # each replica's own t grid; use panel (a)'s tau-indexed mean since
        # tau is the shared grid, then plot each replica's (tau-indexed)
        # deviation against its own t.
        mean_a = np.nanmean(N_by_step, axis=0)
        for i, t in enumerate(trajs):
            t_grid = np.asarray(t.t[1:])
            N_seq = np.array([rec.vec[0][idx] for rec in t.records])
            y = (N_seq / mean_a - 1.0) if args.relative else N_seq
            ax_b.plot(t_grid, y, color=color, alpha=0.35, linewidth=0.8)
        ax_b.plot([], [], color=color, linewidth=2.0, label=rf"$k={k}$")

        # ---- panel (c): interpolated to a common calendar grid ----
        t_lo = min(t.t[1] for t in trajs)
        t_hi = min(t.t[-1] for t in trajs) * 0.999
        if t_hi <= t_lo:
            print(f"[fig] k={k}: panel (c) skipped (no valid common window).")
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
            print(f"[fig] k={k}: panel (c) skipped (no grid point with >=3 valid replicas).")
            continue
        Y_c = _to_relative(N_interp) if args.relative else N_interp
        for i in range(Y_c.shape[0]):
            row_ok = ok & ~np.isnan(Y_c[i])
            if np.sum(row_ok) < 2:
                continue
            ax_c.plot(t_grid_c[row_ok], Y_c[i][row_ok], color=color, alpha=0.35, linewidth=0.8)
        mean_c = np.nanmean(Y_c, axis=0)
        ax_c.plot(t_grid_c[ok], mean_c[ok], color=color, linewidth=2.0, label=rf"$k={k}$")

    ylab = rf"$N_{{{args.nuclide}}}/\overline{{N}}_{{{args.nuclide}}}(k) - 1$" if args.relative \
        else rf"$N_{{{args.nuclide}}}$"

    ax_a.set_xlabel(r"$\tau$"); ax_a.set_ylabel(ylab)
    ax_a.set_title(r"(a) against exposure $\tau$ (relative to own-$k$ mean)"
                    if args.relative else r"(a) against exposure $\tau$")
    ax_a.legend(fontsize=7, loc="best")
    ax_a.axhline(0, color="k", linewidth=0.5, alpha=0.4) if args.relative else None

    ax_b.set_xlabel(r"$t$"); ax_b.set_ylabel(ylab)
    ax_b.set_title(r"(b) against each trajectory's own $t(\tau)$")
    ax_b.set_xscale("log")
    ax_b.legend(fontsize=7, loc="best")

    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"$t$ (common grid, per $k_\text{target}$)"); ax_c.set_ylabel(ylab)
    ax_c.set_title(r"(c) interpolated to a common time grid")
    ax_c.legend(fontsize=7, loc="best")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"[fig] wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
