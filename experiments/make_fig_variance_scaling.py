#!/usr/bin/env python3
"""Camera-ready fig_variance_scaling.pdf for the depletion paper --
replaces both PLACEHOLDER panels with real data:

  (a) exposure-indexed quantities: N(tau*), t(tau*), R(tau*)=S0*M
      -- all read directly from E1's own summary CSV, no interpolation.

  (b) calendar-indexed quantities: N(t*) and R(t*)=S0*M, each at BOTH
      matched burnup (E1's own trajectories, interpolated at each
      k_target's own min replica t_end -- see scaling_table_figure.py's
      module docstring for why this needs a per-k_target target rather
      than E1's --t-star mechanism) and fixed calendar duration (E9's
      native delivery, no interpolation needed at all).

Every quantity uses a single named nuclide (--nuclide), not the
full-inventory sum -- see plot_results.py's own diagnostic for why the
sum is degenerate for at least heu_sphere_nu. R(tau*)/R(t*) use S0*M
directly (M_end column), not an extracted fission-rate diagnostic -- see
the several turns of this project's history on why _total_fission_rate
is not trusted.

Renders with real LaTeX (text.usetex) if a working `latex` binary is
found on PATH, and falls back to matplotlib's built-in mathtext
otherwise -- checked at runtime rather than assumed, since this script
may run in different environments (this sandbox vs. Sawtooth) with
different toolchains, and a hard usetex=True would crash outright
wherever LaTeX isn't installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_common import (load_csv, group_by, M_of_k, fit_loglog,
                              nuclide_index, load_replicas, sigma_pair,
                              safe_interpolate_at_time)


def _configure_latex():
    """Never trust shutil.which alone: a `latex` binary can be on PATH
    with a broken or incomplete installation (missing packages) that
    still crashes on the first real render. Render a real throwaway
    test string and catch the failure instead.
    """
    have_latex = False
    if shutil.which("latex") is not None:
        try:
            plt.rcParams.update({
                "text.usetex": True, "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            })
            fig_test = plt.figure()
            fig_test.text(0.5, 0.5, r"$M^{1.0}$, $R^2=1.0$")
            fig_test.canvas.draw()
            plt.close(fig_test)
            have_latex = True
        except Exception as exc:
            plt.close("all")
            print(f"[fig] `latex` is on PATH but a test render failed "
                  f"({type(exc).__name__}) -- falling back to mathtext. "
                  f"({str(exc).splitlines()[0] if str(exc) else ''})")
    if have_latex:
        print("[fig] latex found and verified working -- rendering with real LaTeX.")
    else:
        plt.rcParams.update({"text.usetex": False,
                              "mathtext.fontset": "cm", "font.family": "serif"})
        print("[fig] rendering with matplotlib's built-in mathtext "
              "(still renders $...$ math, just not via a real TeX engine).")
    return have_latex


def _series_from_csv(rows, case, col, nuclide_ok=True):
    """(Ms, variances, n_replicas) of literal Var(X) vs M, grouped by
    k_target, for one CSV column. n_replicas is returned alongside so
    the caller can plot the sqrt(2/(R-1)) relative uncertainty on each
    variance point (the same convention this project's own methodology
    uses elsewhere -- e.g. the depletion paper's "every replica variance
    carries a sqrt(2/(R-1)) relative uncertainty"). Returns empty arrays
    if fewer than 2 k_target points have >=3 replicas and nonzero
    variance.
    """
    case_rows = [r for r in rows if r["case"] == case and r.get(col) is not None]
    groups = group_by(case_rows, ("k_target",))
    Ms, sigmas, ns = [], [], []
    n_exactly_zero = 0
    for (k_target,), grp in sorted(groups.items()):
        vals = [g[col] for g in grp]
        rs = sigma_pair(vals)[1] ** 2  # literal Var(X), not sigma
        if not np.isfinite(rs):
            continue
        if rs == 0.0:
            n_exactly_zero += 1
            continue
        Ms.append(M_of_k(float(k_target)))
        sigmas.append(rs)
        ns.append(len(vals))
    if n_exactly_zero and not Ms:
        print(f"[fig] {col}: exactly zero relative sigma at all "
              f"{n_exactly_zero} k_target points -- try a different "
              "--nuclide." if nuclide_ok else "")
    return np.array(Ms), np.array(sigmas), np.array(ns)


def _series_matched_burnup(cache_root, case, k_targets, n_steps, max_replicas,
                            idx, quantity):
    """N or R=S0*M at t=t(tau*), matched burnup, as literal Var(X).

    t_star_k -- the calendar-time target every measurement replica is
    interpolated at -- is derived from a SEPARATE reference half of the
    replica pool, not from the same replicas whose spread we then
    measure. Using min(t_end) over the SAME ensemble being measured
    would be self-referential: the replica that sets t_star_k is
    interpolated almost exactly at its own endpoint (near-zero
    extrapolation distance) while every other replica is evaluated
    strictly before its own endpoint, which is an asymmetry with no
    clean characterization. Splitting the pool removes that circularity
    at the cost of roughly half the effective replica count for the
    variance estimate (tracked via the returned n_replicas, and this
    is exactly the sqrt(2/(R-1)) factor that should widen these error
    bars relative to the un-split matched-burnup rows in this figure).
    """
    Ms, sigmas, ns = [], [], []
    for k in sorted(k_targets):
        trajs = load_replicas(cache_root, "e1_master_table", case, k,
                               n_steps, range(max_replicas))
        if len(trajs) < 6:
            print(f"[fig] matched-burnup {quantity} k={k}: only "
                  f"{len(trajs)} e1_master_table replicas, need >=6 to "
                  "split into a reference half and a measurement half.")
            continue
        half = len(trajs) // 2
        ref_trajs, meas_trajs = trajs[:half], trajs[half:]
        t_star_k = min(t.t[-1] for t in ref_trajs) * 0.999
        vals = []
        for t in meas_trajs:
            info, reason = safe_interpolate_at_time(t, t_star_k)
            if info is None:
                print(f"[fig] matched-burnup {quantity} k={k}: skipping "
                      f"one measurement replica ({reason}).")
                continue
            if quantity == "N":
                vals.append(info["N"][idx])
            else:  # R = S0*M_hat
                S0 = t.metadata.get("source_strength", 1.0)
                M_hat = 1.0 / (1.0 - info["k_hat"])
                vals.append(S0 * M_hat)
        if len(vals) < 3:
            print(f"[fig] matched-burnup {quantity} k={k}: only "
                  f"{len(vals)} measurement replicas survived the split "
                  "and interpolation, need >=3.")
            continue
        rs = sigma_pair(vals)[1] ** 2  # literal Var(X), not sigma
        if np.isfinite(rs) and rs > 0:
            Ms.append(M_of_k(k)); sigmas.append(rs); ns.append(len(vals))
    return np.array(Ms), np.array(sigmas), np.array(ns)


def _add_line(ax, Ms, sigmas, ns, label_tex, expected_exp, color):
    """ns is the replica count behind each point; error bars show the
    sqrt(2/(R-1)) relative uncertainty on a sample variance estimate --
    this project's own established convention (see e.g. the depletion
    paper's methodology section) for how much a replica-variance point
    itself should be trusted. NOTE: fit_loglog below is an UNWEIGHTED
    log-log least-squares fit -- it does not use these uncertainties at
    all, so a point with a wide error bar pulls the fitted slope exactly
    as hard as a tightly-resolved one. That's a real limitation of the
    printed/legend slope, not just of the plot: treat R^2 and the fitted
    exponent as more trustworthy where the error bars are visibly small
    and treat them cautiously where the bars are large.
    """
    if len(Ms) < 2:
        print(f"[fig] {label_tex}: fewer than 2 usable points, not plotted.")
        return
    order = np.argsort(Ms)
    Ms, sigmas, ns = Ms[order], sigmas[order], ns[order]
    slope, _, r2 = fit_loglog(Ms, sigmas)
    rel_err = np.sqrt(2.0 / np.clip(ns - 1, 1, None))
    print(f"[fig] {label_tex}: fitted slope={slope:.2f} "
          f"(predicted {expected_exp})  R^2={r2:.3f}  n={len(Ms)}  "
          f"replicas/point={list(ns)}  rel.unc./point="
          f"{np.round(rel_err, 2).tolist()} (unweighted fit)")
    leg = (rf"{label_tex}: data $\propto M^{{{slope:.2f}}}$ "
           rf"($R^2={r2:.2f}$), theory $\propto M^{{{expected_exp}}}$")
    ax.errorbar(Ms, sigmas, yerr=sigmas * rel_err, fmt="o-", color=color,
                markersize=5, capsize=3, elinewidth=1, label=leg)
    # theoretical reference line, anchored through this line's own
    # geometric-mean point, with a direct in-plot label (not just the
    # legend) so it reads unambiguously as the theoretical prediction
    # rather than a second data series
    xlim_lo, xlim_hi = Ms.min() * 0.8, Ms.max() * 1.3
    xs = np.geomspace(xlim_lo, xlim_hi, 20)
    x_mid = np.sqrt(Ms.min() * Ms.max())
    y_mid = np.exp(np.mean(np.log(sigmas)))
    ax.loglog(xs, y_mid * (xs / x_mid) ** expected_exp, "--", color=color,
              alpha=0.4, linewidth=1.2)
    ax.annotate(rf"theory $\propto M^{{{expected_exp}}}$",
                xy=(xs[-1], y_mid * (xs[-1] / x_mid) ** expected_exp),
                xytext=(4, 0), textcoords="offset points",
                color=color, alpha=0.75, fontsize=7, va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e1-summary-csv", type=Path, required=True,
                         help="E1's summary CSV, already per-nuclide "
                              "extracted (extract_nuclide.py).")
    parser.add_argument("--e9-summary-csv", type=Path, required=True,
                         help="E9's summary CSV, already per-nuclide "
                              "extracted, --experiment e9_fixed_duration.")
    parser.add_argument("--cache-root", type=lambda s: Path(s).resolve(), required=True,
                         help="For the matched-burnup panel-b rows, which "
                              "read cached E1 trajectories directly.")
    parser.add_argument("--case", default="heu_sphere_nu")
    parser.add_argument("--k-targets", type=float, nargs="+", required=True)
    parser.add_argument("--n-steps", type=int, required=True,
                         help="E1's n_steps (for the matched-burnup rows' "
                              "trajectory lookup).")
    parser.add_argument("--max-replicas", type=int, default=12)
    parser.add_argument("--nuclide", required=True)
    parser.add_argument("--chain-file", default=None)
    parser.add_argument("--out", type=Path, required=True,
                         help="Output path stem; writes <out>.pdf and <out>.png.")
    args = parser.parse_args()

    _configure_latex()

    chain_file = args.chain_file
    if chain_file is None:
        import openmc
        chain_file = openmc.config.get("chain_file")
        if chain_file is None:
            raise SystemExit("No --chain-file and openmc.config['chain_file'] unset.")
    idx = nuclide_index(chain_file, args.nuclide)
    nuc_tex = args.nuclide  # e.g. "Xe135" -- fine as-is in \text{}

    e1_rows = load_csv(args.e1_summary_csv)
    e9_rows = load_csv(args.e9_summary_csv)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ---- panel (a): exposure-indexed ----
    # Exponents below are Var(X), not Var(X)/X^2 -- table col 5, not col 6.
    Ms, s, ns = _series_from_csv(e1_rows, args.case, f"N_end_{args.nuclide}")
    _add_line(ax_a, Ms, s, ns, rf"$N(\tau_*)$", 0, "tab:blue")

    Ms, s, ns = _series_from_csv(e1_rows, args.case, "t_end")
    _add_line(ax_a, Ms, s, ns, rf"$t(\tau_*)$", 0, "tab:orange")

    Ms, s, ns = _series_from_csv(e1_rows, args.case, "M_end")
    _add_line(ax_a, Ms, s, ns, rf"$R(\tau_*)=S_0M$", 4, "tab:green")

    ax_a.set_xlabel(r"$M = 1/(1-k)$")
    ax_a.set_ylabel(r"replica $\operatorname{Var}(X)$")
    ax_a.set_title(r"(a) exposure-indexed quantities")
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.grid(True, which="both", alpha=0.25)

    # ---- panel (b): calendar-indexed ----
    Ms, s, ns = _series_matched_burnup(args.cache_root, args.case, args.k_targets,
                                        args.n_steps, args.max_replicas, idx, "N")
    _add_line(ax_b, Ms, s, ns, rf"$N(t_*)$, matched burnup", 2, "tab:blue")

    Ms, s, ns = _series_matched_burnup(args.cache_root, args.case, args.k_targets,
                                        args.n_steps, args.max_replicas, idx, "R")
    _add_line(ax_b, Ms, s, ns, rf"$R(t_*)$, matched burnup", 6, "tab:cyan")

    Ms, s, ns = _series_from_csv(e9_rows, args.case, f"N_end_{args.nuclide}")
    _add_line(ax_b, Ms, s, ns, rf"$N(t_*)$, fixed duration", 4, "tab:red")

    Ms, s, ns = _series_from_csv(e9_rows, args.case, "M_end")
    _add_line(ax_b, Ms, s, ns, rf"$R(t_*)$, fixed duration", 8, "tab:purple")

    ax_b.set_xlabel(r"$M = 1/(1-k)$")
    ax_b.set_ylabel(r"replica $\operatorname{Var}(X)$")
    ax_b.set_title(r"(b) calendar-indexed quantities")
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"[fig] wrote {args.out.with_suffix('.pdf')} and "
          f"{args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
