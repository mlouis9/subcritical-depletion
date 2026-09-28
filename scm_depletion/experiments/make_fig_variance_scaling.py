#!/usr/bin/env python3
"""Camera-ready fig_variance_scaling.pdf for the depletion paper --
replaces both PLACEHOLDER panels with real data, plotted as RELATIVE
variance Var(X)/X^2:

  (a) exposure-indexed quantities: N(tau*), t(tau*), R(tau*)=S0*M
      -- all read directly from E1's own summary CSV, no interpolation.

  (b) calendar-indexed quantities: N(t*) and R(t*)=S0*M, each at BOTH
      matched burnup (E1's own trajectories, interpolated at each
      k_target's own min replica t_end -- see scaling_table_figure.py's
      module docstring for why this needs a per-k_target target rather
      than E1's --t-star mechanism) and fixed calendar duration (E9's
      native delivery, no interpolation needed at all).

NSCM (Methods A/C/C') is always plotted. Method B (branching fixed
source) is overlaid when --e1b-summary-csv/--e9b-summary-csv are given,
on the SAME two panels rather than a separate figure: color still
groups by physical quantity (e.g. N(tau*) is tab:blue for both methods),
but Method B uses a dashed line, square marker, dotted theory line, and
an explicit "Method B" in its legend text and theory annotation --
delineated by encoding, not by a second figure, so the same quantity is
directly comparable across methods without cross-referencing two plots.
See _add_line's docstring for the full scheme, and the per-row comments
at each Method B call site for which theory-table column applies (most
rows use "Physical Chain" -- what a fixed-particle-count experiment
actually measures -- two rows fall back to "Branching Fixed-Source"
since "Physical Chain" is blank there, explicitly labeled as a
different normalization basis when that happens).

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
    """(Ms, variances, n_replicas) of RELATIVE variance Var(X)/X^2 vs M,
    grouped by k_target, for one CSV column. n_replicas is returned
    alongside so the caller can plot the sqrt(2/(R-1)) relative
    uncertainty on each variance point (the same convention this
    project's own methodology uses elsewhere -- e.g. the depletion
    paper's "every replica variance carries a sqrt(2/(R-1)) relative
    uncertainty"). Returns empty arrays if fewer than 2 k_target points
    have >=3 replicas and nonzero variance.
    """
    case_rows = [r for r in rows if r["case"] == case and r.get(col) is not None]
    groups = group_by(case_rows, ("k_target",))
    Ms, sigmas, ns = [], [], []
    n_exactly_zero = 0
    for (k_target,), grp in sorted(groups.items()):
        vals = [g[col] for g in grp]
        rs = sigma_pair(vals)[0] ** 2  # relative variance Var(X)/X^2
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
                            idx, quantity, experiment="e1_master_table"):
    """N or R=S0*M at t=t(tau*), matched burnup, as RELATIVE variance
    Var(X)/X^2.

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
        trajs = load_replicas(cache_root, experiment, case, k,
                               n_steps, range(max_replicas))
        if len(trajs) < 6:
            print(f"[fig] matched-burnup {quantity} k={k}: only "
                  f"{len(trajs)} {experiment} replicas, need >=6 to "
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
        rs = sigma_pair(vals)[0] ** 2  # relative variance Var(X)/X^2
        if np.isfinite(rs) and rs > 0:
            Ms.append(M_of_k(k)); sigmas.append(rs); ns.append(len(vals))
    return np.array(Ms), np.array(sigmas), np.array(ns)


def _add_line(ax, Ms, sigmas, ns, label_tex, expected_exp, color,
              marker="o", linestyle="-", theory_linestyle="--",
              method_label=""):
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

    expected_exp may be None: no theory reference line is fabricated
    when the source table doesn't give one for this row/method.

    marker/linestyle distinguish NSCM (default: circle, solid) from
    Method B (square, dashed) when both are plotted with the SAME color
    for the same physical quantity -- color groups by quantity, marker/
    linestyle disambiguates by method, and the legend text names the
    method explicitly too (method_label), so the distinction survives
    even in grayscale or a small legend swatch. theory_linestyle
    similarly separates the two methods' theory lines when both are
    dashed the same color: NSCM stays "--", Method B uses ":" (see
    call sites).
    """
    if len(Ms) < 2:
        print(f"[fig] {label_tex}: fewer than 2 usable points, not plotted.")
        return
    order = np.argsort(Ms)
    Ms, sigmas, ns = Ms[order], sigmas[order], ns[order]
    slope, _, r2 = fit_loglog(Ms, sigmas)
    rel_err = np.sqrt(2.0 / np.clip(ns - 1, 1, None))
    theory_str = "none" if expected_exp is None else str(expected_exp)
    print(f"[fig] {label_tex}{' ' + method_label if method_label else ''}: "
          f"fitted slope={slope:.2f} (predicted {theory_str})  R^2={r2:.3f}  "
          f"n={len(Ms)}  replicas/point={list(ns)}  rel.unc./point="
          f"{np.round(rel_err, 2).tolist()} (unweighted fit)")
    full_label = f"{label_tex}, {method_label}" if method_label else label_tex
    if expected_exp is None:
        leg = rf"{full_label}: data $\propto M^{{{slope:.2f}}}$ ($R^2={r2:.2f}$)"
    else:
        leg = (rf"{full_label}: data $\propto M^{{{slope:.2f}}}$ "
               rf"($R^2={r2:.2f}$), theory $\propto M^{{{expected_exp}}}$")
    ax.errorbar(Ms, sigmas, yerr=sigmas * rel_err, fmt=f"{marker}{linestyle}",
                color=color, markersize=5, capsize=3, elinewidth=1, label=leg)
    if expected_exp is None:
        return
    # theoretical reference line, anchored through this line's own
    # geometric-mean point, with a direct in-plot label (not just the
    # legend) so it reads unambiguously as the theoretical prediction
    # rather than a second data series
    xlim_lo, xlim_hi = Ms.min() * 0.8, Ms.max() * 1.3
    xs = np.geomspace(xlim_lo, xlim_hi, 20)
    x_mid = np.sqrt(Ms.min() * Ms.max())
    y_mid = np.exp(np.mean(np.log(sigmas)))
    ax.loglog(xs, y_mid * (xs / x_mid) ** expected_exp, theory_linestyle,
              color=color, alpha=0.4, linewidth=1.2)
    # "NSCM" is dropped from the in-plot theory annotation (it stays in
    # the legend text via full_label above, where it's still useful for
    # telling series apart) -- with only one method plotted, or in the
    # overlay figure's shared axes, an unqualified "theory" line is
    # unambiguous; "Method B theory" is kept since that distinction
    # still matters whenever both methods share a panel.
    theory_text = (rf"{method_label} theory $\propto M^{{{expected_exp}}}$"
                   if method_label and method_label != "NSCM"
                   else rf"theory $\propto M^{{{expected_exp}}}$")
    ax.annotate(theory_text,
                xy=(xs[-1], y_mid * (xs[-1] / x_mid) ** expected_exp),
                xytext=(4, 0), textcoords="offset points",
                color=color, alpha=0.75, fontsize=7, va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e1-summary-csv", type=Path, default=None,
                         help="E1's summary CSV, already per-nuclide "
                              "extracted (extract_nuclide.py). Required "
                              "unless --skip-nscm is given.")
    parser.add_argument("--e9-summary-csv", type=Path, default=None,
                         help="E9's summary CSV, already per-nuclide "
                              "extracted, --experiment e9_fixed_duration. "
                              "Required unless --skip-nscm is given.")
    parser.add_argument("--skip-nscm", action="store_true",
                         help="Plot ONLY Method B (--e1b-summary-csv/"
                              "--e9b-summary-csv, both required in this "
                              "mode) -- a standalone Method B figure "
                              "instead of the default NSCM-vs-Method-B "
                              "overlay. --e1-summary-csv/--e9-summary-csv "
                              "are not needed in this mode.")
    parser.add_argument("--e1b-summary-csv", type=Path, default=None,
                         help="Optional: E1B's summary CSV (Method B, "
                              "branching fixed-source, analog of E1), "
                              "already per-nuclide extracted, --experiment "
                              "e1b_master_table. Overlays Method B on panel "
                              "(a) if given.")
    parser.add_argument("--e9b-summary-csv", type=Path, default=None,
                         help="Optional: E9B's summary CSV (Method B, "
                              "analog of E9), already per-nuclide extracted, "
                              "--experiment e9b_fixed_duration. Overlays "
                              "Method B on panel (b) if given.")
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

    if args.skip_nscm:
        if args.e1b_summary_csv is None or args.e9b_summary_csv is None:
            raise SystemExit("--skip-nscm requires both --e1b-summary-csv "
                              "and --e9b-summary-csv (nothing else to plot).")
    else:
        if args.e1_summary_csv is None or args.e9_summary_csv is None:
            raise SystemExit("--e1-summary-csv and --e9-summary-csv are "
                              "required unless --skip-nscm is given.")

    _configure_latex()

    chain_file = args.chain_file
    if chain_file is None:
        import openmc
        chain_file = openmc.config.get("chain_file")
        if chain_file is None:
            raise SystemExit("No --chain-file and openmc.config['chain_file'] unset.")
    idx = nuclide_index(chain_file, args.nuclide)
    nuc_tex = args.nuclide  # e.g. "Xe135" -- fine as-is in \text{}

    e1_rows = load_csv(args.e1_summary_csv) if not args.skip_nscm else None
    e9_rows = load_csv(args.e9_summary_csv) if not args.skip_nscm else None
    e1b_rows = load_csv(args.e1b_summary_csv) if args.e1b_summary_csv else None
    e9b_rows = load_csv(args.e9b_summary_csv) if args.e9b_summary_csv else None

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ---- panel (a): exposure-indexed ----
    # Exponents below are RELATIVE variance Var(X)/X^2 -- Table I's
    # "(relative)" rows directly, or absolute/X^2 derived by hand
    # otherwise (X^2 ~ M^2 for scale-dependent R = S0*M*r, ~M^0 for N).
    # NSCM: color-per-quantity, solid line, circle marker, "--" theory.
    # Method B (when e1b/e9b rows are supplied): the SAME color as its
    # NSCM counterpart -- delineated instead by dashed line, square
    # marker, ":" theory line, and an explicit "Method B" in both the
    # legend text and the in-plot theory annotation (see _add_line's
    # docstring). Two different theory bases are in play for Method B,
    # both explained at their call sites below.
    if not args.skip_nscm:
        Ms, s, ns = _series_from_csv(e1_rows, args.case, f"N_end_{args.nuclide}")
        _add_line(ax_a, Ms, s, ns, rf"$N(\tau_*)$", 0, "tab:blue", method_label="NSCM")

        Ms, s, ns = _series_from_csv(e1_rows, args.case, "t_end")
        _add_line(ax_a, Ms, s, ns, rf"$t(\tau_*)$", 2, "tab:orange", method_label="NSCM")

        Ms, s, ns = _series_from_csv(e1_rows, args.case, "M_end")
        _add_line(ax_a, Ms, s, ns, rf"$R(\tau_*)=S_0M$", 2, "tab:green", method_label="NSCM")

    if e1b_rows is not None:
        # Theory exponents from the paper's "Physical Chain" column
        # (relative), NOT "Branching Fixed-Source": Branching is
        # explicitly at fixed total tracking work C~P*M (one extra
        # power of M vs. per-history variance for any non-exposure-
        # conditioned endpoint), but E1B/E9B run at fixed (n_particles,
        # n_generations) -- fixed PRIMARY history count P, not fixed
        # work C. A fixed-P experiment measures the same M-scaling as
        # the underlying physical process itself (an overall 1/P
        # prefactor doesn't touch the M-exponent), which is exactly
        # what "Physical Chain" reports. These two rows are both
        # exposure-conditioned endpoints, where the paper states
        # Physical and Branching coincide anyway (work to deliver a
        # prescribed exposure is O(tau_star), M-independent) -- the
        # distinction only bites for panel (b)'s calendar endpoints.
        Ms, s, ns = _series_from_csv(e1b_rows, args.case, f"N_end_{args.nuclide}")
        _add_line(ax_a, Ms, s, ns, rf"$N(\tau_*)$", 0, "tab:blue",
                   marker="s", linestyle="--", theory_linestyle=":",
                   method_label="Method B")
        Ms, s, ns = _series_from_csv(e1b_rows, args.case, "M_end")
        _add_line(ax_a, Ms, s, ns, rf"$R(\tau_*)=S_0M$", 1, "tab:green",
                   marker="s", linestyle="--", theory_linestyle=":",
                   method_label="Method B")

    ax_a.set_xlabel(r"$M = 1/(1-k)$")
    ax_a.set_ylabel(r"$\operatorname{Var}(X)/X^2$")
    ax_a.set_title(r"(a) exposure-indexed quantities, Method B" if args.skip_nscm
                   else r"(a) exposure-indexed quantities")
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.grid(True, which="both", alpha=0.25)

    # ---- panel (b): calendar-indexed ----
    if not args.skip_nscm:
        Ms, s, ns = _series_matched_burnup(args.cache_root, args.case, args.k_targets,
                                            args.n_steps, args.max_replicas, idx, "N")
        _add_line(ax_b, Ms, s, ns, rf"$N(t_*)$, matched burnup", 2, "tab:blue",
                  method_label="NSCM")

        Ms, s, ns = _series_matched_burnup(args.cache_root, args.case, args.k_targets,
                                            args.n_steps, args.max_replicas, idx, "R")
        _add_line(ax_b, Ms, s, ns, rf"$R(t_*)$, matched burnup", 4, "tab:cyan",
                  method_label="NSCM")

        Ms, s, ns = _series_from_csv(e9_rows, args.case, f"N_end_{args.nuclide}")
        _add_line(ax_b, Ms, s, ns, rf"$N(t_*)$, fixed duration", 4, "tab:red",
                  method_label="NSCM")

        Ms, s, ns = _series_from_csv(e9_rows, args.case, "M_end")
        _add_line(ax_b, Ms, s, ns, rf"$R(t_*)$, fixed duration", 6, "tab:purple",
                  method_label="NSCM")

    if e1b_rows is not None:
        Ms, s, ns = _series_matched_burnup(args.cache_root, args.case,
                                            args.k_targets, args.n_steps,
                                            args.max_replicas, idx, "N",
                                            experiment="e1b_master_table")
        _add_line(ax_b, Ms, s, ns, rf"$N(t_*)$, matched burnup", 2, "tab:blue",
                   marker="s", linestyle="--", theory_linestyle=":",
                   method_label="Method B")
        # R(t*), matched burnup: the paper's "Physical Chain" column is
        # blank ("--") for this row -- fixed-P Monte Carlo has no
        # per-history baseline the paper derives here. Rather than omit
        # the theory line entirely, this uses "Branching Fixed-Source"
        # (relative M^4, same value as NSCM's own line above) as the
        # best available reference: it is NOT the same normalization
        # basis as the four "Physical Chain" lines elsewhere in this
        # figure (per unit work C~PM, not per history P), so its theory
        # annotation says so explicitly rather than presenting it as
        # equivalent to the others.
        Ms, s, ns = _series_matched_burnup(args.cache_root, args.case,
                                            args.k_targets, args.n_steps,
                                            args.max_replicas, idx, "R",
                                            experiment="e1b_master_table")
        _add_line(ax_b, Ms, s, ns, rf"$R(t_*)$, matched burnup", 4, "tab:cyan",
                   marker="s", linestyle="--", theory_linestyle=":",
                   method_label="Method B (per-cost theory)")

    if e9b_rows is not None:
        # E9B is a direct, native mirror of E9 (same fixed t_max, same
        # delivery convention) -- no approximation involved here, unlike
        # E1B. N(t*) fixed duration: Physical Chain column gives M^3
        # (relative, same as absolute since a_X=0 for N), NOT the M^4
        # NSCM's own line above uses -- genuinely different predictions
        # for the two methods, not a copy of one onto the other.
        Ms, s, ns = _series_from_csv(e9b_rows, args.case, f"N_end_{args.nuclide}")
        _add_line(ax_b, Ms, s, ns, rf"$N(t_*)$, fixed duration", 3, "tab:red",
                   marker="s", linestyle="--", theory_linestyle=":",
                   method_label="Method B")
        # R(t*), fixed duration: same "Physical Chain" blank as the
        # matched-burnup R row above -- "Branching Fixed-Source"
        # (relative M^6, matching NSCM's own line) used instead, same
        # per-cost-not-per-history caveat.
        Ms, s, ns = _series_from_csv(e9b_rows, args.case, "M_end")
        _add_line(ax_b, Ms, s, ns, rf"$R(t_*)$, fixed duration", 6, "tab:purple",
                   marker="s", linestyle="--", theory_linestyle=":",
                   method_label="Method B (per-cost theory)")

    ax_b.set_xlabel(r"$M = 1/(1-k)$")
    ax_b.set_ylabel(r"$\operatorname{Var}(X)/X^2$")
    ax_b.set_title(r"(b) calendar-indexed quantities, Method B" if args.skip_nscm
                   else r"(b) calendar-indexed quantities")
    ax_b.legend(fontsize=7, loc="upper left")
    ax_b.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"))
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"[fig] wrote {args.out.with_suffix('.pdf')} and "
          f"{args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
