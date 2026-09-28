#!/usr/bin/env python3
"""Run the full per-nuclide post-processing suite in one command.

Replaces the "extract -> plot variance scaling -> plot trajectory bands"
sequence in README_SCM_DEPLETION.md, looped over as many nuclides as you
like, and adds the two diagnostics written alongside this script
(make_fig_trajectory_bands_relative.py, diagnose_R_variance.py) so a
single invocation produces everything needed to read one nuclide's
results:

    outputs/<nuclide>/
        e1_summary.<nuclide>.csv
        e9_summary.<nuclide>.csv
        [e1b/e9b variants if requested]
        fig_variance_scaling.pdf / .png
        fig_trajectory_bands_relative.pdf / .png
        clock_variance_report.txt

Nothing here re-runs transport or touches the cache except to *read*
already-checkpointed trajectory.pkl files -- this is purely a
post-processing convenience wrapper around the existing, independently
tested CLI scripts (extract_nuclide.py, make_fig_variance_scaling.py)
plus the two new ones. It shells out to them rather than importing their
`main` functions, deliberately: each is independently useful and
independently tested from the command line, and shelling out means a
failure in one nuclide's post-processing (missing trajectories, a
nuclide not in the chain, etc.) can't corrupt another's in-process state.

Usage
-----
python run_nuclide_suite.py \\
    --e1-summary-csv cache/e1_master_table_summary.csv \\
    --e9-summary-csv cache/e9_fixed_duration_summary.csv \\
    --cache-root cache --case heu_sphere_nu \\
    --k-targets 0.50 0.70 0.80 0.90 0.95 0.97 0.98 0.99 0.995 \\
    --n-steps 200 \\
    --nuclides Xe135 I131 U235 \\
    --out-root outputs
    [--e1b-summary-csv ... --e9b-summary-csv ...]   # optional Method B overlay
    [--t-star 1.728e7]                              # optional, for the
                                                      # clock-variance diagnostic;
                                                      # defaults to 90% of each
                                                      # k_target's own shortest
                                                      # replica t_end
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # extract_nuclide.py, make_fig_variance_scaling.py,
                                          # make_fig_trajectory_bands_relative.py, and
                                          # diagnose_R_variance.py are expected to sit next to
                                          # this script -- i.e. all four ship in experiments/.


def run(cmd, **kw):
    print(f"[suite] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e1-summary-csv", type=Path, required=True)
    p.add_argument("--e9-summary-csv", type=Path, required=True)
    p.add_argument("--e1b-summary-csv", type=Path, default=None)
    p.add_argument("--e9b-summary-csv", type=Path, default=None)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--case", default="heu_sphere_nu")
    p.add_argument("--k-targets", type=float, nargs="+", required=True)
    p.add_argument("--n-steps", type=int, required=True)
    p.add_argument("--max-replicas", type=int, default=12)
    p.add_argument("--nuclides", nargs="+", required=True)
    p.add_argument("--chain-file", default=None)
    p.add_argument("--t-star", type=float, default=None,
                    help="Calendar target for diagnose_R_variance.py. If "
                         "omitted, each k_target uses its own 0.9*min(replica "
                         "t_end), printed to the console so you can reuse it.")
    p.add_argument("--out-root", type=Path, default=Path("outputs"))
    p.add_argument("--skip-trajectory-bands", action="store_true",
                    help="Trajectory-band figures redraw every replica's full "
                         "curve and are the slowest/heaviest part of this "
                         "suite; skip them for a quick scaling-only pass.")
    p.add_argument("--skip-clock-diagnostic", action="store_true")
    args = p.parse_args()

    chain_args = ["--chain-file", args.chain_file] if args.chain_file else []

    for nuclide in args.nuclides:
        out_dir = args.out_root / nuclide
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[suite] ==== {nuclide} -> {out_dir} ====")

        # ---- 1. per-nuclide extraction (E1, E9, and Method B variants) ----
        extracted = {}
        for tag, src_csv, experiment in [
            ("e1", args.e1_summary_csv, "e1_master_table"),
            ("e9", args.e9_summary_csv, "e9_fixed_duration"),
            ("e1b", args.e1b_summary_csv, "e1b_master_table"),
            ("e9b", args.e9b_summary_csv, "e9b_fixed_duration"),
        ]:
            if src_csv is None:
                continue
            out_csv = out_dir / f"{tag}_summary.{nuclide}.csv"
            run([sys.executable, HERE / "extract_nuclide.py",
                 "--summary-csv", src_csv, "--cache-root", args.cache_root,
                 "--experiment", experiment, "--nuclide", nuclide,
                 *chain_args, "--out-csv", out_csv])
            extracted[tag] = out_csv

        # ---- 2. variance-scaling figure (panels a/b) ----
        cmd = [sys.executable, HERE / "make_fig_variance_scaling.py",
               "--e1-summary-csv", extracted["e1"],
               "--e9-summary-csv", extracted["e9"],
               "--cache-root", args.cache_root, "--case", args.case,
               "--k-targets", *[str(k) for k in args.k_targets],
               "--n-steps", args.n_steps, "--max-replicas", args.max_replicas,
               "--nuclide", nuclide, *chain_args,
               "--out", out_dir / "fig_variance_scaling"]
        if "e1b" in extracted:
            cmd += ["--e1b-summary-csv", extracted["e1b"]]
        if "e9b" in extracted:
            cmd += ["--e9b-summary-csv", extracted["e9b"]]
        run(cmd)

        # ---- 3. trajectory-band figure, RELATIVE-deviation variant ----
        if not args.skip_trajectory_bands:
            run([sys.executable, HERE / "make_fig_trajectory_bands_relative.py",
                 "--cache-root", args.cache_root, "--case", args.case,
                 "--k-targets", *[str(k) for k in args.k_targets],
                 "--n-steps", args.n_steps, "--max-replicas", args.max_replicas,
                 "--nuclide", nuclide, *chain_args,
                 "--out", out_dir / "fig_trajectory_bands_relative"])

        # ---- 4. R(t*) variance decomposition (clock spread vs. local k noise) ----
        if not args.skip_clock_diagnostic:
            cmd = [sys.executable, HERE / "diagnose_R_variance.py",
                   "--cache-root", args.cache_root, "--case", args.case,
                   "--k-targets", *[str(k) for k in args.k_targets],
                   "--n-steps", args.n_steps, "--max-replicas", args.max_replicas,
                   "--out-txt", out_dir / "clock_variance_report.txt"]
            if args.t_star is not None:
                cmd += ["--t-star", args.t_star]
            run(cmd)

    print(f"\n[suite] done -- see {args.out_root}/<nuclide>/")


if __name__ == "__main__":
    main()
