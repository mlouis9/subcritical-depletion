#!/usr/bin/env python3
"""Extract a single nuclide's inventory from already-cached E1
trajectories, no rerun required -- for when the full-inventory sum
(N_end_sum, N_at_t_star_sum) is dominated by near-static actinide mass
and can't resolve the genuine, much smaller variation among fission
products (see plot_results.py's own diagnostic message for this).

Every trajectory.pkl already stores the full per-nuclide density vector
at every checkpointed step (StepRecord.vec); this only needed working
out which vector index corresponds to which nuclide name, via the
depletion chain file (openmc.deplete.Chain.from_xml), which is a light,
transport-free load -- not a rerun of anything.

Writes an enriched copy of --summary-csv with two new columns,
N_end_<nuclide> and (if --t-star is given) N_at_t_star_<nuclide>, ready
for: python plot_results.py e1 --summary-csv <output> --nuclide <name>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scm_depletion.trajectory import SCMTrajectory, WindowTooShort
from analysis_common import find_trajectory_dir


def nuclide_index(chain_file: str, nuclide: str) -> int:
    """The vector-column index for `nuclide` in this chain -- a light,
    transport-free load (no openmc.lib, no model, no operator), since
    the ordering is a property of the chain file alone.
    """
    from openmc.deplete import Chain
    chain = Chain.from_xml(chain_file)
    names = [n.name for n in chain.nuclides]
    if nuclide not in names:
        raise SystemExit(
            f"{nuclide!r} is not in this chain ({chain_file}). "
            f"Available: {', '.join(sorted(names))}")
    return names.index(nuclide)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--cache-root", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--experiment", default="e1_master_table",
                         help="Same tag the original run used with "
                              "replica_cache_dir/transport_workdir.")
    parser.add_argument("--nuclide", required=True,
                         help="e.g. Xe135, Cs137, U235.")
    parser.add_argument("--chain-file", default=None,
                         help="Defaults to openmc.config['chain_file'] "
                              "(whatever the original run used).")
    parser.add_argument("--mat-index", type=int, default=0,
                         help="Which burnable material's vector to read "
                              "(0 unless the case has more than one).")
    parser.add_argument("--t-star", type=float, default=None,
                         help="If given, also compute N_at_t_star_<nuclide> "
                              "by interpolation (see backfill_t_star.py for "
                              "the monotonicity caveat this is subject to "
                              "as well).")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    chain_file = args.chain_file
    if chain_file is None:
        import openmc
        chain_file = openmc.config.get("chain_file")
        if chain_file is None:
            raise SystemExit(
                "No --chain-file given and openmc.config['chain_file'] is "
                "not set; pass --chain-file explicitly.")

    idx = nuclide_index(chain_file, args.nuclide)
    print(f"[extract] {args.nuclide} is column {idx} in {chain_file}")

    rows = list(csv.DictReader(open(args.summary_csv, newline="")))
    if not rows:
        raise SystemExit(f"{args.summary_csv} has no rows.")

    end_col = f"N_end_{args.nuclide}"
    tstar_col = f"N_at_t_star_{args.nuclide}"

    n_ok = n_skipped_missing = n_skipped_nonmonotonic = n_skipped_window = 0
    for row in rows:
        cache_dir = find_trajectory_dir(
            args.cache_root, args.experiment,
            case=row["case"], k_target=row["k_target"],
            n_steps=row["n_steps"], replica=int(row["replica"]))

        if cache_dir is None or not (cache_dir / "trajectory.pkl").exists():
            n_skipped_missing += 1
            print(f"[extract] SKIP case={row['case']} k_target={row['k_target']} "
                  f"replica={row['replica']}: no trajectory.pkl found.")
            continue

        trajectory = SCMTrajectory.load(cache_dir)
        row[end_col] = repr(float(trajectory.records[-1].vec[args.mat_index][idx]))
        n_ok += 1

        if args.t_star is not None:
            t = np.asarray(trajectory.t)
            bad_steps = np.where(np.diff(t) <= 0)[0]
            if len(bad_steps) > 0:
                n_skipped_nonmonotonic += 1
                print(f"[extract] {row['case']} k_target={row['k_target']} "
                      f"replica={row['replica']}: non-monotonic clock, "
                      f"skipping {tstar_col} for this row (N_end_ still "
                      "written).")
                continue
            try:
                info = trajectory.interpolate_at_time(
                    args.t_star, mat_index=args.mat_index)
            except WindowTooShort:
                n_skipped_window += 1
                print(f"[extract] {row['case']} k_target={row['k_target']} "
                      f"replica={row['replica']}: t_star out of range, "
                      f"skipping {tstar_col} for this row.")
                continue
            row[tstar_col] = repr(float(info["N"][idx]))

    fieldnames = list(rows[0].keys())
    for extra in (end_col, tstar_col):
        if extra not in fieldnames and any(extra in r for r in rows):
            fieldnames.append(extra)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[extract] {n_ok} rows got {end_col}; "
          f"{n_skipped_missing} missing trajectory, "
          f"{n_skipped_nonmonotonic} non-monotonic, "
          f"{n_skipped_window} out of t_star range -> {args.out_csv}")


if __name__ == "__main__":
    main()
