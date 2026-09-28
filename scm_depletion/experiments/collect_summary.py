#!/usr/bin/env python3
"""Merge the per-task summary files an E1 or E9 SLURM array wrote
(_expcommon.append_summary_row, one file per task under
<summary_csv>.rows/) into a single summary CSV. Run this once, after
every array task for an experiment has finished, before extract_nuclide.py
or make_fig_variance_scaling.py.

Usage::

    python collect_summary.py cache/e1_master_table_summary.csv
    python collect_summary.py cache/e9_fixed_duration_summary.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _expcommon import collect_summary_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary_csv", type=Path,
                         help="Path passed as --summary-csv to the SLURM array "
                              "(e.g. cache/e1_master_table_summary.csv).")
    args = parser.parse_args()
    collect_summary_csv(args.summary_csv)


if __name__ == "__main__":
    main()
