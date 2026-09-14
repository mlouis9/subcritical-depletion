#!/usr/bin/env python3
"""One-time calibration: find the nu-multiplier alpha achieving each
target k_eff for a case, and write the result to a small JSON lookup
table that every E1-E8 script reads via --alpha-lookup-json (theory
note Sec. 7.1, 8.0; companion paper Remark 2).

This is deliberately separate from the E1-E8 drivers: it is run once
per (case, k_target) pair rather than once per replica, and it needs an
eigenvalue-mode transport solve rather than an SCM one.

``run_eigenvalue_fn`` below is a thin wrapper around ordinary
``openmc.lib`` eigenvalue-mode transport; if this project already has a
dedicated eigenvalue-calibration utility (referred to as ``kcalib.py``
in the parent project), prefer that over duplicating it here -- this
implementation is provided so the pipeline is usable standalone.

Each eigenvalue evaluation runs in its own temporary directory via
``isolated_transport_session`` (see scm_transport.py). This is not
optional: ``calibrate_alpha_for_keff`` bisects, calling this function
many times, and this script is normally launched as several concurrent
SLURM array tasks (one per case) that would otherwise all
``export_to_xml()``/``openmc.lib.init()``/write statepoints into the
same shared working directory at once -- the exact mechanism behind an
"unable to open file"/HDF5-corruption/MPI_ABORT crash observed running
this script without it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import openmc
import openmc.lib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scm_depletion import caseregistry
from scm_depletion.scm_transport import isolated_transport_session


def run_eigenvalue_fn(model: "openmc.Model", particles: int, batches: int) -> float:
    model.settings.run_mode = "eigenvalue"  # explicit, not relied on as a
                                             # silent Settings() default,
                                             # since model_builder is shared
                                             # with the SCM depletion drivers
    model.settings.particles = particles
    model.settings.batches = batches

    with tempfile.TemporaryDirectory(prefix="scm_calib_") as tmp:
        with isolated_transport_session(tmp):
            model.export_to_xml()
            openmc.lib.init()
            openmc.lib.run()
            return openmc.lib.keff()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, choices=list(caseregistry.CASE_REGISTRY))
    parser.add_argument("--k-targets", type=float, nargs="+", required=True)
    parser.add_argument("--bracket", type=float, nargs=2, default=[0.7, 1.3],
                         help="Starting nu-multiplier bracket, expanded "
                              "geometrically only as far as needed to "
                              "contain each k-target (see "
                              "calibrate_alpha_for_keff's docstring for "
                              "why this starts narrow rather than wide).")
    parser.add_argument("--expansion-factor", type=float, default=1.3)
    parser.add_argument("--max-expansions", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--particles", type=int, default=2000,
                         help="Kept modest by default: calibration needs "
                              "just enough precision to bisect reliably, "
                              "not depletion-quality statistics, and this "
                              "function may be called dozens of times per "
                              "k-target.")
    parser.add_argument("--batches", type=int, default=60,
                         help="Total batches (inactive+active) for each "
                              "calibration eigenvalue solve.")
    parser.add_argument("--out", type=Path, required=True,
                         help="Path to the JSON lookup table to write/update.")
    args = parser.parse_args()

    case = caseregistry.CASE_REGISTRY[args.case]

    table = {}
    if args.out.exists():
        table = json.loads(args.out.read_text())
    table.setdefault(args.case, {})

    def _run(model):
        return run_eigenvalue_fn(model, args.particles, args.batches)

    for k_target in args.k_targets:
        alpha = caseregistry.calibrate_alpha_for_keff(
            case.model_builder, k_target, _run,
            bracket=tuple(args.bracket), tol=args.tol,
            expansion_factor=args.expansion_factor,
            max_expansions=args.max_expansions)
        table[args.case][str(k_target)] = alpha
        print(f"[calibrate] case={args.case} k_target={k_target} -> alpha={alpha:.6f}")

        # Write after every k_target, not just at the end: a later
        # k_target failing (e.g. unreachable by nu-scaling alone) should
        # not discard alpha values already found for earlier ones.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
