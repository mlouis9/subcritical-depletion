"""Shared setup/CLI boilerplate for the E1-E8 experiment drivers.

Every experiment script in this directory follows the same pattern:
parse a small set of arguments identifying (case, k target, replica),
build the model and operator for that point, run the appropriate
ensemble/integrator, cache the result under a deterministic path, and
(for the last replica of a batch, or in a separate aggregation pass)
summarize replica statistics to a CSV row. This module factors out the
parts that are identical across all eight scripts so each one only has
to state what is specific to its own measurement.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path

# Make the package importable when these scripts are run directly from
# the `experiments/` directory without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scm_depletion import caseregistry, common  # noqa: E402


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--case", required=True, choices=list(caseregistry.CASE_REGISTRY),
                    help="Benchmark case name (see scm_depletion.caseregistry).")
    p.add_argument("--k-target", type=float, required=True,
                    help="Target multiplication k for this point.")
    p.add_argument("--alpha", type=float, default=None,
                    help="Pre-calibrated nu-multiplier achieving --k-target for "
                         "this case. Either this or --alpha-lookup-json must be "
                         "given.")
    p.add_argument("--alpha-lookup-json", type=Path, default=None,
                    help="Path to the JSON table written by "
                         "run_calibrate_alpha.py; if given, alpha is looked "
                         "up as table[case][str(k_target)] instead of being "
                         "passed explicitly via --alpha.")
    p.add_argument("--replica", type=int, required=True,
                    help="Replica index; determines the RNG seed via "
                         "common.seed_for(case, k-target, replica) and the "
                         "output subdirectory.")
    p.add_argument("--n-particles", type=int, default=None,
                    help="Particles per generation. Defaults to the case's "
                         "default_n_particles.")
    p.add_argument("--n-generations", type=int, default=None,
                    help="Active generations per depletion step. Defaults to "
                         "the case's default_n_generations.")
    p.add_argument("--n-steps", type=int, default=200,
                    help="Number of depletion steps in the primary grid.")
    p.add_argument("--tau-max", type=float, default=None,
                    help="Exposure-window upper bound for exposure-mode runs. "
                         "If omitted, the case must supply a default via "
                         "case-specific logic in the calling script.")
    p.add_argument("--cache-root", type=lambda s: Path(s).resolve(),
                   default=Path("/home/louimatt3/projects/openmc/scm_tests/"
                                "scm_depletion_cache").resolve(),
                   help="Root directory for cache-keyed trajectory storage "
                        "(defaults to the Sawtooth scratch layout used "
                        "elsewhere in this project). Resolved to an "
                        "absolute path immediately: a relative value here "
                        "would otherwise be silently reinterpreted against "
                        "whatever directory scm_transport."
                        "isolated_transport_session has chdir'd into by "
                        "the time it's used inside integrator.run().")
    p.add_argument("--summary-csv", type=Path, default=None,
                    help="If given, append a one-line summary to this CSV "
                         "after the run (header written if the file does "
                         "not yet exist).")
    return p


def resolve_alpha(args) -> float:
    """Return args.alpha directly, or look it up from
    --alpha-lookup-json[case][k_target] if --alpha was not given.
    """
    if args.alpha is not None:
        return args.alpha
    if args.alpha_lookup_json is not None:
        table = json.loads(Path(args.alpha_lookup_json).read_text())
        try:
            return float(table[args.case][str(args.k_target)])
        except KeyError as exc:
            raise SystemExit(
                f"No calibrated alpha for case={args.case!r} "
                f"k_target={args.k_target} in {args.alpha_lookup_json}"
            ) from exc
    raise SystemExit("Either --alpha or --alpha-lookup-json must be given.")


def case_and_defaults(args):
    case = caseregistry.CASE_REGISTRY[args.case]
    n_particles = args.n_particles or case.default_n_particles
    n_generations = args.n_generations or case.default_n_generations
    return case, n_particles, n_generations


def replica_cache_dir(cache_root: Path, experiment: str, args) -> Path:
    key = common.cache_key(case=args.case, k=args.k_target,
                            n_particles=args.n_particles,
                            n_generations=args.n_generations,
                            n_steps=args.n_steps)
    return cache_root / experiment / key / f"replica_{args.replica:04d}"


def seed_for_run(args) -> int:
    return common.seed_for(args.case, args.k_target, args.replica)


def _task_id() -> str:
    """A stable identifier for the current process, for per-task summary
    files. Prefers the SLURM array identity (stable and human-readable
    in a job's file listing); falls back to the PID for interactive/
    non-SLURM runs.
    """
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id and task_id:
        return f"{job_id}_{task_id}"
    return f"pid{os.getpid()}"


def append_summary_row(csv_path: Path, row: dict):
    """Append one row to this experiment's summary CSV.

    Does NOT append directly to ``csv_path`` -- E1-E8 are normally run as
    SLURM arrays, with many tasks calling this against the *same*
    csv_path concurrently, and "if the file doesn't exist yet, write a
    header" is a check-then-act race across processes: two tasks
    starting close together can both see "doesn't exist" and both write
    a header line into the middle of the file, corrupting it for
    plot_results.py's csv.DictReader-based loading. Writes within a
    single process are safe (this function may be called many times per
    task, e.g. once per probe in E3); it's only the cross-*process*
    sharing of one file that isn't.

    Instead, each task's rows accumulate in their own small file under
    ``<csv_path>.rows/<task_id>.csv``, named by SLURM array identity (or
    PID outside SLURM) -- no two concurrent processes ever write the
    same file. Call :func:`collect_summary_csv` once, after every array
    task has finished, to merge these into ``csv_path`` itself before
    plotting; plot_results.py reads the merged file, not the per-task
    ones.
    """
    if csv_path is None:
        return
    rows_dir = csv_path.parent / f"{csv_path.stem}.rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    task_path = rows_dir / f"{_task_id()}.csv"
    write_header = not task_path.exists()
    with open(task_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def collect_summary_csv(csv_path: Path) -> int:
    """Merge every per-task file under ``<csv_path>.rows/`` (written by
    :func:`append_summary_row`) into ``csv_path`` itself. Run this once,
    after every SLURM array task for an experiment has finished, before
    handing csv_path to plot_results.py.

    Different tasks can legitimately have different columns -- E1's rows
    gain three extra columns when --t-star is given, and not every task
    in an array necessarily uses it the same way -- so this computes the
    union of fieldnames across every per-task file rather than assuming
    they match; a row missing a column the union has gets a blank entry
    for it, not an error.

    Returns the number of rows written. Safe to re-run: it always
    rebuilds csv_path from the current contents of the .rows directory,
    so re-running after more array tasks finish (or after a rerun of a
    failed task) picks up everything currently on disk.
    """
    rows_dir = csv_path.parent / f"{csv_path.stem}.rows"
    task_files = sorted(rows_dir.glob("*.csv")) if rows_dir.exists() else []
    if not task_files:
        raise FileNotFoundError(
            f"No per-task files found under {rows_dir} -- nothing to "
            "collect. Did the experiment's SLURM array actually run "
            "(and pass --summary-csv), or has collect_summary_csv "
            "already been pointed at the wrong path?")

    all_rows = []
    fieldnames_union: list[str] = []
    seen = set()
    for task_file in task_files:
        with open(task_file, newline="") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)
                for k in row.keys():
                    if k not in seen:
                        seen.add(k)
                        fieldnames_union.append(k)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_union)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[collect] {len(all_rows)} rows from {len(task_files)} task files "
          f"-> {csv_path}")
    return len(all_rows)


def transport_workdir(cache_root: Path, experiment: str, args, tag: str = "_transport") -> Path:
    """A directory unique to this (experiment, case, k_target, particle/
    generation/step counts, replica) point, for
    :func:`build_operator_and_model` to isolate ``openmc.lib``'s working
    directory in. Deliberately reuses the same cache-key scheme as
    :func:`replica_cache_dir`, under a ``tag`` subdirectory, so a script
    building more than one operator (E2, E4, E8) can pass a distinct
    ``tag`` per operator and get distinct, discoverable directories
    sitting next to that operator's own results rather than colliding.
    """
    return replica_cache_dir(cache_root, experiment, args) / tag


@contextlib.contextmanager
def build_operator_and_model(case, args, n_particles, n_generations,
                              cache_dir: Path, workdir: Path,
                              run_mode: str = "subcritical multiplication",
                              operator_cls=None,
                              calculate_subcritical_k: bool = False):
    """Build the model and operator for one replica, as a context
    manager: the ``with`` block must contain everything that touches
    this operator, from construction through the integrator's last
    ``operator(vec)`` call, because ``openmc.lib``'s working directory
    and initialized state are only isolated/valid for the block's
    duration -- see
    :func:`scm_depletion.scm_transport.isolated_transport_session` for
    exactly which two failure modes this prevents (a concurrent-task
    file-corruption crash, and a silent stale-geometry bug when a
    second operator is built later in the same process).

    Usage::

        cache_dir = replica_cache_dir(args.cache_root, "e1_master_table", args)
        workdir = transport_workdir(args.cache_root, "e1_master_table", args)
        with build_operator_and_model(case, args, n_particles,
                                       n_generations, cache_dir, workdir) as \\
                (model, operator, initial_vec):
            integrator = ExposureIntegrator(operator, initial_vec, ...)
            integrator.run(trajectory, dtau_grid, resume=True)

    This is the point at which the case's model_builder (a fork-specific
    integration point -- see caseregistry.py) is invoked, along with any
    per-replica settings (seed, particle/generation counts) that are the
    same across every experiment. The nu-multiplier alpha is resolved via
    :func:`resolve_alpha` (either --alpha directly or a lookup from
    --alpha-lookup-json).

    ``run_mode``/``operator_cls``/``calculate_subcritical_k`` are
    keyword-only with defaults matching the previous, SCM-only behavior
    exactly, so E1/E9/calibration needed no changes to keep working. For
    Method B (E1B/E9B, branching fixed-source), pass
    ``run_mode="fixed source"``,
    ``operator_cls=BranchingFixedSourceOperator``, and
    ``calculate_subcritical_k=True`` -- the last of these is what keeps
    the k/kq/ks C-API readout valid outside SUBCRITICAL_MULTIPLICATION
    mode (see ``scm_transport._fetch_k_factors``'s docstring).

    ``run_mode`` is set here, not by ``model_builder``, because the same
    model_builder is shared with :func:`~scm_depletion.caseregistry.
    calibrate_alpha_for_keff`, which needs ordinary eigenvalue mode. See
    the ``BenchmarkCase.model_builder`` docstring in caseregistry.py.

    The initial nuclide vector comes from
    ``operator.initial_condition()`` (inherited from
    ``openmc.deplete.CoupledOperator``), which also performs
    ``openmc.lib.init()``; there is no separate hand-built vector to
    maintain per case.

    **The seed is derived here, not by the caller, and depends on how
    many steps are already checkpointed at ``cache_dir``** (0 for a
    fresh run). This matters for correctness, not just style: a SLURM
    timeout followed by a resubmission starts a *new process*, and
    ``openmc.lib``'s random-number stream position is in-memory state
    that a fresh process cannot inherit -- it starts over from whatever
    ``settings.seed`` implies. A seed that depended only on (case,
    k_target, replica) would therefore make the resumed process's first
    batch replay the *exact* random draws the original process's first
    batch already used, rather than continuing with fresh, independent
    ones -- silently breaking the cross-step independence every
    variance/M-scaling result in the theory note assumes, specifically
    for trajectories that happen to span more than one process. This has
    no effect on a trajectory that completes within one uninterrupted
    process (the common case with a generous --time), and it changes
    nothing about *which* cache directory a resumed run targets --
    array-index-to-parameter mapping in the submit_*.sh scripts is
    already deterministic and untouched by this.
    """
    from scm_depletion.scm_transport import (
        SCMCoupledOperator, isolated_transport_session)
    from scm_depletion.trajectory import SCMTrajectory

    n_done = 0
    if (Path(cache_dir) / "trajectory.pkl").exists():
        n_done = SCMTrajectory.load(cache_dir).n_steps
    seed = common.seed_for(args.case, args.k_target, args.replica, n_done)

    with isolated_transport_session(workdir):
        model = case.model_builder(resolve_alpha(args))
        model.settings.run_mode = run_mode
        if calculate_subcritical_k:
            model.settings.calculate_subcritical_k = True
        model.settings.seed = seed
        model.settings.particles = n_particles
        model.settings.batches = n_generations  # active cycles per depletion step
        op_cls = operator_cls or SCMCoupledOperator
        operator = op_cls(model, source_strength=case.source_strength)
        initial_vec = operator.initial_condition()
        yield model, operator, initial_vec
