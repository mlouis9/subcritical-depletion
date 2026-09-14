"""Shared analysis helpers for E1/E9 post-processing (extract_nuclide.py,
make_fig_variance_scaling.py). Everything here reads already-cached
trajectories or summary CSVs -- no transport, no reruns.

Consolidated from what was originally split across several exploratory
analysis scripts (paper-figure drafts that didn't survive, a fission-rate
diagnostic once used to track down a bug in an approach later abandoned)
during the codebase simplification down to E1+E9.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from scm_depletion.trajectory import SCMTrajectory, WindowTooShort


# --------------------------------------------------------------------------
# Cache lookup
# --------------------------------------------------------------------------

def find_trajectory_dir(cache_root: Path, experiment: str, case: str,
                         k_target: str, n_steps: str, replica: int):
    """Locate a replica's cache directory by globbing on the same
    human-readable prefix replica_cache_dir/common.cache_key always
    produces (``case-<case>_k-<k_target>_n_steps-<n_steps>_<hash>``),
    rather than recomputing the hash suffix ourselves.

    This is deliberate, not a shortcut: cache_key hashes *every* keyword
    it's given, including ones that don't show up in the readable
    prefix (only int/float/str/bool values are shown; anything else,
    e.g. None, still feeds the hash). replica_cache_dir is called with
    the *raw* CLI args, not the resolved n_particles/n_generations --
    so a run where --n-particles was never passed (the normal case)
    hashes n_particles=None, while reconstructing the key from a CSV's
    *resolved* n_particles column would hash n_particles=8000 -- a
    different key, a different directory, even though both refer to
    the same run. Rather than guess which convention produced a given
    CSV, this matches only on the prefix every convention shares and
    accepts whichever single directory is actually there.
    """
    exp_root = Path(cache_root) / experiment
    pattern = f"case-{case}_k-{k_target}_n_steps-{n_steps}_*"
    matches = sorted(exp_root.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous: {len(matches)} cache directories match {pattern} "
            f"under {exp_root} -- expected exactly one. Not guessing which "
            "one is right; inspect these directories by hand: "
            + ", ".join(str(m) for m in matches))
    if not matches:
        return None
    return matches[0] / f"replica_{replica:04d}"


def load_replicas(cache_root, experiment, case, k_target, n_steps, replicas):
    """All available replica trajectories for one (experiment, case,
    k_target, n_steps); missing ones are skipped, not padded with
    placeholders, and the caller sees fewer trajectories than requested
    rather than an error.
    """
    out = []
    for r in replicas:
        d = find_trajectory_dir(cache_root, experiment, case=case,
                                 k_target=k_target, n_steps=n_steps, replica=r)
        if d is not None and (d / "trajectory.pkl").exists():
            out.append(SCMTrajectory.load(d))
    return out


# --------------------------------------------------------------------------
# Nuclide indexing
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Replica statistics
# --------------------------------------------------------------------------

def sigma_pair(values):
    """(relative sigma, absolute sigma) from the same replica values."""
    values = np.asarray(values, float)
    if len(values) < 3:
        return float("nan"), float("nan")
    abs_sigma = float(values.std(ddof=1))
    rel_sigma = abs(abs_sigma / values.mean()) if values.mean() != 0 else float("nan")
    return rel_sigma, abs_sigma


def M_of_k(k: float) -> float:
    return 1.0 / (1.0 - k)


def fit_loglog(x, y):
    """Least-squares slope/intercept of log(y) vs log(x), plus R^2 of
    that fit -- a low R^2 means the "slope" is not a reliable estimate,
    most often too few distinct M values or too few replicas per point.
    Unweighted: does not account for the sqrt(2/(R-1)) uncertainty on
    each y-value (see make_fig_variance_scaling.py's own note on this).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


# --------------------------------------------------------------------------
# Safe interpolation (guards a non-monotonic clock -- a transient
# supercritical excursion, however brief, breaks PCHIP's strictly-
# increasing requirement; see trajectory.py's interpolate_at_time)
# --------------------------------------------------------------------------

def safe_interpolate_at_time(trajectory, t_star: float, mat_index: int = 0):
    """trajectory.interpolate_at_time, guarded against a non-monotonic t
    array: t is the deterministic grid for a calendar integrator, but
    for an EXPOSURE integrator (E1) t is accumulated and dt = h*q/S0 can
    go negative if k>1 transiently. Returns (info_dict, None) or
    (None, reason).
    """
    t = np.asarray(trajectory.t)
    bad = np.where(np.diff(t) <= 0)[0]
    if len(bad) > 0:
        k_there = trajectory.records[int(bad[0])].k_mid.k
        return None, (f"non-monotonic t ({len(bad)} bad step(s), first at "
                       f"step {int(bad[0])}, k_mid={k_there:.6f} -- likely a "
                       "transient supercritical excursion)")
    try:
        return trajectory.interpolate_at_time(t_star, mat_index=mat_index), None
    except WindowTooShort as exc:
        return None, str(exc)


# --------------------------------------------------------------------------
# CSV loading (no pandas dependency, deliberately)
# --------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """Read a summary CSV into a list of dicts, coercing every value that
    parses as a float. Non-numeric fields (case names) are left as
    strings.
    """
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = None
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return rows


def group_by(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return dict(groups)
