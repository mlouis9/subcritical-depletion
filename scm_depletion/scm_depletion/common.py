"""Shared statistical and bookkeeping utilities for the numerical program.

Mirrors the conventions already used for the parent paper's static-case
numerical study (``smcommon.py``, ``caseregistry.py``): replica statistics
with their own reported uncertainty, a fixed-effects (never pooled)
regression for exponents fit across benchmark families, deterministic
cache keys and seeds so that a campaign is exactly reproducible and
resumable, and the proportional budget allocation derived in Sec. 6.6.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "cache_key", "seed_for", "replica_stats", "ratio_with_err",
    "proportional_allocation", "fixed_effects_powerlaw",
    "interpolate_replicas_at_time",
]


# --------------------------------------------------------------------------
# Cache keys and seeds
# --------------------------------------------------------------------------

def cache_key(**params) -> str:
    """A short, deterministic, filesystem-safe key for a parameter set.

    Sorted-JSON-then-hash, so the same logical case always maps to the
    same directory regardless of keyword order, and any change to a
    parameter value changes the key (rather than silently reusing stale
    cached results).
    """
    blob = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    tag = "_".join(f"{k}-{v}" for k, v in sorted(params.items())
                   if isinstance(v, (int, float, str, bool)))
    tag = tag[:80] if tag else "case"
    return f"{tag}_{digest}"


def seed_for(*parts, salt: str = "scm-depletion") -> int:
    """A deterministic 31-bit seed derived from an arbitrary tuple of
    identifying parts (case name, k target, replica index, ...), so that
    replicas are independent, reproducible, and never accidentally
    reused across different (case, k, ensemble) combinations.
    """
    blob = salt + "|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha1(blob.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1) + 1


# --------------------------------------------------------------------------
# Replica statistics
# --------------------------------------------------------------------------

def replica_stats(values: np.ndarray) -> dict:
    """Mean, standard error of the mean, sample variance, and the
    variance's own relative standard error sqrt(2/(R-1)) (paper Sec. 7.1).
    """
    values = np.asarray(values, dtype=np.float64)
    R = values.shape[0]
    if R < 2:
        raise ValueError("replica_stats requires at least 2 replicas")
    mean = values.mean(axis=0)
    var = values.var(axis=0, ddof=1)
    sem = np.sqrt(var / R)
    var_rel_se = np.sqrt(2.0 / (R - 1))
    return {"mean": mean, "sem": sem, "var": var,
            "var_rel_se": var_rel_se, "R": R}


def ratio_with_err(a, a_err, b, b_err, cov_ab: float = 0.0):
    """Ordinary uncorrelated (or partially correlated, via ``cov_ab``)
    first-order error propagation for a/b, matching the project's existing
    ``smcommon.ratio_with_err`` convention.
    """
    ratio = a / b
    rel_var = (a_err / a) ** 2 + (b_err / b) ** 2 - 2.0 * cov_ab / (a * b)
    return ratio, abs(ratio) * np.sqrt(max(rel_var, 0.0))


# --------------------------------------------------------------------------
# Budget allocation (Sec. 6.6)
# --------------------------------------------------------------------------

def proportional_allocation(step_sizes: Sequence[float],
                             total_budget: float) -> np.ndarray:
    """n_l proportional to h_l (Eq. 44): the variance-optimal allocation of
    a fixed total budget across a (possibly non-uniform) exposure or
    calendar grid. Returns floating-point history counts; round at the
    call site if an integer particle count is required.
    """
    w = np.asarray(step_sizes, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError("all step sizes must be positive")
    return total_budget * w / w.sum()


# --------------------------------------------------------------------------
# Fixed-effects regression (Sec. 7.4, 7.11)
# --------------------------------------------------------------------------

def fixed_effects_powerlaw(x_by_group: Dict[str, np.ndarray],
                            y_by_group: Dict[str, np.ndarray],
                            y_err_by_group: Dict[str, np.ndarray] = None
                            ) -> Tuple[float, float, Dict[str, float]]:
    """Fit one shared power-law exponent p and one amplitude per group to
    y ~ amplitude_g * x^p, i.e. log y = log(amplitude_g) + p * log(x), by
    weighted least squares on the design matrix [one-hot group | log x].

    This is the fixed-effects (never pooled) regression already used for
    the finite-population coefficient c_sat in the parent paper, applied
    here to the M-exponents of the master scaling table (Sec. 7.4) and to
    the step-size stability collapse (Sec. 7.8). Pooling across benchmark
    families directly is a version of Simpson's paradox for regression and
    must be avoided.

    Returns
    -------
    (p_hat, p_std, amplitudes) : shared exponent, its standard error, and
        a dict of per-group amplitude estimates.
    """
    groups = list(x_by_group.keys())
    rows_x, rows_y, rows_w, group_index = [], [], [], []
    for gi, g in enumerate(groups):
        x = np.asarray(x_by_group[g], dtype=np.float64)
        y = np.asarray(y_by_group[g], dtype=np.float64)
        if y_err_by_group is not None:
            w = 1.0 / np.asarray(y_err_by_group[g], dtype=np.float64) ** 2
        else:
            w = np.ones_like(y)
        rows_x.append(np.log(x))
        rows_y.append(np.log(y))
        rows_w.append(w)
        group_index.append(np.full(x.shape, gi))

    log_x = np.concatenate(rows_x)
    log_y = np.concatenate(rows_y)
    weight = np.concatenate(rows_w)
    gidx = np.concatenate(group_index)

    n_groups = len(groups)
    design = np.zeros((log_x.size, n_groups + 1))
    for gi in range(n_groups):
        design[gidx == gi, gi] = 1.0
    design[:, -1] = log_x

    sw = np.sqrt(weight)
    design_w = design * sw[:, None]
    target_w = log_y * sw

    coef, residuals, rank, sv = np.linalg.lstsq(design_w, target_w, rcond=None)

    dof = max(log_x.size - (n_groups + 1), 1)
    resid = design_w @ coef - target_w
    reduced_chi2 = float(resid @ resid) / dof
    try:
        cov = reduced_chi2 * np.linalg.inv(design_w.T @ design_w)
        p_std = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError:
        p_std = float("nan")

    p_hat = float(coef[-1])
    amplitudes = {g: float(np.exp(coef[gi])) for gi, g in enumerate(groups)}
    return p_hat, p_std, amplitudes


# --------------------------------------------------------------------------
# Interpolate-per-replica-then-average (prescription #9, Sec. 6.8)
# --------------------------------------------------------------------------

def interpolate_replicas_at_time(trajectories: Sequence, t_star: float,
                                  mat_index: int = 0) -> dict:
    """Interpolate N(t*) on each replica trajectory *individually*, then
    take replica statistics of the interpolated values.

    Averaging trajectories first and interpolating the average is biased,
    because the exposure-to-time map is nonlinear and replica-specific
    (Sec. 6.6, prescription #9); this function performs the correct order
    of operations.
    """
    values = np.stack([traj.interpolate_at_time(t_star, mat_index=mat_index)["N"]
                        for traj in trajectories])
    return replica_stats(values)
