"""Benchmark case registry for the SCM depletion numerical program.

Mirrors the convergence-theory project's ``caseregistry.py``: a small,
explicit table of named cases, each providing a model-builder callable and
the physical/statistical defaults needed to run it, so that every
experiment script (``experiments/run_e*.py``) refers to cases by name
rather than duplicating geometry-construction code.

One thing is deliberately left as an integration point rather than
fabricated here, because this project already has working code for it
(``mg_benchmarks.py``, ``mc_runners.py``, ``kcalib.py`` per the parent
project) and duplicating it incorrectly would be worse than leaving an
explicit hook: the actual ``openmc.Model`` construction for each physical
family (HEU plate stack, 17x17 PWR, SFR, HEU sphere -- Table 3 of the
companion paper) belongs in ``model_builder``, which this module leaves as
a placeholder raising ``NotImplementedError`` with the case name in the
message. Wire in the existing project's model builders here.

``apply_nu_multiplier`` is implemented (see ``nu_multiplier.py``): it
rewrites the HDF5 nuclide data for the fissile nuclides, leaving all cross
sections and fission-product yields fixed, exactly as Remark 2 of the
companion paper does for the multigroup convergence benchmarks via a
uniform nu*Sigma_f multiplier. This requires no OpenMC C++ changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = [
    "K_GRID_TRUNCATION_STUDY", "K_GRID_FOM_STUDY", "N_GRID",
    "BenchmarkCase", "CASE_REGISTRY", "register_case",
    "apply_nu_multiplier", "calibrate_alpha_for_keff",
]

# Matches the parent paper's grids (Table 1 / Sec. 7.3-7.4), reused here so
# that depletion results are directly comparable at matched k.
K_GRID_TRUNCATION_STUDY: List[float] = [0.900, 0.950, 0.980, 0.995]
K_GRID_FOM_STUDY: List[float] = [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98,
                                  0.99, 0.995, 0.998]
N_GRID: List[int] = [500, 1000, 2000, 8000, 32000, 128000]


@dataclass
class BenchmarkCase:
    """One named depletion benchmark family.

    Attributes
    ----------
    name : str
        Case identifier, used to build cache keys and SLURM job names.
    model_builder : callable(alpha) -> openmc.Model
        Returns the model's geometry, materials (with nu already scaled
        by ``apply_nu_multiplier``), settings.source, and every other
        physical specification -- at nu-multiplier ``alpha`` (alpha=1.0
        recovers the unperturbed physical system).

        Deliberately does *not* set ``settings.run_mode``: leave it at
        the ``openmc.Settings`` default (``'eigenvalue'``), since the
        same model_builder is used both by :func:`calibrate_alpha_for_keff`
        (which needs eigenvalue mode) and by the SCM depletion drivers
        (which set ``run_mode = 'subcritical multiplication'`` themselves
        in ``experiments/_expcommon.py`` after calling this function).
        Setting ``calculate_subcritical_k`` here would additionally be
        actively wrong: its setter raises ``ValueError`` unless
        ``run_mode == 'fixed source'``, which is never this function's
        job to set.
    source_strength : float
        S0 [neutrons/sec], the external source strength used throughout
        the depletion campaign for this case.
    default_n_particles, default_n_generations : int
        Defaults for the particle-per-generation and active-generation
        counts, used unless an experiment overrides them.
    category : str
        ``"nu_multiplier"`` for the sweepable analysis family used for
        the scaling/loop-gain experiments (E1, E3-E6), or ``"physical"``
        for a realism-control family swept by an actual geometric control
        variable (used for E7-E8 and as a check on E1's assumption that
        the nu-multiplier family leaves the trajectory invariant).

    Note: there is deliberately no ``initial_vec_builder`` field.
    ``CoupledOperator.initial_condition()`` (inherited unchanged by
    ``SCMCoupledOperator``) already derives the correct initial nuclide
    vector from ``model.materials``' burnable materials, in the ordering
    the depletion chain expects, and also performs the
    ``openmc.lib.init()`` call. Hand-building this vector separately per
    case would only add a class of ordering bugs with no benefit; see
    ``experiments/_expcommon.py::build_operator_and_model``.
    """
    name: str
    model_builder: Callable[[float], object]
    source_strength: float
    default_n_particles: int = 8000
    default_n_generations: int = 200
    category: str = "nu_multiplier"
    critical_model_builder: Optional[Callable[[float], object]] = None
    """See docstring note below: an optional ordinary eigenvalue-mode
    counterpart of this case (same geometry/composition, no external
    source), used only by the E8 conventional-depletion comparison
    (Sec. 7.10). Left as None by default; E8 skips that leg with a clear
    message rather than silently omitting it if this is unset."""


CASE_REGISTRY: Dict[str, BenchmarkCase] = {}


def register_case(case: BenchmarkCase):
    CASE_REGISTRY[case.name] = case
    return case


def _unbuilt(name: str):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f"model_builder for case {name!r} has not been wired in. "
            "Point this at the existing project's model-building code "
            "(mc_runners.py / mg_benchmarks.py) for the corresponding "
            "physical family."
        )
    return _raise


# Placeholders for the four families of Table 3 (companion paper): wire
# in the real model builders before use.
#
# heu_sphere_nu's source_strength (3.45e14 n/s, not the earlier 1.0e7)
# is chosen, not arbitrary: at the highest-M k_target (k=0.995, M=200)
# it gives a per-depletion-step burnup of at most 0.1 MWd/kgHM over a
# 1-day calendar step -- a standard, conservative depletion step size
# (cf. Serpent's own documented 0.1-1.0 MWd/kgU early-life step
# convention) -- with a 200-day total campaign (200 steps). The
# original 1.0e7 n/s gave a power of only ~26 uW at the same k_target
# (~1.4e7x below the ~40 kW/kgHM power density used in standard
# depletion-methodology test cases, e.g. Isotalo, OSTI 1362197), making
# ANY power-normalized burnup target correspond to an astronomically
# (and physically meaningless) long calendar duration regardless of
# what MWd/kg value was chosen. See run_e1_master_table.py /
# run_e9_fixed_duration.py's submit scripts for the exact derivation.
for _name, _S0 in [
    ("heu_plate_stack_nu", 1.0e8),
    ("pwr_17x17_nu", 1.0e9),
    ("sfr_metal_fuel_nu", 1.0e9),
    ("heu_sphere_nu", 3.45e14),
]:
    register_case(BenchmarkCase(
        name=_name,
        model_builder=_unbuilt(_name),
        source_strength=_S0,
        category="nu_multiplier",
    ))


# --------------------------------------------------------------------------
# nu-multiplier application (fork interface point #2)
# --------------------------------------------------------------------------

def apply_nu_multiplier(model, alpha: float, fissile_nuclides="auto",
                         output_dir=None, source_xml=None, tag=None):
    """Scale the fission neutron yield (nu) by ``alpha``, leaving all
    cross sections, scattering physics, and fission-product yields fixed
    (companion paper Remark 2; theory note Sec. 7.1).

    Implemented as a data-level rewrite of the HDF5 nuclide files (see
    ``nu_multiplier.build_nu_scaled_library`` for the full pipeline and
    the OpenMC-source-grounded API notes): this requires no OpenMC C++
    changes, since the C++ engine reads whatever HDF5 files
    ``cross_sections.xml`` points it at, and the rewrite happens before
    that file is ever touched by transport.

    Mutates ``model.materials.cross_sections`` to point at the new,
    per-alpha library and returns ``model``. ``output_dir`` defaults to
    a ``nu_scaled_libraries/`` directory next to the cache root if not
    given explicitly; pass one explicitly (or a distinct ``tag``) when
    calling this repeatedly at different alpha, e.g. from
    :func:`calibrate_alpha_for_keff`, so scaled libraries for different
    alpha don't collide.
    """
    from pathlib import Path
    from .nu_multiplier import build_nu_scaled_library

    if output_dir is None:
        output_dir = Path("nu_scaled_libraries")

    new_xml = build_nu_scaled_library(
        model, fissile_nuclides, alpha, output_dir,
        source_xml=source_xml, tag=tag)
    model.materials.cross_sections = str(new_xml)
    return model


def calibrate_alpha_for_keff(model_builder: Callable[[float], object],
                              keff_target: float,
                              run_eigenvalue_fn: Callable[[object], float],
                              bracket=(0.7, 1.3), tol: float = 1e-4,
                              max_iter: int = 40,
                              max_expansions: int = 12,
                              expansion_factor: float = 1.3) -> float:
    """Bisect for the nu-multiplier alpha that hits a target eigenvalue.

    ``run_eigenvalue_fn`` is injected rather than implemented here so that
    this module does not duplicate the existing project's eigenvalue-mode
    runner (``kcalib.py``); pass a callable that builds and runs an
    ordinary k-eigenvalue calculation on the model returned by
    ``model_builder(alpha)`` and returns the estimated k_eff.

    ``bracket`` is deliberately narrow (0.7-1.3, i.e. at most a 30% swing
    in the fission neutron yield) and expanded *geometrically* only as
    far as actually needed to contain ``keff_target``, rather than
    starting from a wide fixed bracket. A wide bracket's upper end can
    land on an alpha that is wildly supercritical for the geometry in
    question -- multiplying nu by, say, 5x pushes almost any subcritical
    assembly far enough past k=1 that eigenvalue power iteration's
    fission bank overflows (the "shared fission bank is full... Results
    may be non-deterministic" warning), which is at best a biased k_eff
    for that evaluation and, if it corrupts write access to a shared
    statepoint file, has been observed to abort the whole run. Expanding
    only as far as needed, one bracket-doubling-ish step at a time,
    keeps every evaluated alpha as close to keff_target as the search
    itself requires.
    """
    lo, hi = bracket
    k_lo = run_eigenvalue_fn(model_builder(lo))
    k_hi = run_eigenvalue_fn(model_builder(hi))

    expansions = 0
    while not (min(k_lo, k_hi) <= keff_target <= max(k_lo, k_hi)):
        expansions += 1
        if expansions > max_expansions:
            raise ValueError(
                f"keff_target={keff_target} not bracketed after "
                f"{max_expansions} expansions; last bracket "
                f"[{lo}, {hi}] -> [{k_lo}, {k_hi}]. Either keff_target is "
                "unreachable for this geometry by nu-scaling alone, or "
                "expansion_factor/max_expansions need adjusting.")
        if keff_target < min(k_lo, k_hi):
            lo = lo / expansion_factor
            k_lo = run_eigenvalue_fn(model_builder(lo))
        else:
            hi = hi * expansion_factor
            k_hi = run_eigenvalue_fn(model_builder(hi))

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        k_mid = run_eigenvalue_fn(model_builder(mid))
        if abs(k_mid - keff_target) < tol:
            return mid
        if (k_mid < keff_target) == (k_lo < keff_target):
            lo, k_lo = mid, k_mid
        else:
            hi, k_hi = mid, k_mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Wire in the real geometry (bench_models.py), replacing the placeholders
# registered above. Deferred import: bench_models.py imports this module
# back (lazily, inside its own functions) to avoid a circular import at
# module-load time; by this point CASE_REGISTRY already holds all four
# placeholder entries for register_all() to overwrite in place.
# --------------------------------------------------------------------------
from . import bench_models as _bench_models
_bench_models.register_all()
del _bench_models
