"""Depletion for the normalized subcritical-multiplication (SCM) method.

This package implements the two integration modes analyzed in the
companion paper "Depletion in the Normalized Subcritical Multiplication
Method":

- ``ExposureIntegrator`` (Method C): steps in the exposure coordinate
  tau, with d tau = S0 M dt.  The depletion ordinate is multiplication-free;
  the physical clock t(tau) is accumulated as a side quantity.
- ``CalendarIntegrator`` (Method A): steps in physical time t directly,
  rescaling the normalized reaction rates by S0*M_hat at every step.

Both integrators are built on top of ``openmc.deplete.CoupledOperator``
and the existing CRAM solvers in ``openmc.deplete.cram``; no change to the
OpenMC C++ core is required beyond the ``subcritical_k_factors()``
C-API extension documented alongside this package (see the paper's
Implementation section and openmc_capi_patch/ in the repository root).

See ``scm_transport.py`` for the single integration point that must be
adapted to the exact C-API of a given SCM-enabled OpenMC build.
"""

from .scm_transport import (
    SCMOperatorResult, SCMCoupledOperator, BranchingFixedSourceOperator, KFactors,
    ForcedKSequence, FrozenComposition, KOverride, CompositionPerturbation,
    VariableParticleCount, isolated_transport_session,
)
from .matrix_utils import split_full_matrix, exposure_matrix, calendar_matrix, branching_matrix
from .integrators import (ExposureIntegrator, CalendarIntegrator,
                           BranchingCalendarIntegrator, RegulatedBeamIntegrator, StepStats)
from .trajectory import SCMTrajectory
from . import caseregistry
from . import bench_models
from . import common
from .nu_multiplier import build_nu_scaled_library, scale_nu

__all__ = [
    "SCMOperatorResult", "SCMCoupledOperator", "BranchingFixedSourceOperator", "KFactors",
    "ForcedKSequence", "FrozenComposition", "KOverride", "CompositionPerturbation",
    "VariableParticleCount", "isolated_transport_session",
    "split_full_matrix", "exposure_matrix", "calendar_matrix", "branching_matrix",
    "ExposureIntegrator", "CalendarIntegrator", "BranchingCalendarIntegrator",
    "RegulatedBeamIntegrator", "StepStats",
    "SCMTrajectory",
    "caseregistry", "bench_models", "common",
    "build_nu_scaled_library", "scale_nu",
]
