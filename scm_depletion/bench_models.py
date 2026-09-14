"""Model construction for the four nu-multiplier benchmark families
(companion paper Table 3: HEU plate stack, 17x17 PWR assembly, SFR
metal-fuel assembly, HEU sphere).

Geometry, materials, and sources are adapted directly from the parent
project's ``ce_families.py`` (``build_stack``, ``build_pwr``,
``build_sfr``, ``build_sphere``) -- tested code from the companion
paper's own convergence-theory study, not fabricated here. Three things
are changed relative to that file, all specific to depletion rather
than to a fixed-source convergence-rate study:

  1. Each family's geometric knob (active/fuelled height) is fixed at
     one nominal value rather than swept -- ``ce_families.py`` sweeps
     height to change k; this package sweeps k at fixed geometry via
     the nu-multiplier alpha instead (companion Remark 2), so that the
     transmutation physics does not change across the k sweep. The
     stack family's nominal height, x=15.0 cm, is ``ce_families.py``'s
     own published near-critical reference point (k_eff=0.99574,
     Forget 2026 Sec. IV.B); the PWR and SFR heights are plausible
     values within their brackets, NOT calibrated reference points --
     adjust ``_PWR_HEIGHT``/``_SFR_HEIGHT`` below if a different
     starting k is wanted.
  2. The fuel material(s) in each case are marked ``.depletable = True``
     with an explicit ``.volume`` set (neither is needed for a
     fixed-source convergence-rate run, both are required for
     depletion).
  3. ``settings.run_mode`` is left unset (default ``'eigenvalue'``),
     per the ``BenchmarkCase.model_builder`` contract in
     caseregistry.py -- the SCM depletion drivers set it to
     ``'subcritical multiplication'`` themselves after calling this.

Every material below follows ``ce_families.py``'s convention of using
one consistent percent type (all-weight or all-atom) per material --
mixing them on the same material raises "Cannot mix atom and weight
percents", which is what the SFR case in an earlier version of this
file did wrong (a bare ``add_element("Zr", ...)`` defaulting to atom
percent on a material otherwise built entirely in weight percent).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import openmc
import openmc.stats


# --------------------------------------------------------------------------
# nu-multiplier application
# --------------------------------------------------------------------------
# Imports build_nu_scaled_library directly from nu_multiplier.py rather
# than apply_nu_multiplier from caseregistry.py, since caseregistry.py
# imports this module (to wire the four cases into CASE_REGISTRY) --
# importing caseregistry.py back from here at module load time would be
# circular. nu_multiplier.py has no such dependency.

def _apply_alpha(model: "openmc.Model", alpha: float, fissile_nuclides,
                  case_name: str) -> "openmc.Model":
    from .nu_multiplier import build_nu_scaled_library
    output_dir = Path("nu_scaled_libraries") / case_name
    new_xml = build_nu_scaled_library(
        model, fissile_nuclides, alpha, output_dir,
        tag=f"alpha{alpha:.6f}")
    model.materials.cross_sections = str(new_xml)
    return model


def _default_settings(source, particles=8000, batches=200) -> "openmc.Settings":
    s = openmc.Settings()
    s.particles = particles
    s.batches = batches
    s.source = source
    # run_mode intentionally left at the Settings() default ('eigenvalue');
    # see BenchmarkCase.model_builder's docstring in caseregistry.py.
    return s


def watt_point(xyz) -> "openmc.IndependentSource":
    """Matches ce_families.py's watt_point exactly."""
    return openmc.IndependentSource(
        space=openmc.stats.Point(tuple(float(c) for c in xyz)),
        energy=openmc.stats.Watt(),
        angle=openmc.stats.Isotropic(),
    )


# ==========================================================================
# 1. HEU plate stack (fast spectrum) -- ce_families.py: build_stack
# ==========================================================================

_STACK_DENSITY_U = 19.1
_STACK_PLATE_HALF, _STACK_FUEL_HALF, _STACK_PLATE_T = 15.0, 7.5, 1.0
_STACK_HEIGHT = 15.0  # ce_families.py's own near-critical reference point


def _stack_heu(wt: float, mid: int, density=_STACK_DENSITY_U) -> "openmc.Material":
    m = openmc.Material(material_id=mid, name=f"HEU{wt:g}")
    m.add_nuclide("U235", wt, "wo")
    m.add_nuclide("U238", 100.0 - wt, "wo")
    m.set_density("g/cm3", density)
    return m


def _stack_du(mid: int, density=_STACK_DENSITY_U) -> "openmc.Material":
    m = openmc.Material(material_id=mid, name="DU")
    m.add_nuclide("U238", 1.0)
    m.set_density("g/cm3", density)
    # openmc.Material.add_nuclide auto-sets depletable=True for any
    # actinide (Z>=89, see material.py add_nuclide) -- U238 qualifies,
    # even though this material is a passive reflector/support layer in
    # both call sites (heu_plate_stack_model_builder's "base",
    # heu_sphere_model_builder's "refl"), never meant to be tracked by
    # the depletion solver. Left un-overridden, CoupledOperator demands
    # a .volume for it and raises "Volume not specified for depletable
    # material" the moment a depletion operator is built. Override
    # explicitly, after the auto-set, rather than working around it at
    # every call site.
    m.depletable = False
    return m


def heu_plate_stack_model_builder(alpha: float, x: float = _STACK_HEIGHT
                                   ) -> "openmc.Model":
    """HEU plate stack at fixed fuelled height ``x`` (default: the
    published near-critical reference height), with nu scaled by
    ``alpha``. Geometry is ce_families.py's ``build_stack`` verbatim.
    """
    x = float(max(0.05, x))
    n_full = int(np.floor(x - 1e-9))
    frac = x - n_full

    fuel, base = _stack_heu(88.0, 1), _stack_du(2)
    fuel.depletable = True
    fuel.volume = (2.0 * _STACK_FUEL_HALF) ** 2 * x
    materials = openmc.Materials([fuel, base])

    h_total = _STACK_PLATE_T + x
    xs = [openmc.XPlane(surface_id=1, x0=-_STACK_PLATE_HALF, boundary_type="vacuum"),
          openmc.XPlane(surface_id=2, x0=+_STACK_PLATE_HALF, boundary_type="vacuum")]
    ys = [openmc.YPlane(surface_id=3, y0=-_STACK_PLATE_HALF, boundary_type="vacuum"),
          openmc.YPlane(surface_id=4, y0=+_STACK_PLATE_HALF, boundary_type="vacuum")]
    z0 = openmc.ZPlane(surface_id=5, z0=0.0, boundary_type="vacuum")
    z1 = openmc.ZPlane(surface_id=6, z0=h_total, boundary_type="vacuum")
    outer = +xs[0] & -xs[1] & +ys[0] & -ys[1] & +z0 & -z1

    fx = [openmc.XPlane(surface_id=11, x0=-_STACK_FUEL_HALF),
          openmc.XPlane(surface_id=12, x0=+_STACK_FUEL_HALF)]
    fy = [openmc.YPlane(surface_id=13, y0=-_STACK_FUEL_HALF),
          openmc.YPlane(surface_id=14, y0=+_STACK_FUEL_HALF)]
    insert = +fx[0] & -fx[1] & +fy[0] & -fy[1]

    cells = []
    zb = openmc.ZPlane(surface_id=20, z0=_STACK_PLATE_T)
    cells.append(openmc.Cell(cell_id=1, fill=base, region=outer & -zb))

    prev, sid, cid = zb, 21, 100
    edges = [_STACK_PLATE_T + i for i in range(1, n_full + 1)]
    if frac > 1e-6:
        edges.append(_STACK_PLATE_T + x)
    for ztop in edges:
        top = openmc.ZPlane(surface_id=sid, z0=ztop)
        layer = outer & +prev & -top
        cells.append(openmc.Cell(cell_id=cid, fill=fuel, region=layer & insert))
        cells.append(openmc.Cell(cell_id=cid + 1, fill=base, region=layer & ~insert))
        prev, sid, cid = top, sid + 1, cid + 2

    geometry = openmc.Geometry(cells)
    source = watt_point((0.0, 0.0, 0.9 * _STACK_PLATE_T))
    settings = _default_settings(source)

    model = openmc.model.Model(geometry, materials, settings, openmc.Tallies([]))
    model = _apply_alpha(model, alpha, fissile_nuclides=["U235", "U238"],
                          case_name="heu_plate_stack_nu")
    return model


# ==========================================================================
# 2. HEU sphere (fast spectrum) -- ce_families.py: build_sphere
# ==========================================================================

_SPHERE_R_FUEL, _SPHERE_R_REFL = 4.8, 12.2


def heu_sphere_model_builder(alpha: float, spectrum: str = "watt",
                              source_r: float = 0.0) -> "openmc.Model":
    """HEU sphere with a DU reflector, fixed geometry (ce_families.py:
    build_sphere does not sweep -- this family's k is set entirely by
    alpha here).
    """
    fuel, refl = _stack_heu(93.0, 1), _stack_du(2)
    fuel.depletable = True
    fuel.volume = (4.0 / 3.0) * math.pi * _SPHERE_R_FUEL**3
    materials = openmc.Materials([fuel, refl])

    s_in = openmc.Sphere(surface_id=1, r=_SPHERE_R_FUEL)
    s_out = openmc.Sphere(surface_id=2, r=_SPHERE_R_REFL, boundary_type="vacuum")
    geometry = openmc.Geometry([
        openmc.Cell(cell_id=1, fill=fuel, region=-s_in),
        openmc.Cell(cell_id=2, fill=refl, region=+s_in & -s_out),
    ])

    energy = {"thermal": openmc.stats.Maxwell(0.0253),
              "dt": openmc.stats.Discrete([14.1e6], [1.0]),
              "watt": openmc.stats.Watt()}[spectrum.lower()]
    source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, float(source_r))),
        energy=energy, angle=openmc.stats.Isotropic())
    settings = _default_settings(source)

    model = openmc.model.Model(geometry, materials, settings, openmc.Tallies([]))
    model = _apply_alpha(model, alpha, fissile_nuclides=["U235", "U238"],
                          case_name="heu_sphere_nu")
    return model


# ==========================================================================
# 3. 17x17 PWR assembly (thermal spectrum) -- ce_families.py: build_pwr
# ==========================================================================

_PWR_PITCH = 1.26
_PWR_N = 17
_PWR_BORON_PPM = 0.0
_PWR_HEIGHT = 60.0  # NOT a calibrated reference point -- see module docstring
_PWR_GT = [(2, 5), (2, 8), (2, 11), (3, 3), (3, 13),
           (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
           (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
           (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
           (13, 3), (13, 13), (14, 5), (14, 8), (14, 11)]


def _pwr_materials(boron_ppm=_PWR_BORON_PPM):
    fuel = openmc.Material(material_id=1, name="UO2 3.1 w/o")
    fuel.add_element("U", 1.0, enrichment=3.1)
    fuel.add_element("O", 2.0)
    fuel.set_density("g/cm3", 10.29)

    clad = openmc.Material(material_id=2, name="Zircaloy-4")
    clad.add_element("Zr", 98.23, "wo")
    clad.add_element("Sn", 1.45, "wo")
    clad.add_element("Fe", 0.21, "wo")
    clad.add_element("Cr", 0.11, "wo")
    clad.set_density("g/cm3", 6.55)

    gap = openmc.Material(material_id=3, name="He gap")
    gap.add_element("He", 1.0)
    gap.set_density("g/cm3", 0.001598)

    # all-weight-percent so boron can be added without mixing percent types
    water = openmc.Material(material_id=4, name="borated water")
    water.add_element("H", 11.1894, "wo")
    water.add_element("O", 88.8106, "wo")
    if boron_ppm > 0:
        water.add_element("B", boron_ppm * 1.0e-4, "wo")
    water.set_density("g/cm3", 0.7405)
    water.add_s_alpha_beta("c_H_in_H2O")
    return openmc.Materials([fuel, clad, gap, water]), (fuel, clad, gap, water)


def pwr_17x17_model_builder(alpha: float, x: float = _PWR_HEIGHT,
                             boron_ppm: float = _PWR_BORON_PPM) -> "openmc.Model":
    """17x17 PWR assembly at fixed active height ``x``, radially
    reflective, nu scaled by ``alpha``. Geometry is ce_families.py's
    ``build_pwr`` verbatim.
    """
    h = float(max(2.0, x))
    materials, (fuel, clad, gap, water) = _pwr_materials(boron_ppm)
    fuel.depletable = True
    r_f_for_volume = 0.39218
    n_fuel_pins = _PWR_N**2 - len(_PWR_GT)
    fuel.volume = n_fuel_pins * math.pi * r_f_for_volume**2 * h

    r_f = openmc.ZCylinder(surface_id=1, r=0.39218)
    r_g = openmc.ZCylinder(surface_id=2, r=0.40005)
    r_c = openmc.ZCylinder(surface_id=3, r=0.45720)
    pin = openmc.Universe(cells=[
        openmc.Cell(fill=fuel, region=-r_f),
        openmc.Cell(fill=gap, region=+r_f & -r_g),
        openmc.Cell(fill=clad, region=+r_g & -r_c),
        openmc.Cell(fill=water, region=+r_c)])

    g_i = openmc.ZCylinder(surface_id=4, r=0.56230)
    g_o = openmc.ZCylinder(surface_id=5, r=0.60230)
    tube = openmc.Universe(cells=[
        openmc.Cell(fill=water, region=-g_i),
        openmc.Cell(fill=clad, region=+g_i & -g_o),
        openmc.Cell(fill=water, region=+g_o)])

    mod = openmc.Universe(cells=[openmc.Cell(fill=water)])

    gt_set = set(_PWR_GT)
    grid = [[tube if (i, j) in gt_set else pin for j in range(_PWR_N)]
            for i in range(_PWR_N)]

    lat = openmc.RectLattice()
    half = _PWR_N * _PWR_PITCH / 2.0
    lat.lower_left = (-half, -half)
    lat.pitch = (_PWR_PITCH, _PWR_PITCH)
    lat.universes = grid
    lat.outer = mod

    px0 = openmc.XPlane(surface_id=11, x0=-half, boundary_type="reflective")
    px1 = openmc.XPlane(surface_id=12, x0=+half, boundary_type="reflective")
    py0 = openmc.YPlane(surface_id=13, y0=-half, boundary_type="reflective")
    py1 = openmc.YPlane(surface_id=14, y0=+half, boundary_type="reflective")
    pz0 = openmc.ZPlane(surface_id=15, z0=0.0, boundary_type="vacuum")
    pz1 = openmc.ZPlane(surface_id=16, z0=h, boundary_type="vacuum")
    region = +px0 & -px1 & +py0 & -py1 & +pz0 & -pz1

    geometry = openmc.Geometry([openmc.Cell(fill=lat, region=region)])
    source = watt_point((0.0, 0.0, 0.05 * h))
    settings = _default_settings(source)

    model = openmc.model.Model(geometry, materials, settings, openmc.Tallies([]))
    model = _apply_alpha(model, alpha, fissile_nuclides=["U235", "U238"],
                          case_name="pwr_17x17_nu")
    return model


# ==========================================================================
# 4. SFR metal-fuel assembly (fast spectrum) -- ce_families.py: build_sfr
# ==========================================================================

_SFR_PITCH = 0.95
_SFR_RINGS = 9
_SFR_SMEAR = 0.75
_SFR_HEIGHT = 100.0  # NOT a calibrated reference point -- see module docstring


def _sfr_materials(smear=_SFR_SMEAR):
    fuel = openmc.Material(material_id=1, name="U-20Pu-10Zr")
    fuel.add_element("U", 70.0, "wo", enrichment=17.0)
    fuel.add_nuclide("Pu239", 20.0 * 0.62, "wo")
    fuel.add_nuclide("Pu240", 20.0 * 0.24, "wo")
    fuel.add_nuclide("Pu241", 20.0 * 0.09, "wo")
    fuel.add_nuclide("Pu242", 20.0 * 0.05, "wo")
    fuel.add_element("Zr", 10.0, "wo")  # explicit "wo": the earlier bug was
                                         # this call defaulting to atom
                                         # percent on an otherwise
                                         # all-weight-percent material.
    fuel.set_density("g/cm3", 15.8 * float(smear))

    clad = openmc.Material(material_id=2, name="HT9")
    for el, wt in (("Fe", 85.0), ("Cr", 12.0), ("Mo", 1.0), ("Ni", 0.5),
                   ("Mn", 0.5), ("W", 0.5), ("V", 0.3), ("Si", 0.2)):
        clad.add_element(el, wt, "wo")
    clad.set_density("g/cm3", 7.8)

    na = openmc.Material(material_id=3, name="sodium")
    na.add_element("Na", 1.0)
    na.set_density("g/cm3", 0.85)
    return openmc.Materials([fuel, clad, na]), (fuel, clad, na)


def _hex_region(edge, sid, boundary):
    """HexagonalPrism across OpenMC versions (class vs. helper function) --
    verbatim from ce_families.py."""
    try:
        hexs = openmc.model.HexagonalPrism(edge_length=edge, orientation="y",
                                            boundary_type=boundary)
        return -hexs
    except AttributeError:
        return openmc.model.hexagonal_prism(edge_length=edge, orientation="y",
                                             boundary_type=boundary)


def sfr_metal_fuel_model_builder(alpha: float, x: float = _SFR_HEIGHT,
                                  smear: float = _SFR_SMEAR) -> "openmc.Model":
    """SFR-like metal-fuel assembly at fixed active height ``x``,
    radially reflective (hexagonal), nu scaled by ``alpha``. Geometry is
    ce_families.py's ``build_sfr`` verbatim.
    """
    h = float(max(2.0, x))
    materials, (fuel, clad, na) = _sfr_materials(smear)
    fuel.depletable = True
    n_rings_total = 1 + 6 * sum(range(1, _SFR_RINGS))  # 1 + 6*(1+...+(N-1))
    fuel.volume = n_rings_total * math.pi * 0.300**2 * h

    r_f = openmc.ZCylinder(surface_id=1, r=0.300)
    r_ci = openmc.ZCylinder(surface_id=2, r=0.320)
    r_co = openmc.ZCylinder(surface_id=3, r=0.370)
    pin = openmc.Universe(universe_id=1, cells=[
        openmc.Cell(cell_id=1, fill=fuel, region=-r_f),
        openmc.Cell(cell_id=2, fill=na, region=+r_f & -r_ci),
        openmc.Cell(cell_id=3, fill=clad, region=+r_ci & -r_co),
        openmc.Cell(cell_id=4, fill=na, region=+r_co)])
    cool = openmc.Universe(universe_id=2,
                            cells=[openmc.Cell(cell_id=5, fill=na)])

    # lattice_id must not collide with a universe_id (see ce_families.py's
    # own comment on this): lattice_id=100 is deliberately far from
    # universe_id in {1, 2}.
    lat = openmc.HexLattice(lattice_id=100)
    lat.center = (0.0, 0.0)
    lat.pitch = (_SFR_PITCH,)
    lat.orientation = "y"
    lat.outer = cool
    lat.universes = [[pin] * (6 * (r - 1) if r > 1 else 1)
                      for r in range(_SFR_RINGS, 0, -1)]

    flat_to_flat = (2 * _SFR_RINGS - 1) * _SFR_PITCH
    edge = flat_to_flat / np.sqrt(3.0) + 0.5 * _SFR_PITCH
    z0 = openmc.ZPlane(surface_id=11, z0=0.0, boundary_type="vacuum")
    z1 = openmc.ZPlane(surface_id=12, z0=h, boundary_type="vacuum")
    region = _hex_region(edge, 10, "reflective") & +z0 & -z1
    geometry = openmc.Geometry([openmc.Cell(cell_id=20, fill=lat, region=region)])

    source = watt_point((0.0, 0.0, 0.05 * h))
    settings = _default_settings(source)

    model = openmc.model.Model(geometry, materials, settings, openmc.Tallies([]))
    model = _apply_alpha(
        model, alpha, fissile_nuclides=["U235", "U238", "Pu239", "Pu240",
                                        "Pu241", "Pu242"],
        case_name="sfr_metal_fuel_nu")
    return model


# --------------------------------------------------------------------------
# Wire the four builders into CASE_REGISTRY
# --------------------------------------------------------------------------

def register_all():
    """Call once (from caseregistry.py, at import time) to replace the
    four placeholder model_builders with the real ones above."""
    from . import caseregistry

    builders = {
        "heu_sphere_nu": heu_sphere_model_builder,
        "pwr_17x17_nu": pwr_17x17_model_builder,
        "sfr_metal_fuel_nu": sfr_metal_fuel_model_builder,
        "heu_plate_stack_nu": heu_plate_stack_model_builder,
    }
    for name, builder in builders.items():
        if name in caseregistry.CASE_REGISTRY:
            caseregistry.CASE_REGISTRY[name].model_builder = builder
