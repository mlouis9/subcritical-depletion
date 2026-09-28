"""Build a nu-rescaled cross-section library (companion paper Remark 2;
theory note Sec. 7.1, 8.0): scale the fission neutron yield of chosen
nuclides by a factor alpha, leaving cross sections, scattering physics,
and fission-product yields untouched, so that k sweeps while the
transmutation physics stays fixed.

This is a data-level rewrite of the HDF5 nuclide files, done once per
alpha before running any transport. It requires no OpenMC C++ changes:
the C++ engine reads whatever HDF5 files cross_sections.xml points it
at, and this module produces a new cross_sections.xml pointing the
scaled nuclides at freshly written HDF5 data (and everything else at
the original files, unchanged).

Every API used here is confirmed against the actual OpenMC source
(product.py, neutron.py, library.py, material.py) rather than assumed:

  - Product.yield_ is an openmc.data.Function1D, concretely either a
    Tabulated1D (rescale via its .y array) or a Polynomial (rescale via
    its .coef array -- valid because nu(E) is linear in the
    coefficients, so scaling every coefficient scales the polynomial's
    value at every energy by the same factor).
  - A fission reaction's emitted neutrons are Product objects in
    rx.products (filtered by product.particle == 'neutron'); a
    precomputed convenience copy may additionally sit in
    rx.derived_products[0] ("total_nu") and is scaled identically for
    consistency.
  - IncidentNeutron.from_hdf5 accepts either an h5py.Group directly, or
    a filename (in which case it silently uses only the *first*
    top-level group in the file) -- since a registered library file's
    'materials' list can contain more than one nuclide, this module
    always opens the file itself and passes the specific named group,
    never a bare filename, to avoid loading the wrong nuclide.
  - IncidentNeutron.export_to_hdf5(path, mode='a') appends a group named
    self.name, so every scaled nuclide for one alpha can be written into
    a single shared file.
  - DataLibrary.register_file / .export_to_xml / .get_by_material give
    the whole cross_sections.xml read/write/lookup surface needed; no
    hand-rolled XML.
  - Materials.cross_sections is the settable attribute a Model's library
    is pointed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

import h5py

from openmc.data import DataLibrary, IncidentNeutron, FISSION_MTS
from openmc.data.function import Tabulated1D, Polynomial

__all__ = ["build_nu_scaled_library", "scale_nu"]


def _scale_yield(product, alpha: float) -> None:
    y = product.yield_
    if isinstance(y, Tabulated1D):
        y.y = y.y * alpha
    elif isinstance(y, Polynomial):
        y.coef = y.coef * alpha
    else:
        raise NotImplementedError(
            f"Unhandled Function1D subtype {type(y)!r} for nu scaling; "
            "Product.yield_ is documented as Tabulated1D or Polynomial "
            "but this installation returned something else.")


def scale_nu(nuclide: IncidentNeutron, alpha: float) -> bool:
    """Scale every fission-MT neutron product's yield by alpha, in place.

    Returns True if at least one fission reaction was found and scaled
    (i.e. this nuclide is actually fissile in this library), False
    otherwise -- callers should treat False as "nothing to do" for
    'auto' detection, or as an error for an explicit request.
    """
    scaled_any = False
    for mt in FISSION_MTS:
        if mt not in nuclide.reactions:
            continue
        rx = nuclide.reactions[mt]
        for product in rx.products:
            if product.particle == "neutron":
                _scale_yield(product, alpha)
                scaled_any = True
        for product in rx.derived_products:
            if product.particle == "neutron":
                _scale_yield(product, alpha)
    return scaled_any


def _load_named_nuclide(library: DataLibrary, name: str) -> IncidentNeutron:
    entry = library.get_by_material(name, data_type="neutron")
    if entry is None:
        raise KeyError(f"No neutron data registered for {name!r} in this library.")
    with h5py.File(entry["path"], "r") as h5file:
        return IncidentNeutron.from_hdf5(h5file[name])


def build_nu_scaled_library(
    model,
    fissile_nuclides: Union[Iterable[str], str],
    alpha: float,
    output_dir: Union[str, Path],
    source_xml: Optional[Union[str, Path]] = None,
    tag: Optional[str] = None,
) -> Path:
    """Write a new cross_sections.xml with the given nuclides' nu scaled
    by alpha, and return its path.

    Parameters
    ----------
    model : openmc.Model
        Only used to resolve nuclide names when fissile_nuclides == 'auto'
        (every nuclide referenced by model.materials that has at least
        one fission MT is scaled). Not otherwise mutated by this
        function -- setting model.materials.cross_sections to the
        returned path is left to the caller.
    fissile_nuclides : iterable of str, or 'auto'
        Explicit nuclide names to scale (e.g. ['U235', 'U238']) -- cheap
        and exact, and the right choice for repeated calls, e.g. inside
        an alpha-calibration bisection. 'auto' loads every candidate
        nuclide once to check for a fission MT, which is more convenient
        but redoes that check on every call.
    alpha : float
        Multiplicative factor applied to nu. alpha == 1.0 still produces
        a full round-tripped copy (useful as a data-pipeline sanity
        check, comparing transport results against the untouched
        library).
    output_dir : path-like
        Directory for the new HDF5 file and cross_sections.xml. Use a
        distinct output_dir or tag per alpha; this function does not
        namespace by alpha on your behalf beyond what `tag` provides.
    source_xml : path-like, optional
        Existing cross_sections.xml to start from. Defaults to
        openmc.config['cross_sections'].
    tag : str, optional
        Included in output filenames; defaults to a formatted alpha
        value.

    Returns
    -------
    Path to the new cross_sections.xml. Assign it via
    ``model.materials.cross_sections = str(returned_path)`` (or pass
    ``cross_sections=`` to ``openmc.Model(...)``) before running.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"alpha{alpha:.6f}"

    source_library = DataLibrary.from_xml(source_xml)

    if fissile_nuclides == "auto":
        candidate_names = set()
        for material in model.materials:
            candidate_names.update(material.get_nuclides())
        resolved: List[str] = []
        for name in sorted(candidate_names):
            entry = source_library.get_by_material(name)
            if entry is None:
                continue
            nuclide = _load_named_nuclide(source_library, name)
            if any(mt in nuclide.reactions for mt in FISSION_MTS):
                resolved.append(name)
        fissile_nuclides = resolved
    else:
        fissile_nuclides = list(fissile_nuclides)

    if not fissile_nuclides:
        raise ValueError(
            "No fissile nuclides found (or none given) to scale nu for; "
            "either the model has no fissile material or 'auto' detection "
            "found nothing -- pass an explicit list to be sure.")

    scaled_path = output_dir / f"nu_scaled_{tag}.h5"
    if scaled_path.exists():
        scaled_path.unlink()  # export_to_hdf5 defaults to append mode;
                               # start clean so a rerun with a different
                               # fissile_nuclides set can't leave stale
                               # groups behind.

    for name in fissile_nuclides:
        nuclide = _load_named_nuclide(source_library, name)
        if not scale_nu(nuclide, alpha):
            raise ValueError(
                f"{name!r} was requested for nu scaling but has no fission "
                f"MTs (checked {sorted(FISSION_MTS)}); remove it from "
                "fissile_nuclides or check the nuclide name.")
        nuclide.export_to_hdf5(scaled_path, mode="a")

    new_library = DataLibrary()
    new_library.register_file(scaled_path)  # re-derives materials/type
                                             # from the file itself

    scaled_set = set(fissile_nuclides)
    for entry in source_library:
        if entry["type"] == "depletion_chain":
            new_library.append(entry)
            continue
        remaining = [m for m in entry["materials"] if m not in scaled_set]
        if remaining:
            new_entry = dict(entry)
            new_entry["materials"] = remaining
            new_library.append(new_entry)

    new_xml_path = output_dir / f"cross_sections_{tag}.xml"
    new_library.export_to_xml(new_xml_path)
    return new_xml_path
