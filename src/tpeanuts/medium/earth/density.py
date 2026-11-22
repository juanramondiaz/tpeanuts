#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""The PREM electron/neutron density table, and linear-in-radius interpolation.

Reads ``tpeanuts.data.earth``'s ``density/prem_density.csv`` (see that
package's own ``README.md`` for provenance and the full column list --
this module only reads ``radius_fraction``/``electron_density_mol_cm3``/
``neutron_density_mol_cm3``, ignoring the many other geophysical columns
there). ``radius_fraction`` is ``r / R_E`` (dimensionless, matches
``tpeanuts.util.constant.R_E`` exactly -- the table's own surface row is
at ``radius_km = 6371.0``).

The table is not strictly increasing in ``radius_fraction``: PREM has
several genuine physical discontinuities (core-mantle boundary,
crust-ocean boundary, ...) where two consecutive rows share the *same*
``radius_fraction`` with different densities -- ``interpolate_prem_density``
resolves an exact query at such a radius deterministically (picks the
inner side), documented in its own docstring, rather than dividing by an
exact zero.

Module contents:
    PremTable
        Frozen dataclass: the three loaded columns, as tensors.
    load_prem_table(...)
        Read ``tpeanuts.data.earth``'s PREM density table into a
        ``PremTable``.
    prem_discontinuity_radii(...)
        The radii (``r / R_E``) of the table's genuine density jumps.
    interpolate_prem_density(...)
        Piecewise-linear-in-radius interpolation of electron/neutron
        density at arbitrary query points.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import torch

from tpeanuts.data import package_data_dir
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor

_PREM_CSV_PATH = package_data_dir() / "earth" / "density" / "prem_density.csv"


@dataclass(frozen=True)
class PremTable:
    """The PREM density table, loaded as sorted 1-D tensors.

    Attributes:
        radius_fraction: ``r / R_E``, shape ``(n_rows,)``, sorted
            ascending (non-strictly -- see this module's own docstring).
        n_e_mol_cm3: Electron density at each row, mol/cm^3, shape
            ``(n_rows,)``.
        n_n_mol_cm3: Neutron density at each row, mol/cm^3, shape
            ``(n_rows,)``.
    """

    radius_fraction: torch.Tensor
    n_e_mol_cm3: torch.Tensor
    n_n_mol_cm3: torch.Tensor


@lru_cache(maxsize=None)
def load_prem_table(context: RuntimeContext) -> PremTable:
    """Read ``data/prem_density.csv`` into a ``PremTable`` on the given context.

    Cached per ``context`` (``RuntimeContext`` is a hashable, frozen
    (device, dtype) pair -- confirmed directly): the file is fixed,
    read-only data, so re-parsing the CSV on every ``EarthMedium.profile()``
    call is pure, avoidable cost, not a correctness concern -- nothing
    anywhere in this project mutates a ``PremTable``'s own tensors in
    place (only ever indexed/read, e.g. by
    ``interpolate_prem_density``). This matters most for a caller that
    builds many ``EarthMedium`` profiles in a Python loop (e.g. a buried
    -detector, multi-angle composition via
    ``tpeanuts.pipeline.atmospheric_neutrino``, which requires scalar ``eta_rad``
    at ``depth_m>0`` and so loops), where the CSV would otherwise be
    reread on every single iteration.

    Args:
        context: RuntimeContext the returned tensors are built on.

    Returns:
        PremTable with rows in the CSV's own (radius-ascending) order.
    """
    radius_fraction = []
    n_e_mol_cm3 = []
    n_n_mol_cm3 = []
    with open(_PREM_CSV_PATH, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            radius_fraction.append(float(row["radius_fraction"]))
            n_e_mol_cm3.append(float(row["electron_density_mol_cm3"]))
            n_n_mol_cm3.append(float(row["neutron_density_mol_cm3"]))

    return PremTable(
        radius_fraction=as_tensor(radius_fraction, device=context.device, dtype=context.dtype),
        n_e_mol_cm3=as_tensor(n_e_mol_cm3, device=context.device, dtype=context.dtype),
        n_n_mol_cm3=as_tensor(n_n_mol_cm3, device=context.device, dtype=context.dtype),
    )


@lru_cache(maxsize=None)
def prem_discontinuity_radii(context: RuntimeContext) -> torch.Tensor:
    """Radii ``r / R_E`` at which the PREM table jumps in density, ascending.

    A jump is a pair of consecutive rows that share the same ``radius_fraction`` and differ in electron density (see
    this module's own docstring). A chord segment that straddles one is integrated with a density that is wrong on
    one side of it, which is why ``tpeanuts.medium.earth.geometry.aligned_chord_segments`` cuts the chord there.

    Args:
        context: RuntimeContext the returned tensor is built on.

    Returns:
        1-D tensor with one radius per jump (eight in the shipped table).
    """
    table = load_prem_table(context)
    radius = table.radius_fraction
    jump = (radius[1:] == radius[:-1]) & (table.n_e_mol_cm3[1:] != table.n_e_mol_cm3[:-1])
    return radius[1:][jump].clone()


def interpolate_prem_density(radius_fraction: TensorLike, table: PremTable) -> Tuple[torch.Tensor, torch.Tensor]:
    """Piecewise-linear-in-radius interpolation of electron/neutron density.

    A query landing exactly on one of the table's genuine discontinuity
    radii (two consecutive rows sharing the same ``radius_fraction``, see
    this module's own docstring) deterministically returns the inner
    (smaller-radius) side's density rather than dividing by an exact zero
    gap -- physically, density is two-valued exactly at such a radius, so
    any deterministic choice is as valid as another; landing exactly on
    one via a continuous ``eta_rad`` sweep is a measure-zero event in
    practice regardless.

    Args:
        radius_fraction: Query points, ``r / R_E``, any shape, must be
            within ``[0, 1]`` (the table's own domain -- no extrapolation).
        table: PremTable to interpolate (see ``load_prem_table``).

    Returns:
        ``(n_e_mol_cm3, n_n_mol_cm3)``, both shaped like ``radius_fraction``,
        differentiable w.r.t. ``radius_fraction`` (the table itself is a
        fixed buffer).

    Raises:
        ValueError: If any query point falls outside ``[0, 1]``.
    """
    query = as_tensor(radius_fraction, device=table.radius_fraction.device, dtype=table.radius_fraction.dtype)
    if torch.any(query < 0) or torch.any(query > 1):
        raise ValueError(
            "radius_fraction must be within [0, 1] -- no extrapolation beyond the PREM table's own range."
        )

    n_rows = table.radius_fraction.shape[0]
    idx_right = searchsorted_contiguous(table.radius_fraction, query, right=False).clamp(1, n_rows - 1)
    idx_left = idx_right - 1

    x0 = table.radius_fraction[idx_left]
    x1 = table.radius_fraction[idx_right]
    frac = (query - x0) / (x1 - x0).clamp_min(torch.finfo(table.radius_fraction.dtype).eps)

    n_e0, n_e1 = table.n_e_mol_cm3[idx_left], table.n_e_mol_cm3[idx_right]
    n_n0, n_n1 = table.n_n_mol_cm3[idx_left], table.n_n_mol_cm3[idx_right]
    n_e_mol_cm3 = n_e0 + frac * (n_e1 - n_e0)
    n_n_mol_cm3 = n_n0 + frac * (n_n1 - n_n0)
    return n_e_mol_cm3, n_n_mol_cm3
