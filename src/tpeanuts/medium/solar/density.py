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

"""The solar electron/neutron density table, and linear-in-radius interpolation.

Reads ``tpeanuts.data.solar``'s ``{provider}/density/*.csv`` (see that
package's own ``README.md`` for provenance and which providers carry a
neutron-density column directly). ``radius`` is ``r / R_sun``
(dimensionless, in ``[0, 1]``) -- unlike PREM's ``radius_fraction``, the
solar density grids ported so far are strictly increasing with no
repeated-radius discontinuities, so ``interpolate_solar_density`` needs
none of ``tpeanuts.medium.earth.density.interpolate_prem_density``'s
discontinuity handling.

Three independent providers (matching ``tpeanuts.source.solar.io``'s own
split): the table path is resolved per provider rather than fixed to a
single file, unlike ``tpeanuts.medium.earth.density`` (Earth has exactly
one PREM table). ``load_solar_density_table`` is cached per
``(path, context)`` (``RuntimeContext`` is hashable -- confirmed
directly, same reasoning as ``tpeanuts.medium.earth.density.
load_prem_table``'s own cache) behind a frozen ``SolarDensityTable``, not
a plain dict: a ``functools.lru_cache`` returns the *same* object on
every call, and a mutable dict returned that way would be a real risk of
a caller corrupting the shared cache entry.

Module contents:
    SolarDensityTable
        Frozen dataclass: radius grid plus electron/neutron density,
        as tensors (``n_n_mol_cm3`` is None for a provider whose table
        has no neutron-density column -- see this module's own docstring
        and ``tpeanuts.data.solar``'s ``README.md``).
    solar_density_path(...)
        Return the canonical density-table path for one provider.
    load_solar_density_table(...)
        Read a provider's (or an explicit path's) density CSV into a
        ``SolarDensityTable``.
    interpolate_solar_density(...)
        Piecewise-linear-in-radius interpolation of electron/neutron
        density at arbitrary query points.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.data import package_data_dir
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor

Tensor = torch.Tensor

_DATA_DIR = package_data_dir() / "solar"

_DENSITY_FILES: dict[str, str] = {
    "bahcall": "bahcall/density/bp2000_density_electron_neutron_monotonic.csv",
    "zenodo": "zenodo/density/density_SF3_AGSS09.csv",
    # v1_0's own medium/solar/io.py still points at the plain
    # "density_b16_agss09.csv", but that file only tabulates radius up
    # to 0.5 R_sun (confirmed directly) -- it cannot satisfy
    # SolarMedium's own "must reach the solar surface"
    # invariant. "..._extended.csv" (present in the same v1_0 folder,
    # reaching radius=1.0) is used here instead; user-confirmed.
    "legacy": "legacy/density/density_b16_agss09_extended.csv",
}


@dataclass(frozen=True)
class SolarDensityTable:
    """The solar density table, loaded as sorted 1-D tensors.

    Attributes:
        radius: ``r / R_sun``, shape ``(n_rows,)``, strictly increasing.
        n_e_mol_cm3: Electron density at each row, mol/cm^3, shape
            ``(n_rows,)``.
        n_n_mol_cm3: Neutron density at each row, mol/cm^3, shape
            ``(n_rows,)``, or None if the source table has no
            ``neutron_density_mol_cm3`` column.
    """

    radius: Tensor
    n_e_mol_cm3: Tensor
    n_n_mol_cm3: Optional[Tensor]


def solar_density_path(provider: str) -> Path:
    """Return the canonical density-table path for one solar provider."""
    try:
        relative = _DENSITY_FILES[provider]
    except KeyError as exc:
        choices = ", ".join(sorted(_DENSITY_FILES))
        raise ValueError(f"Unknown solar density provider {provider!r}; available providers: {choices}") from exc
    return _DATA_DIR / relative


@lru_cache(maxsize=None)
def _load_solar_density_table_cached(path: Path, context: RuntimeContext) -> SolarDensityTable:
    radius = []
    n_e_mol_cm3 = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        has_neutron = "neutron_density_mol_cm3" in (reader.fieldnames or ())
        n_n_mol_cm3: list[float] = []
        for row in reader:
            radius.append(float(row["radius"]))
            n_e_mol_cm3.append(float(row["electron_density_mol_cm3"]))
            if has_neutron:
                n_n_mol_cm3.append(float(row["neutron_density_mol_cm3"]))

    return SolarDensityTable(
        radius=as_tensor(radius, device=context.device, dtype=context.dtype),
        n_e_mol_cm3=as_tensor(n_e_mol_cm3, device=context.device, dtype=context.dtype),
        n_n_mol_cm3=as_tensor(n_n_mol_cm3, device=context.device, dtype=context.dtype) if has_neutron else None,
    )


def load_solar_density_table(
    path: Optional[Union[str, Path]] = None,
    *,
    provider: Optional[str] = None,
    context: RuntimeContext,
) -> SolarDensityTable:
    """Load a provider's (or an explicit path's) solar density table.

    Args:
        path: Optional override path to the density CSV. None loads the
            selected provider's (or the configured default) table.
        provider: Optional canonical provider name, used to resolve
            ``path`` when it is None.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        SolarDensityTable with rows in the CSV's own (radius-ascending)
        order, cached per ``(path, context)`` -- see this module's own
        docstring.

    Raises:
        ValueError: If ``provider`` is not a recognised solar provider.
    """
    if path is None:
        provider = provider or default.SOLAR_PROVIDER
        path = solar_density_path(provider)
    return _load_solar_density_table_cached(Path(path), context)


def interpolate_solar_density(radius: TensorLike, table: SolarDensityTable) -> Tuple[Tensor, Optional[Tensor]]:
    """Piecewise-linear-in-radius interpolation of electron/neutron density.

    Args:
        radius: Query points, ``r / R_sun``, any shape, must be within
            ``[0, 1]`` (the table's own domain -- no extrapolation).
        table: SolarDensityTable to interpolate (see
            ``load_solar_density_table``).

    Returns:
        ``(n_e_mol_cm3, n_n_mol_cm3)``, both shaped like ``radius``,
        differentiable w.r.t. ``radius`` (the table itself is a fixed
        buffer). ``n_n_mol_cm3`` is None when ``table.n_n_mol_cm3`` is
        None.

    Raises:
        ValueError: If any query point falls outside ``[0, 1]``.
    """
    query = as_tensor(radius, device=table.radius.device, dtype=table.radius.dtype)
    if torch.any(query < 0) or torch.any(query > 1):
        raise ValueError(
            "radius must be within [0, 1] -- no extrapolation beyond the solar density table's own range."
        )

    n_rows = table.radius.shape[0]
    idx_right = searchsorted_contiguous(table.radius, query, right=False).clamp(1, n_rows - 1)
    idx_left = idx_right - 1

    x0 = table.radius[idx_left]
    x1 = table.radius[idx_right]
    frac = (query - x0) / (x1 - x0).clamp_min(torch.finfo(table.radius.dtype).eps)

    n_e0, n_e1 = table.n_e_mol_cm3[idx_left], table.n_e_mol_cm3[idx_right]
    n_e_mol_cm3 = n_e0 + frac * (n_e1 - n_e0)

    if table.n_n_mol_cm3 is None:
        return n_e_mol_cm3, None

    n_n0, n_n1 = table.n_n_mol_cm3[idx_left], table.n_n_mol_cm3[idx_right]
    n_n_mol_cm3 = n_n0 + frac * (n_n1 - n_n0)
    return n_e_mol_cm3, n_n_mol_cm3
