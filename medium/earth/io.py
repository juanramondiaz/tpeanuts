#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Provider-neutral readers for canonical radial Earth models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.util.io import package_dir
from tpeanuts.util.type import as_tensor

Tensor = torch.Tensor

_PROVIDER_DENSITY_FILES = {
    "prem": "prem/density/prem_density.csv",
    "ak135": "ak135/density/ak135f_density.csv",
    "legacy": "legacy/density/earth_density.csv",
}


def earth_provider_path(provider: str) -> Path:
    """Return the canonical radial-density table for an Earth provider."""
    try:
        relative = _PROVIDER_DENSITY_FILES[provider]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROVIDER_DENSITY_FILES))
        raise ValueError(f"Unknown Earth density provider {provider!r}; available: {choices}") from exc
    return package_dir() / default.earth_reference_data_dir / relative


def earth_provider_fit_paths(provider: str) -> tuple[Path, Path]:
    """Return electron/neutron even-power fit tables for a provider."""
    base = package_dir() / default.earth_reference_data_dir / provider / "fit"
    electron = base / "even_power_electron.csv"
    neutron = base / "even_power_neutron.csv"
    if not electron.is_file():
        raise ValueError(f"Earth provider {provider!r} has no even-power fit: {electron}")
    return electron, neutron


@dataclass(frozen=True)
class EarthDensityTable:
    """Pointwise spherical Earth model ordered by increasing radius."""

    radius_km: Tensor
    radius_fraction: Tensor
    mass_density_g_cm3: Tensor
    electron_fraction: Tensor | None = None
    electron_density_mol_cm3: Tensor | None = None
    neutron_density_mol_cm3: Tensor | None = None
    depth_km: Tensor | None = None


def load_earth_density(
    path: str | Path | None = None,
    *,
    provider: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> EarthDensityTable:
    """Load a canonical PREM/ak135 radial-density CSV.

    Required columns are ``radius_km``, ``radius_fraction`` and
    ``mass_density_g_cm3``. Composition-derived columns are optional because
    seismic reference models do not determine the electron fraction.
    """
    if path is None:
        path = earth_provider_path(provider or default.earth_density_provider)
    table = pd.read_csv(path)
    required = {"radius_km", "radius_fraction", "mass_density_g_cm3"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Earth density table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    table = table.sort_values("radius_km", kind="stable")

    def optional(name: str) -> Tensor | None:
        if name not in table:
            return None
        return as_tensor(table[name].to_numpy(), device=device, dtype=dtype)

    return EarthDensityTable(
        radius_km=as_tensor(table["radius_km"].to_numpy(), device=device, dtype=dtype),
        radius_fraction=as_tensor(
            table["radius_fraction"].to_numpy(), device=device, dtype=dtype
        ),
        mass_density_g_cm3=as_tensor(
            table["mass_density_g_cm3"].to_numpy(), device=device, dtype=dtype
        ),
        electron_fraction=optional("electron_fraction"),
        electron_density_mol_cm3=optional("electron_density_mol_cm3"),
        neutron_density_mol_cm3=optional("neutron_density_mol_cm3"),
        depth_km=optional("depth_km"),
    )
