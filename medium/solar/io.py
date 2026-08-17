#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#  This module is part of the Master's Thesis (MSc Dissertation):
#  - Fast Simulation of Neutrino Oscillations in Matter
#
#  Author:
#      Juan Ramon Diaz Santos <diazjuan@alumni.uv.es>
#
#  Supervisors:
#      Roberto Ruiz de Austri Bazan <rruiz@ific.uv.es>
#      Michele Lucente <michele.lucente@unibo.it>
#
#  Date:
#      June 2026
# =============================================================================

"""
Solar medium (structural) input/output helpers for the torch-native solar block.

Loads the density/composition tables that describe the solar *medium*
(electron/neutron density as a function of radius), independent of the
production-source tables (radial production distributions, total fluxes,
per-source spectra) which live in ``source.solar.io``. The torch
implementation reads primary solar files from:

    tpeanuts/data/solar

Legacy validation utilities read original peanuts files from:

    tpeanuts/data/peanuts

Module functions:
    solar_provider_path(...)
        Return the canonical density-table path for one solar-provider.
    load_solar_density(...)
        Load a canonical electron/neutron density profile.
    load_solar_composition(...)
        Load the solar structure+composition table and derive the
        neutron-to-electron number-density ratio n_n(r)/n_e(r), used to
        build the neutron-density profile for the 3+1 sterile
        neutral-current term.
"""



from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.util.io import (
    package_dir,
    require_columns,
    validate_nonnegative_column,
    validate_radial_grid,
)
from tpeanuts.util.type import as_tensor


Tensor = torch.Tensor

# Only the density product: production/flux/spectrum provider tables live in
# source.solar.io (see that module's docstring for the split rationale).
_PROVIDER_FILES: dict[str, dict[str, str]] = {
    "bahcall": {"density": "bahcall/density/bp2000_density_electron_neutron_monotonic.csv"},
    "zenodo": {"density": "zenodo/density/density_SF3_AGSS09.csv"},
    "legacy": {"density": "legacy/density/density_b16_agss09.csv"},
}

# Mass number A and proton number Z for every composition column of the
# zenodo "struct+nu_SF3_*" solar-model tables (see load_solar_composition).
# Isotope-resolved species (H1, He3, He4, and the CNO isotopes) use their
# exact (A, Z); trace metals from Ne upward are not isotope-resolved in the
# table, so each is assigned its dominant naturally-occurring/solar isotope.
# This is a controlled approximation: metals collectively contribute only
# ~1-2% of the solar mass fraction (the rest is H1/He4/He3), so the choice
# of isotope for any single trace metal has a negligible effect on the
# derived neutron density -- see load_solar_composition below.
_COMPOSITION_ISOTOPE_AZ: dict[str, tuple[int, int]] = {
    "H1": (1, 1),
    "He4": (4, 2),
    "He3": (3, 2),
    "C12": (12, 6),
    "C13": (13, 6),
    "N14": (14, 7),
    "N15": (15, 7),
    "O16": (16, 8),
    "O17": (17, 8),
    "O18": (18, 8),
    "Ne": (20, 10),   # Ne20, dominant solar/natural isotope
    "Na": (23, 11),   # Na23 (100% natural)
    "Mg": (24, 12),   # Mg24, dominant isotope
    "Al": (27, 13),   # Al27 (100% natural)
    "Si": (28, 14),   # Si28, dominant isotope
    "P": (31, 15),    # P31 (100% natural)
    "S": (32, 16),    # S32, dominant isotope
    "Cl": (35, 17),   # Cl35, dominant isotope
    "Ar": (36, 18),   # Ar36, dominant *solar* isotope (unlike Ar40 in Earth's atmosphere)
    "K": (39, 19),    # K39, dominant isotope
    "Ca": (40, 20),   # Ca40, dominant isotope
    "Sc": (45, 21),   # Sc45 (100% natural)
    "Ti": (48, 22),   # Ti48, dominant isotope
    "V": (51, 23),    # V51, dominant isotope
    "Cr": (52, 24),   # Cr52, dominant isotope
    "Mn": (55, 25),   # Mn55 (100% natural)
    "Fe": (56, 26),   # Fe56, dominant isotope
    "Co": (59, 27),   # Co59 (100% natural)
    "Ni": (58, 28),   # Ni58, dominant isotope
}

# Non-composition columns preceding the isotope mass-fraction block in the
# zenodo "struct+nu_SF3_*" tables (fixed column order; the files have no
# machine-parseable header row -- the "#"-prefixed header line does not
# tokenize 1:1 against the data rows, so the full column order is pinned
# here explicitly rather than parsed).
_STRUCT_NU_LEADING_COLUMNS: tuple[str, ...] = (
    "R_sun", "mass_sun", "L_sun", "logR", "logT", "logP", "logRho", "Csound",
    "dm", "nu_pp", "nu_pep", "nu_hep", "nu_7Be", "nu_8B", "nu_13N", "nu_15O",
    "nu_17F", "log_ne",
)


def solar_provider_path(provider: str, product: str) -> Path:
    """Return the canonical density-table path for one solar-provider product."""
    try:
        relative = _PROVIDER_FILES[provider][product]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROVIDER_FILES))
        raise ValueError(f"Unknown solar provider/product {provider!r}/{product!r}; available providers: {choices}") from exc
    return package_dir() / default.solar_data_dir / relative


def load_solar_density(
    path: str | Path | None = None,
    *,
    provider: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Load a canonical solar electron/neutron density table.

    The required columns are ``radius`` and
    ``electron_density_mol_cm3``. ``neutron_density_mol_cm3`` is optional,
    allowing Standard-Model-only datasets to omit it.
    """
    if path is None:
        path = solar_provider_path(provider, "density") if provider else package_dir() / default.solar_data_dir / default.solar_density_filename
    table = pd.read_csv(path)
    table_name = "Solar density table"
    require_columns(table, {"radius", "electron_density_mol_cm3"}, table_name=table_name)
    radius = validate_radial_grid(table, table_name=table_name)
    electron_density = validate_nonnegative_column(
        table, "electron_density_mol_cm3", table_name=table_name
    )
    result = {
        "radius": as_tensor(radius.to_numpy(), device=device, dtype=dtype),
        "electron_density": as_tensor(
            electron_density.to_numpy(), device=device, dtype=dtype,
        ),
    }
    if "neutron_density_mol_cm3" in table:
        neutron_density = validate_nonnegative_column(
            table, "neutron_density_mol_cm3", table_name=table_name
        )
        result["neutron_density"] = as_tensor(neutron_density.to_numpy(), device=device, dtype=dtype)
    return result


def load_solar_composition(
    path: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Derive the neutron-to-electron density ratio from the solar composition.

    Reads a whitespace-delimited zenodo ``struct+nu_SF3_*``-style solar
    structure table (radial mass fractions of H1, He3, He4, and every
    tabulated heavier isotope/element) and derives the dimensionless ratio

        r(R) = n_n(R) / n_e(R)

    assuming a fully ionized plasma, where both number densities are
    computed *self-consistently from composition alone*:

        n_e(R) proportional to  sum_i X_i(R) / A_i * Z_i
        n_n(R) proportional to  sum_i X_i(R) / A_i * (A_i - Z_i)

    with X_i the tabulated mass fraction, A_i the mass number, and Z_i the
    proton number of species i (``_COMPOSITION_ISOTOPE_AZ``). The common
    proportionality factor (mass density over the atomic mass unit) cancels
    exactly in the ratio, so this function never needs to interpret the
    table's own density/electron-density columns or their absolute units --
    only the (dimensionless, self-normalizing) mass fractions matter. The
    ratio is what ``SolarMediumProfile`` multiplies by its own (independently
    validated) electron-density profile to obtain ``density_n`` on the
    profile's native radius grid (see ``medium.solar.profile``).

    Args:
        path: Optional override path to the struct+nu composition table.
            None loads the configured default from
            ``package_dir() / default.solar_data_dir /
            default.solar_composition_filename``.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary with:
            "radius": Tensor of solar radius fractions rho = r/R_sun on the
                table's own grid, shape (n,).
            "neutron_to_electron_ratio": Dimensionless tensor n_n(R)/n_e(R)
                on the same grid, shape (n,).
    """
    if path is None:
        path = package_dir() / default.solar_data_dir / default.solar_composition_filename

    columns = (*_STRUCT_NU_LEADING_COLUMNS, *_COMPOSITION_ISOTOPE_AZ.keys())
    table = pd.read_csv(path, sep=r"\s+", names=columns, skiprows=1)

    radius = as_tensor(table["R_sun"].to_numpy(), device=device, dtype=dtype)

    n_e_over_common_factor = torch.zeros_like(radius)
    n_n_over_common_factor = torch.zeros_like(radius)
    for name, (mass_number, proton_number) in _COMPOSITION_ISOTOPE_AZ.items():
        mass_fraction = as_tensor(table[name].to_numpy(), device=device, dtype=dtype)
        number_density_over_common_factor = mass_fraction / mass_number
        n_e_over_common_factor = (
            n_e_over_common_factor + number_density_over_common_factor * proton_number
        )
        n_n_over_common_factor = (
            n_n_over_common_factor
            + number_density_over_common_factor * (mass_number - proton_number)
        )

    ratio = n_n_over_common_factor / torch.clamp(
        n_e_over_common_factor, min=torch.finfo(dtype).tiny
    )

    return {
        "radius": radius,
        "neutron_to_electron_ratio": ratio,
    }
