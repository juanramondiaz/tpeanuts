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
Solar data input/output helpers for the torch-native solar block.

The torch implementation reads primary solar files from:

    tpeanuts/data/solar

Legacy validation utilities read original peanuts files from:

    tpeanuts/data/peanuts

Module functions:
    solar_provider_path(...)
        Return the canonical table path for one solar-provider product.
    load_solar_density(...)
        Load a canonical electron/neutron density profile.
    load_solar_production(...)
        Load source-dependent radial production distributions, tagged with
        the provider's production-fraction measure (``"shell_fraction"`` or
        ``"radial_pdf"``, see ``_PROVIDER_PRODUCTION_MEASURE``) so callers
        reduce them over radius correctly.
    load_solar_fluxes(...)
        Load total per-source solar neutrino fluxes from CSV.
    load_solar_composition(...)
        Load the solar structure+composition table and derive the
        neutron-to-electron number-density ratio n_n(r)/n_e(r), used to
        build the neutron-density profile for the 3+1 sterile
        neutral-current term.
    load_spectrum_csv(...)
        Load a two-column (energy, spectrum) production-spectrum table from
        CSV.
    load_solar_probability(...)
        Load a canonical energy-dependent survival-probability table.
    load_sun_earth_distance(...)
        Load the date-resolved Sun-Earth distance table from CSV.
"""



from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.util.io import package_dir
from tpeanuts.util.type import as_tensor


Tensor = torch.Tensor

_PROVIDER_FILES: dict[str, dict[str, str]] = {
    "bahcall": {"density": "bahcall/density/bp2000_density_electron_neutron_monotonic.csv", "production": "bahcall/production/bp2004_production.csv", "flux": "bahcall/flux/fluxes_bahcall_bp2004.csv"},
    "zenodo": {"density": "zenodo/density/density_SF3_AGSS09.csv", "production": "zenodo/production/production_SF3_AGSS09.csv", "flux": "zenodo/flux/fluxes_SF3_AGSS09.csv"},
    "legacy": {"density": "legacy/density/density_b16_agss09.csv", "production": "legacy/production/production_b16_agss09.csv", "flux": "legacy/flux/fluxes_b16_agss09.csv"},
}

_SPECTRUM_FILES: dict[str, dict[str, dict[str, str]]] = {
    "legacy": {
        "pp": {"default": "legacy/spectrum/spectrum_pp.csv"},
        "hep": {"default": "legacy/spectrum/spectrum_hep.csv"},
        "7Be": {"default": "legacy/spectrum/spectrum_7Be.csv", "ground": "legacy/spectrum/spectrum_7Be_ground.csv", "excited": "legacy/spectrum/spectrum_7Be_excited.csv"},
        "8B": {"default": "legacy/spectrum/spectrum_8B_winter.csv", "winter": "legacy/spectrum/spectrum_8B_winter.csv", "ortiz": "legacy/spectrum/spectrum_8B_ortiz.csv"},
        "13N": {"default": "legacy/spectrum/spectrum_13N.csv"},
        "15O": {"default": "legacy/spectrum/spectrum_15O.csv"},
        "17F": {"default": "legacy/spectrum/spectrum_17F.csv"},
    },
    "bahcall": {
        "pp": {"default": "bahcall/spectrum/spectrum_pp.csv"},
        "hep": {"default": "bahcall/spectrum/spectrum_hep.csv"},
        "8B": {"default": "bahcall/spectrum/spectrum_8B.csv"},
        "13N": {"default": "bahcall/spectrum/spectrum_13N.csv"},
        "15O": {"default": "bahcall/spectrum/spectrum_15O.csv"},
        "17F": {"default": "bahcall/spectrum/spectrum_17F.csv"},
    },
}

# How each provider's "<source> fraction" production columns are normalized
# over radius (see load_solar_production):
#   "shell_fraction" -- the value in each row is already the share of total
#       production occurring in that tabulated radial shell (as emitted
#       directly by a stellar-structure code); the columns sum (plain sum,
#       not integrated) to ~1 over the table's own -- possibly non-uniform --
#       shells, so the physically correct reduction is a discrete weighted
#       sum, not a trapezoidal integral (integrating would incorrectly
#       reweight each shell by its local grid spacing).
#   "radial_pdf" -- the value is a continuous production density dN/dr,
#       sampled on a fine grid, and requires trapezoidal integration over
#       radius to recover a total fraction of 1.
# Verified directly against the bundled tables: Bahcall's bp2004_production
# fractions sum to ~1 per source (shell_fraction); the zenodo SF3 and legacy
# B16 tables instead integrate (trapz) to ~1 (radial_pdf).
_PROVIDER_PRODUCTION_MEASURE: dict[str, str] = {
    "bahcall": "shell_fraction",
    "zenodo": "radial_pdf",
    "legacy": "radial_pdf",
}


def solar_provider_path(provider: str, product: str) -> Path:
    """Return the canonical table path for one solar-provider product."""
    try:
        relative = _PROVIDER_FILES[provider][product]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROVIDER_FILES))
        raise ValueError(f"Unknown solar provider/product {provider!r}/{product!r}; available providers: {choices}") from exc
    return package_dir() / default.solar_data_dir / relative


def solar_spectrum_path(
    source: str,
    *,
    provider: str = default.solar_spectrum_provider,
    variant: str = "default",
) -> Path:
    """Return a canonical production-spectrum path for a source."""
    try:
        relative = _SPECTRUM_FILES[provider][source][variant]
    except KeyError as exc:
        available = sorted(_SPECTRUM_FILES.get(provider, {}).get(source, {}))
        raise ValueError(
            f"No solar spectrum for provider/source/variant "
            f"{provider!r}/{source!r}/{variant!r}; available variants: {available}"
        ) from exc
    return package_dir() / default.solar_data_dir / relative


def load_solar_spectrum(
    source: str,
    *,
    provider: str = default.solar_spectrum_provider,
    variant: str = "default",
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Load one provider-selected solar production spectrum."""
    return load_spectrum_csv(
        solar_spectrum_path(source, provider=provider, variant=variant),
        energy_column="energy_MeV",
        spectrum_column="spectrum",
        device=device,
        dtype=dtype,
    )


def available_solar_spectrum_sources(provider: str) -> tuple[str, ...]:
    """Return source names with a spectrum registered for ``provider``."""
    if provider not in _SPECTRUM_FILES:
        raise ValueError(f"Unknown solar spectrum provider: {provider!r}")
    return tuple(_SPECTRUM_FILES[provider])


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


def _require_columns(table: pd.DataFrame, required: set[str], *, table_name: str) -> None:
    """Reject a table that is empty or lacks canonical columns."""
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if table.empty:
        raise ValueError(f"{table_name} contains no data rows.")


def _numeric_column(table: pd.DataFrame, column: str, *, table_name: str) -> pd.Series:
    """Return one finite numeric column or raise a descriptive error."""
    values = pd.to_numeric(table[column], errors="coerce")
    if values.isna().any() or not bool(np.isfinite(values.to_numpy()).all()):
        raise ValueError(f"{table_name} column {column!r} contains non-finite or non-numeric values.")
    return values


def _validate_radial_grid(
    table: pd.DataFrame,
    *,
    table_name: str,
    radius_column: str = "radius",
) -> pd.Series:
    """Validate a canonical r/R_sun grid without sorting or deduplicating it."""
    radius = _numeric_column(table, radius_column, table_name=table_name)
    if len(radius) < 2:
        raise ValueError(f"{table_name} radius must contain at least two points.")
    if not bool((radius.diff().iloc[1:] > 0.0).all()):
        raise ValueError(f"{table_name} radius must be strictly increasing with no duplicates.")
    if not bool(((radius >= 0.0) & (radius <= 1.0)).all()):
        raise ValueError(f"{table_name} radius must satisfy 0 <= r/R_sun <= 1.")
    return radius


def _validate_nonnegative_column(
    table: pd.DataFrame,
    column: str,
    *,
    table_name: str,
) -> pd.Series:
    """Return one finite, non-negative numeric column."""
    values = _numeric_column(table, column, table_name=table_name)
    if not bool((values >= 0.0).all()):
        raise ValueError(f"{table_name} column {column!r} must be non-negative.")
    return values


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
    _require_columns(table, {"radius", "electron_density_mol_cm3"}, table_name=table_name)
    radius = _validate_radial_grid(table, table_name=table_name)
    electron_density = _validate_nonnegative_column(
        table, "electron_density_mol_cm3", table_name=table_name
    )
    result = {
        "radius": as_tensor(radius.to_numpy(), device=device, dtype=dtype),
        "electron_density": as_tensor(
            electron_density.to_numpy(), device=device, dtype=dtype,
        ),
    }
    if "neutron_density_mol_cm3" in table:
        neutron_density = _validate_nonnegative_column(
            table, "neutron_density_mol_cm3", table_name=table_name
        )
        result["neutron_density"] = as_tensor(neutron_density.to_numpy(), device=device, dtype=dtype)
    return result


def load_solar_production(
    path: str | Path | None = None,
    *,
    provider: str | None = None,
    production_measure: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor | dict[str, Tensor] | str]:
    """Load radial solar-neutrino production distributions.

    Accepts the canonical wide schema with ``radius`` and one
    ``"<source> fraction"`` column per source. Extra physical columns are
    preserved by the source archive but ignored by this runtime loader.

    Args:
        path: Optional override path to the production CSV. None loads the
            selected provider's (or the configured default) table.
        provider: Optional canonical provider name (``"bahcall"``,
            ``"zenodo"`` or ``"legacy"``), used both to resolve ``path`` when
            it is None and, together with ``_PROVIDER_PRODUCTION_MEASURE``,
            to default ``production_measure``.
        production_measure: How the ``"<source> fraction"`` columns are
            normalized over radius -- ``"shell_fraction"`` (discrete
            per-shell weights that already sum to ~1, e.g. Bahcall) or
            ``"radial_pdf"`` (a continuous density dN/dr requiring
            trapezoidal integration to reach 1, e.g. zenodo SF3 or legacy
            B16). None (the default) resolves from ``provider`` --
            including the package-default provider
            (``tpeanuts.config.default.solar_provider``) when both ``path``
            and ``provider`` are omitted -- via
            ``_PROVIDER_PRODUCTION_MEASURE``; an unrecognised or explicit
            ``path`` override with no ``provider`` defaults to
            ``"radial_pdf"`` (the historical behaviour) since the table's
            origin is otherwise unknown.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary with ``"radius"``, ``"fractions"`` (per-source tensors),
        and ``"production_measure"`` (``"shell_fraction"`` or
        ``"radial_pdf"``).
    """
    if path is None:
        effective_provider = provider or default.solar_provider
        path = solar_provider_path(provider, "production") if provider else package_dir() / default.solar_data_dir / default.solar_production_filename
    else:
        effective_provider = provider
    table = pd.read_csv(path)
    table_name = "Solar production table"
    _require_columns(table, {"radius"}, table_name=table_name)
    radius_series = _validate_radial_grid(table, table_name=table_name)
    fractions = {
        column.removesuffix(" fraction"): as_tensor(table[column].to_numpy(), device=device, dtype=dtype)
        for column in table.columns if column.endswith(" fraction")
    }
    if not fractions:
        raise ValueError("Solar production table contains no '<source> fraction' columns")

    if production_measure is None:
        production_measure = _PROVIDER_PRODUCTION_MEASURE.get(effective_provider, "radial_pdf")
    if production_measure not in ("shell_fraction", "radial_pdf"):
        raise ValueError(
            "production_measure must be 'shell_fraction' or 'radial_pdf', "
            f"got {production_measure!r}."
        )

    radius = as_tensor(radius_series.to_numpy(), device=device, dtype=dtype)
    for source, fraction in fractions.items():
        if fraction.shape != radius.shape:
            raise ValueError(
                f"Solar production distribution {source!r} must have the same length as radius."
            )
        if not torch.isfinite(fraction).all():
            raise ValueError(f"Solar production distribution {source!r} contains non-finite values.")
        normalization = (
            fraction.sum()
            if production_measure == "shell_fraction"
            else torch.trapezoid(fraction, x=radius)
        )
        if not torch.isfinite(normalization) or normalization <= 0:
            raise ValueError(f"Solar production distribution {source!r} has non-positive normalization.")

    return {
        "radius": radius,
        "fractions": fractions,
        "production_measure": production_measure,
    }


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
    ratio is what ``SolarProfile`` multiplies by its own (independently
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


def load_solar_fluxes(
    path: str | Path | None = None,
    *,
    provider: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Load total per-source solar neutrino fluxes from CSV.

    Reads a two-column CSV (legacy ``fluxes_b16``-style layout) with a
    "fraction" column giving the source name (e.g. "pp", "8B") and a "flux"
    column giving its total integrated flux (in the standard solar-model
    units, neutrinos / cm^2 / s, as tabulated by the source CSV).

    Args:
        path: Optional override path to the flux CSV. None loads the
            configured default solar flux table from
            ``package_dir() / default.solar_data_dir /
            default.solar_fluxes_filename``.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary mapping each source name to its scalar total-flux tensor.
    """
    if path is None:
        path = solar_provider_path(provider, "flux") if provider else package_dir() / default.solar_data_dir / default.solar_fluxes_filename

    table = pd.read_csv(path)
    table_name = "Solar flux table"
    _require_columns(table, {"fraction", "flux"}, table_name=table_name)
    if table["fraction"].isna().any():
        raise ValueError("Solar flux table contains an empty source name.")
    sources = table["fraction"].astype(str).str.strip()
    if bool((sources == "").any()):
        raise ValueError("Solar flux table contains an empty source name.")
    duplicated = sources[sources.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Solar flux table contains duplicate sources: {duplicated}")
    flux_values = _validate_nonnegative_column(table, "flux", table_name=table_name)

    return {
        source: as_tensor(value, device=device, dtype=dtype)
        for source, value in zip(sources, flux_values)
    }


def load_spectrum_csv(
    path: str | Path,
    *,
    energy_column: str | None = None,
    spectrum_column: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Load a two-column (energy, spectrum) production-spectrum table.

    Used for source-specific neutrino production spectra (e.g. the 8B or hep
    beta-decay spectral shapes) that weight the energy dependence of a solar
    source's flux.

    Args:
        path: Path to the spectrum CSV.
        energy_column: Name of the column holding neutrino energy in MeV.
            None defaults to the first column in the file.
        spectrum_column: Name of the column holding the (typically
            unnormalized) spectral weight / probability density. None
            defaults to the second column in the file.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary with "energy" (MeV) and "spectrum" (spectral weight)
        tensors, each shape (n,).
    """
    table = pd.read_csv(path)

    if energy_column is None:
        energy_column = table.columns[0]

    if spectrum_column is None:
        spectrum_column = table.columns[1]

    return {
        "energy": as_tensor(table[energy_column].to_numpy(), device=device, dtype=dtype),
        "spectrum": as_tensor(table[spectrum_column].to_numpy(), device=device, dtype=dtype),
    }


def load_solar_probability(
    path: str | Path,
    *,
    energy_column: str = "energy_MeV",
    probability_columns: tuple[str, ...] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Load a source-published or experimentally fitted probability table."""
    table = pd.read_csv(path)
    if energy_column not in table:
        raise ValueError(f"Solar probability table is missing required column: {energy_column}")
    if probability_columns is None:
        probability_columns = tuple(
            column for column in table.columns
            if column != energy_column and "uncertainty" not in column
        )
    if not probability_columns:
        raise ValueError("Solar probability table contains no probability columns")
    return {
        "energy": as_tensor(table[energy_column].to_numpy(), device=device, dtype=dtype),
        "probabilities": {
            column: as_tensor(table[column].to_numpy(), device=device, dtype=dtype)
            for column in probability_columns
        },
    }


def load_sun_earth_distance(
    path: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, list[str] | Tensor]:
    """Load the date-resolved Sun-Earth distance table.

    Used to convert a calendar date into the physical Sun-Earth baseline for
    coherent or incoherent solar-to-Earth propagation (the Earth's orbit is
    elliptical, so this distance varies by about +-1.7% over the year).

    Args:
        path: Optional override path to the distance CSV. None loads the
            bundled default table from
            ``package_dir() / default.solar_data_dir /
            default.solar_sun_earth_distance_filename``.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary with:
            "date": List of date strings, one per row.
            "distance_km": Tensor of Sun-Earth distances in km, shape (n,).
            "distance_AU": Tensor of Sun-Earth distances in astronomical
                units, shape (n,).

    Raises:
        ValueError: If the table is missing any of the required columns
            "date", "distance_km", or "distance_AU".
    """
    if path is None:
        path = (
            package_dir()
            / default.solar_data_dir
            / default.solar_sun_earth_distance_filename
        )

    table = pd.read_csv(path)

    required = {"date", "distance_km", "distance_AU"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Sun-earth distance table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return {
        "date": [str(value) for value in table["date"].to_list()],
        "distance_km": as_tensor(table["distance_km"].to_numpy(), device=device, dtype=dtype),
        "distance_AU": as_tensor(table["distance_AU"].to_numpy(), device=device, dtype=dtype),
    }
