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
    package_dir(...)
        Return the tpeanuts package root directory.
    default_solar_data_dir(...)
        Return the default directory holding torch-native solar data files.
    default_legacy_data_dir(...)
        Return the default directory holding legacy peanuts data files.
    as_tensor(...)
        Convert array-like values to a torch tensor on the requested
        device/dtype.
    load_b16_solar_model(...)
        Load the tabulated B16 solar radius, electron density, and
        per-source production-fraction profile from CSV.
    load_b16_fluxes(...)
        Load total per-source solar neutrino fluxes from CSV.
    load_spectrum_csv(...)
        Load a two-column (energy, spectrum) production-spectrum table from
        CSV.
    load_sun_earth_distance(...)
        Load the date-resolved Sun-Earth distance table from CSV.
"""



from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
import torch

import tpeanuts.util.default as default
from tpeanuts.util.torch_util import default_device


Tensor = torch.Tensor


def package_dir() -> Path:
    """Return the tpeanuts package root directory.

    Returns:
        Path to the directory two levels above this file, i.e. the
        repository/package root containing ``data/``.
    """
    return Path(__file__).resolve().parents[2]


def default_solar_data_dir() -> Path:
    """Return the default directory holding torch-native solar data files.

    Returns:
        Path to ``<package_dir>/<default.solar_data_dir>`` (normally
        ``tpeanuts/data/solar``).
    """
    return package_dir() / default.solar_data_dir


def default_legacy_data_dir() -> Path:
    """Return the default directory holding legacy peanuts data files.

    Returns:
        Path to ``<package_dir>/<default.legacy_data_dir>`` (normally
        ``tpeanuts/data/peanuts``), used by validation helpers that compare
        against the original peanuts implementation.
    """
    return package_dir() / default.legacy_data_dir


def as_tensor(values, *, device: Union[str, torch.device], dtype: torch.dtype) -> Tensor:
    """Convert array-like values to a torch tensor on the requested device/dtype.

    Args:
        values: Array-like input (e.g. a NumPy array, list, or scalar)
            accepted by ``torch.as_tensor``.
        device: Target torch device, or None to use the package default
            device (see ``tpeanuts.util.torch_util.default_device``).
        dtype: Target torch dtype.

    Returns:
        Tensor with the requested device and dtype.
    """
    return torch.as_tensor(values, device=default_device(device), dtype=dtype)


def load_b16_solar_model(
    path: str | Path | None = None,
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Load the tabulated B16 solar density and production-fraction profile.

    Reads a CSV with the legacy ``nudistr_b16_agss09``-style layout: a
    ``radius`` column (solar radius fraction rho = r/R_sun, dimensionless,
    in [0, 1]), a ``density_log_10`` column (base-10 logarithm of the
    electron density in mol/cm^3), and one ``"<source> fraction"`` column
    per solar neutrino production source (e.g. "pp", "8B", "7Be ground"),
    giving the relative radial production-rate distribution of that source
    (unnormalized, i.e. not necessarily integrating to 1 over the radius
    grid).

    Args:
        path: Optional override path to the solar model CSV. None loads the
            bundled default B16 (Bahcall, Serenelli & Basu 2016) model from
            ``default_solar_data_dir() / default.solar_model_filename``.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary with:
            "radius": Tensor of solar radius fractions rho, shape (n,).
            "density": Tensor of electron density in mol/cm^3 (linear scale,
                i.e. ``10 ** density_log_10``), shape (n,).
            "fractions": Dict mapping each source name to its tabulated
                production-fraction tensor, shape (n,).
    """
    if path is None:
        path = default_solar_data_dir() / default.solar_model_filename

    table = pd.read_csv(path)

    radius = as_tensor(table["radius"].to_numpy(), device=device, dtype=dtype)
    density = 10.0 ** as_tensor(table["density_log_10"].to_numpy(), device=device, dtype=dtype)

    fractions = {}
    for column in table.columns:
        if column.endswith(" fraction"):
            name = column.replace(" fraction", "")
            fractions[name] = as_tensor(table[column].to_numpy(), device=device, dtype=dtype)

    return {
        "radius": radius,
        "density": density,
        "fractions": fractions,
    }


def load_b16_fluxes(
    path: str | Path | None = None,
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor]:
    """Load total per-source solar neutrino fluxes from CSV.

    Reads a two-column CSV (legacy ``fluxes_b16``-style layout) with a
    "fraction" column giving the source name (e.g. "pp", "8B") and a "flux"
    column giving its total integrated flux (in the standard solar-model
    units, neutrinos / cm^2 / s, as tabulated by the source CSV).

    Args:
        path: Optional override path to the flux CSV. None loads the bundled
            default B16 flux table from
            ``default_solar_data_dir() / default.solar_fluxes_filename``.
        device: Target torch device for the loaded tensors. None uses the
            package default device.
        dtype: Target torch dtype for the loaded tensors.

    Returns:
        Dictionary mapping each source name to its scalar total-flux tensor.
    """
    if path is None:
        path = default_solar_data_dir() / default.solar_fluxes_filename

    table = pd.read_csv(path)

    return {
        str(row["fraction"]): as_tensor(row["flux"], device=device, dtype=dtype)
        for _, row in table.iterrows()
    }


def load_spectrum_csv(
    path: str | Path,
    *,
    energy_column: str | None = None,
    spectrum_column: str | None = None,
    device: Union[str, torch.device] | None = None,
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


def load_sun_earth_distance(
    path: str | Path | None = None,
    *,
    device: Union[str, torch.device] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, list[str] | Tensor]:
    """Load the date-resolved Sun-Earth distance table.

    Used to convert a calendar date into the physical Sun-Earth baseline for
    coherent or incoherent solar-to-Earth propagation (the Earth's orbit is
    elliptical, so this distance varies by about +-1.7% over the year).

    Args:
        path: Optional override path to the distance CSV. None loads the
            bundled default table from
            ``default_solar_data_dir() / default.solar_sun_earth_distance_filename``.
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
        path = default_solar_data_dir() / default.solar_sun_earth_distance_filename

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
