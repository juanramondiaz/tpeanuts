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
Solar-neutrino source I/O: production, total flux, and production spectra.

Provider-neutral loaders for "where do the neutrinos come from" tables --
independent of the solar propagation medium (``medium.solar``, which reads
only the density/composition tables via ``medium.solar.io``). Reads primary
tables from ``tpeanuts/data/solar``.

Module functions:
    solar_source_provider_path(...)
        Return the canonical table path for one solar-source provider
        product ("production" or "flux").
    solar_spectrum_path(...)
        Return a canonical production-spectrum path for a source/variant.
    load_solar_production(...)
        Load source-dependent radial production distributions, tagged with
        the provider's production-fraction measure (``"shell_fraction"`` or
        ``"radial_pdf"``, see ``_PROVIDER_PRODUCTION_MEASURE``) so callers
        reduce them over radius correctly.
    load_solar_fluxes(...)
        Load total per-source solar neutrino fluxes from CSV.
    load_spectrum_csv(...)
        Load a two-column (energy, spectrum) production-spectrum table from
        CSV.
    load_solar_spectrum(...)
        Load one provider-selected solar production spectrum.
    available_solar_spectrum_sources(...)
        Return source names with a spectrum registered for a provider.
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

_PROVIDER_FILES: dict[str, dict[str, str]] = {
    "bahcall": {"production": "bahcall/production/bp2004_production.csv", "flux": "bahcall/flux/fluxes_bahcall_bp2004.csv"},
    "zenodo": {"production": "zenodo/production/production_SF3_AGSS09.csv", "flux": "zenodo/flux/fluxes_SF3_AGSS09.csv"},
    "legacy": {"production": "legacy/production/production_b16_agss09.csv", "flux": "legacy/flux/fluxes_b16_agss09.csv"},
}

_SPECTRUM_FILES: dict[str, dict[str, dict[str, str]]] = {
    "legacy": {
        "pp": {"default": "legacy/spectrum/spectrum_pp.csv"},
        "hep": {"default": "legacy/spectrum/spectrum_hep.csv"},
        "7Be": {"default": "legacy/spectrum/lines_7Be.csv", "ground": "legacy/spectrum/line_7Be_ground.csv", "excited": "legacy/spectrum/line_7Be_excited.csv"},
        "pep": {"default": "legacy/spectrum/line_pep.csv"},
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
        "7Be": {
            "default": "bahcall/spectrum/lines_7Be.csv",
            "ground": "bahcall/spectrum/line_7Be_ground.csv",
            "excited": "bahcall/spectrum/line_7Be_excited.csv",
        },
        "pep": {"default": "bahcall/spectrum/line_pep.csv"},
    },
}

_LINE_SOURCES = {"7Be", "pep"}

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


def solar_source_provider_path(provider: str, product: str) -> Path:
    """Return the canonical table path for one solar-source provider product."""
    try:
        relative = _PROVIDER_FILES[provider][product]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROVIDER_FILES))
        raise ValueError(f"Unknown solar source provider/product {provider!r}/{product!r}; available providers: {choices}") from exc
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
        path = solar_source_provider_path(provider, "production") if provider else package_dir() / default.solar_data_dir / default.solar_production_filename
    else:
        effective_provider = provider
    table = pd.read_csv(path)
    table_name = "Solar production table"
    require_columns(table, {"radius"}, table_name=table_name)
    radius_series = validate_radial_grid(table, table_name=table_name)
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
        path = solar_source_provider_path(provider, "flux") if provider else package_dir() / default.solar_data_dir / default.solar_fluxes_filename

    table = pd.read_csv(path)
    table_name = "Solar flux table"
    require_columns(table, {"fraction", "flux"}, table_name=table_name)
    if table["fraction"].isna().any():
        raise ValueError("Solar flux table contains an empty source name.")
    sources = table["fraction"].astype(str).str.strip()
    if bool((sources == "").any()):
        raise ValueError("Solar flux table contains an empty source name.")
    duplicated = sources[sources.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Solar flux table contains duplicate sources: {duplicated}")
    flux_values = validate_nonnegative_column(table, "flux", table_name=table_name)

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


def load_solar_spectrum(
    source: str,
    *,
    provider: str = default.solar_spectrum_provider,
    variant: str = "default",
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Tensor | str]:
    """Load one provider-selected solar production spectrum."""
    is_line = source in _LINE_SOURCES
    table = load_spectrum_csv(
        solar_spectrum_path(source, provider=provider, variant=variant),
        energy_column="energy_MeV",
        spectrum_column="weight" if is_line else "spectrum",
        device=device,
        dtype=dtype,
    )
    table["kind"] = "line" if is_line else "continuous"
    return table


def available_solar_spectrum_sources(provider: str) -> tuple[str, ...]:
    """Return source names with a spectrum registered for ``provider``."""
    if provider not in _SPECTRUM_FILES:
        raise ValueError(f"Unknown solar spectrum provider: {provider!r}")
    return tuple(_SPECTRUM_FILES[provider])
