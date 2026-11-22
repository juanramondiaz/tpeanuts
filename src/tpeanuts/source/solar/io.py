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

"""Solar-neutrino source I/O: production, total flux, and production spectra.

Provider-neutral loaders for "where do the neutrinos come from" tables --
independent of the solar propagation medium (a future
``tpeanuts.medium.solar``, not ported yet, which will read only its own
density/composition tables). Reads from ``tpeanuts.data.solar`` (see that
package's own ``README.md`` for which files of each provider were ported
and why).

Ported from v1_0's ``source/solar/io.py``, translated to this project's
own established data-loading conventions (no ``pandas``, no
``util.io``/``util.type``/``package_dir()`` -- none of those exist in
this rewrite -- ``csv.DictReader`` + ``tpeanuts.util.tensor.as_tensor`` +
a module-level ``_DATA_DIR`` instead, exactly like
``tpeanuts.source.atmosphere.io``/``tpeanuts.medium.earth.density``
already do). The column-detection and normalization logic itself is
otherwise unchanged from v1_0.

Module contents:
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
    load_solar_flux(...)
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

import csv
from pathlib import Path
from typing import Optional, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.data import package_data_dir
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import as_tensor

Tensor = torch.Tensor

_DATA_DIR = package_data_dir() / "solar"

_PROVIDER_FILES: dict[str, dict[str, str]] = {
    "bahcall": {
        "production": "bahcall/production/bp2004_production.csv",
        "flux": "bahcall/flux/fluxes_bahcall_bp2004.csv",
    },
    "zenodo": {
        "production": "zenodo/production/production_SF3_AGSS09.csv",
        "flux": "zenodo/flux/fluxes_SF3_AGSS09.csv",
    },
    "legacy": {
        "production": "legacy/production/production_b16_agss09.csv",
        "flux": "legacy/flux/fluxes_b16_agss09.csv",
    },
}

_SPECTRUM_FILES: dict[str, dict[str, dict[str, str]]] = {
    "legacy": {
        "pp": {"default": "legacy/spectrum/spectrum_pp.csv"},
        "hep": {"default": "legacy/spectrum/spectrum_hep.csv"},
        "7Be": {
            "default": "legacy/spectrum/lines_7Be.csv",
            "ground": "legacy/spectrum/line_7Be_ground.csv",
            "excited": "legacy/spectrum/line_7Be_excited.csv",
        },
        "pep": {"default": "legacy/spectrum/line_pep.csv"},
        "8B": {
            "default": "legacy/spectrum/spectrum_8B_winter.csv",
            "winter": "legacy/spectrum/spectrum_8B_winter.csv",
            "ortiz": "legacy/spectrum/spectrum_8B_ortiz.csv",
        },
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
# Verified directly against the ported tables: Bahcall's bp2004_production
# fractions sum to ~1 per source (shell_fraction); the zenodo SF3 and legacy
# B16 tables instead integrate (trapz) to ~1 (radial_pdf).
_PROVIDER_PRODUCTION_MEASURE: dict[str, str] = {
    "bahcall": "shell_fraction",
    "zenodo": "radial_pdf",
    "legacy": "radial_pdf",
}


def solar_source_provider_path(provider: str, product: str) -> Path:
    """Return the canonical table path for one solar source provider product."""
    try:
        relative = _PROVIDER_FILES[provider][product]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROVIDER_FILES))
        raise ValueError(
            f"Unknown solar source provider/product {provider!r}/{product!r}; available providers: {choices}"
        ) from exc
    return _DATA_DIR / relative


def solar_spectrum_path(
    source: str,
    *,
    provider: str = default.SOLAR_SPECTRUM_PROVIDER,
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
    return _DATA_DIR / relative


def _read_csv_rows(path: Union[str, Path]) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file into its own field names and rows, in file order.

    Raises:
        ValueError: If the file has no header row.
    """
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path!r}.")
        return list(reader.fieldnames), list(reader)


def load_solar_production(
    path: Optional[Union[str, Path]] = None,
    *,
    provider: Optional[str] = None,
    production_measure: Optional[str] = None,
    context: RuntimeContext,
) -> dict[str, Union[Tensor, dict[str, Tensor], str]]:
    """Load radial solar-neutrino production distributions.

    Accepts the canonical wide schema with ``radius`` and one
    ``"<source> fraction"`` column per source. Extra physical columns
    (e.g. Bahcall's ``temperature_MK``/``density_log_10``/
    ``shell_mass_solar``/``be7_mass_fraction``) are ignored by this
    runtime loader.

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
            (``tpeanuts.config.default.SOLAR_PROVIDER``) when both ``path``
            and ``provider`` are omitted -- via
            ``_PROVIDER_PRODUCTION_MEASURE``; an unrecognised provider or an
            explicit ``path`` override with no ``provider`` defaults to
            ``"radial_pdf"`` (the historical behaviour) since the table's
            origin is otherwise unknown.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        Dictionary with ``"radius"``, ``"fractions"`` (per-source tensors),
        and ``"production_measure"`` (``"shell_fraction"`` or
        ``"radial_pdf"``).

    Raises:
        ValueError: If ``radius`` is missing, not finite and strictly
            increasing, if no ``"<source> fraction"`` column is present, if
            ``production_measure`` is not recognised, or if any source's
            distribution is non-finite or has non-positive normalization.
    """
    if path is None:
        effective_provider: Optional[str] = provider or default.SOLAR_PROVIDER
        path = (
            solar_source_provider_path(provider, "production")
            if provider
            else _DATA_DIR / default.SOLAR_PRODUCTION_FILENAME
        )
    else:
        effective_provider = provider

    fieldnames, rows = _read_csv_rows(path)
    if "radius" not in fieldnames:
        raise ValueError(f"Solar production table is missing the required 'radius' column: {path!r}.")

    radius_values = [float(row["radius"]) for row in rows]
    radius = as_tensor(radius_values, device=context.device, dtype=context.dtype)
    if not torch.isfinite(radius).all() or (radius.numel() > 1 and torch.any(torch.diff(radius) <= 0)):
        raise ValueError("Solar production table's radius grid must be finite and strictly increasing.")

    fraction_columns = [column for column in fieldnames if column.endswith(" fraction")]
    if not fraction_columns:
        raise ValueError(f"Solar production table contains no '<source> fraction' columns: {path!r}.")

    fractions = {
        column.removesuffix(" fraction"): as_tensor(
            [float(row[column]) for row in rows], device=context.device, dtype=context.dtype
        )
        for column in fraction_columns
    }

    if production_measure is None:
        production_measure = _PROVIDER_PRODUCTION_MEASURE.get(effective_provider or "", "radial_pdf")
    if production_measure not in ("shell_fraction", "radial_pdf"):
        raise ValueError(
            "production_measure must be 'shell_fraction' or 'radial_pdf', "
            f"got {production_measure!r}."
        )

    for source, fraction in fractions.items():
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


def load_solar_flux(
    path: Optional[Union[str, Path]] = None,
    *,
    provider: Optional[str] = None,
    context: RuntimeContext,
) -> dict[str, Tensor]:
    """Load total per-source solar neutrino fluxes from CSV.

    Reads a two-column CSV (``fraction``/``flux``) with a ``"fraction"``
    column giving the source name (e.g. ``"pp"``, ``"8B"``) and a
    ``"flux"`` column giving its total integrated flux (in the standard
    solar-model units, neutrinos / cm^2 / s, as tabulated by the source
    CSV).

    Args:
        path: Optional override path to the flux CSV. None loads the
            selected provider's (or the configured default) table.
        provider: Optional canonical provider name, used to resolve
            ``path`` when it is None.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        Dictionary mapping each source name to its scalar total-flux tensor.

    Raises:
        ValueError: If a required column is missing, a source name is
            empty or duplicated, or any flux value is negative.
    """
    if path is None:
        path = solar_source_provider_path(provider, "flux") if provider else _DATA_DIR / default.SOLAR_FLUXES_FILENAME

    fieldnames, rows = _read_csv_rows(path)
    missing = {"fraction", "flux"}.difference(fieldnames)
    if missing:
        raise ValueError(f"Solar flux table is missing required columns {sorted(missing)}: {path!r}.")

    sources: list[str] = []
    flux_values: list[float] = []
    seen: set[str] = set()
    for row in rows:
        source = str(row["fraction"]).strip()
        if not source:
            raise ValueError("Solar flux table contains an empty source name.")
        if source in seen:
            raise ValueError(f"Solar flux table contains duplicate source: {source!r}.")
        seen.add(source)
        flux = float(row["flux"])
        if not torch.isfinite(torch.as_tensor(flux)) or flux < 0:
            raise ValueError(f"Solar total flux {source!r} must be finite and non-negative, got {flux!r}.")
        sources.append(source)
        flux_values.append(flux)

    return {
        source: as_tensor(value, device=context.device, dtype=context.dtype)
        for source, value in zip(sources, flux_values, strict=True)
    }


def load_spectrum_csv(
    path: Union[str, Path],
    *,
    energy_column: Optional[str] = None,
    spectrum_column: Optional[str] = None,
    context: RuntimeContext,
) -> dict[str, Tensor]:
    """Load a two-column (energy, spectrum) production-spectrum table.

    Used for source-specific neutrino production spectra (e.g. the 8B or
    hep beta-decay spectral shapes) that weight the energy dependence of a
    solar source's flux.

    Args:
        path: Path to the spectrum CSV.
        energy_column: Name of the column holding neutrino energy in MeV.
            None defaults to the file's first column.
        spectrum_column: Name of the column holding the (typically
            unnormalized) spectral weight / probability density. None
            defaults to the file's second column.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        Dictionary with "energy" (MeV) and "spectrum" (spectral weight)
        tensors, each shape (n,).
    """
    fieldnames, rows = _read_csv_rows(path)
    if energy_column is None:
        energy_column = fieldnames[0]
    if spectrum_column is None:
        spectrum_column = fieldnames[1]

    return {
        "energy": as_tensor([float(row[energy_column]) for row in rows], device=context.device, dtype=context.dtype),
        "spectrum": as_tensor(
            [float(row[spectrum_column]) for row in rows], device=context.device, dtype=context.dtype
        ),
    }


def load_solar_spectrum(
    source: str,
    *,
    provider: str = default.SOLAR_SPECTRUM_PROVIDER,
    variant: str = "default",
    context: RuntimeContext,
) -> dict[str, Union[Tensor, str]]:
    """Load one provider-selected solar production spectrum."""
    is_line = source in _LINE_SOURCES
    table = load_spectrum_csv(
        solar_spectrum_path(source, provider=provider, variant=variant),
        energy_column="energy_MeV",
        spectrum_column="weight" if is_line else "spectrum",
        context=context,
    )
    result: dict[str, Union[Tensor, str]] = {**table}
    result["kind"] = "line" if is_line else "continuous"
    return result


def available_solar_spectrum_sources(provider: str) -> tuple[str, ...]:
    """Return source names with a spectrum registered for ``provider``."""
    if provider not in _SPECTRUM_FILES:
        raise ValueError(f"Unknown solar spectrum provider: {provider!r}")
    return tuple(_SPECTRUM_FILES[provider])
