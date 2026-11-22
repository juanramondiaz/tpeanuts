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

"""The canonical flat atmospheric-neutrino flux table, and its loader.

Ported from v1_0's ``source/atmosphere/io.py::AtmosphericFluxTable``/
``load_atmospheric_flux`` -- the flat (no production-height axis) part
only, see this package's own ``__init__`` docstring. ``energy_GeV`` is
converted to ``energy_MeV`` at load time (the file on disk keeps the
provider's own GeV unit unchanged, matching v1_0's copy verbatim --
only this loader's own *output* follows this project's MeV convention,
the same boundary-conversion pattern
``tpeanuts.medium.atmosphere.density``/``geometry`` already use for
v1_0's km inputs). ``cos_zenith`` is kept as-is: converting it to this
project's own nadir-angle ``eta_rad`` convention is a geometry concern for a
future ``tpeanuts.pipeline.atmosphere`` to handle, not this purely
data-side module's.

``angle_output_path``/``write_angle_flux_csv`` are the writing-side
counterpart of ``load_atmospheric_flux``: the one place that decides
the per-angle output filename and CSV layout every height-differential
flux generator in this package (``tpeanuts.source.atmosphere.mceq``,
``tpeanuts.source.atmosphere.honda``) writes to, so both generators
produce byte-identical schemas by construction rather than by two
independently-maintained copies of the same writer.

Module contents:
    AtmosphericFluxTable
        Frozen dataclass: the canonical long-form flux table
        (``energy_MeV``, ``cos_zenith``, ``flux``, ``particle``, and the
        optional ``azimuth_deg``/``altitude_m`` axes).
    load_atmospheric_flux(...)
        Read a canonical provider-neutral CSV into an
        ``AtmosphericFluxTable``.
    angle_output_path(...)
        Build the per-angle output CSV path for a height-differential
        flux generator.
    write_angle_flux_csv(...)
        Write one per-angle CSV, one row per ``(particle, energy,
        altitude)`` combination, in the schema ``load_atmospheric_flux``
        reads.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

import tpeanuts.config.default as default
from tpeanuts.data import require
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import as_tensor

_DEFAULT_FLUX_CSV_PATH = default.ATMOSPHERE_DATA_DIR / "honda" / "flux" / "honda_flux.csv"

_GEV_TO_MEV = 1.0e3


@dataclass(frozen=True)
class AtmosphericFluxTable:
    """Canonical long-form atmospheric-neutrino flux table.

    Attributes:
        energy_MeV: True neutrino energy, in MeV, shape ``(n_rows,)``.
        cos_zenith: Cosine of the detector-frame zenith angle, shape
            ``(n_rows,)``, in ``[-1, 1]``.
        flux: Differential flux at each row, in whatever unit the source
            file's own metadata documents (e.g.
            ``m^-2 s^-1 sr^-1 GeV^-1`` for the ported Honda table --
            provider conventions differ, so the unit is never assumed
            here, only carried by the caller's own knowledge of the file).
        particle: Particle label at each row (e.g. ``"nue"``), shape
            ``(n_rows,)``.
        azimuth_deg: Optional azimuth angle, in degrees, shape
            ``(n_rows,)``. None if the source table has no azimuth column.
        altitude_m: Optional production altitude above the surface, in
            metres, shape ``(n_rows,)``. None if the source table has no
            production-height axis (true of every table ported so far --
            see this package's own ``__init__`` docstring).
    """

    energy_MeV: torch.Tensor
    cos_zenith: torch.Tensor
    flux: torch.Tensor
    particle: Tuple[str, ...]
    azimuth_deg: Optional[torch.Tensor] = None
    altitude_m: Optional[torch.Tensor] = None


def load_atmospheric_flux(
    path: Optional[Path] = None,
    *,
    context: RuntimeContext,
) -> AtmosphericFluxTable:
    """Load a canonical provider-neutral atmospheric flux CSV.

    Args:
        path: Path to the CSV. Defaults to the Honda surface-flux table
            ``honda/flux/honda_flux.csv`` under
            ``config.default.ATMOSPHERE_DATA_DIR`` (the per-user generated-data
            directory). It is not shipped with the package (the Honda/HKKM
            terms of use are unverified, see ``DATA_LICENSES.md``): it must
            be generated locally from the Honda tables, or a table passed
            explicitly.
        context: Runtime device/dtype the returned tensors are built on.

    Returns:
        AtmosphericFluxTable with the CSV's own rows, in the file's own
        order.

    Raises:
        DataNotAvailableError: If ``path`` is None and the default Honda
            table has not been generated on this machine.
        ValueError: If a required column (``energy_GeV``, ``cos_zenith``,
            ``particle``, ``flux``) is missing, or if any row has
            ``cos_zenith`` outside ``[-1, 1]``, ``energy_GeV <= 0``, or
            ``flux < 0``.
    """
    if path is None:
        path = require(
            _DEFAULT_FLUX_CSV_PATH,
            what="The default Honda/HKKM surface-flux table",
            how_to_obtain=(
                "It is not distributed with tpeanuts (see DATA_LICENSES.md): build it locally with "
                "`tpeanuts-data build honda-flux` (needs the raw Honda table; add --from DIR if you already have "
                "it) or pass an explicit path. Set TPEANUTS_USER_DATA_DIR to relocate the generated-data directory."
            ),
        )

    energy_GeV = []
    cos_zenith = []
    particle = []
    flux = []
    azimuth_deg = []
    altitude_m = []
    has_azimuth = False
    has_altitude = False

    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"energy_GeV", "cos_zenith", "particle", "flux"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Atmospheric flux table is missing required columns: {sorted(missing)}.")
        has_azimuth = "azimuth_deg" in (reader.fieldnames or ())
        has_altitude = "altitude_km" in (reader.fieldnames or ())

        for row in reader:
            row_energy_GeV = float(row["energy_GeV"])
            row_cos_zenith = float(row["cos_zenith"])
            row_flux = float(row["flux"])
            if not (-1.0 <= row_cos_zenith <= 1.0):
                raise ValueError(f"cos_zenith must lie in [-1, 1], got {row_cos_zenith!r}.")
            if row_energy_GeV <= 0:
                raise ValueError(f"energy_GeV must be strictly positive, got {row_energy_GeV!r}.")
            if row_flux < 0:
                raise ValueError(f"flux must be non-negative, got {row_flux!r}.")

            energy_GeV.append(row_energy_GeV)
            cos_zenith.append(row_cos_zenith)
            particle.append(str(row["particle"]))
            flux.append(row_flux)
            if has_azimuth:
                azimuth_deg.append(float(row["azimuth_deg"]))
            if has_altitude:
                altitude_m.append(float(row["altitude_km"]) * 1.0e3)

    energy_MeV_t = as_tensor(energy_GeV, device=context.device, dtype=context.dtype) * _GEV_TO_MEV
    cos_zenith_t = as_tensor(cos_zenith, device=context.device, dtype=context.dtype)
    flux_t = as_tensor(flux, device=context.device, dtype=context.dtype)
    azimuth_deg_t = as_tensor(azimuth_deg, device=context.device, dtype=context.dtype) if has_azimuth else None
    altitude_m_t = as_tensor(altitude_m, device=context.device, dtype=context.dtype) if has_altitude else None

    return AtmosphericFluxTable(
        energy_MeV=energy_MeV_t,
        cos_zenith=cos_zenith_t,
        flux=flux_t,
        particle=tuple(particle),
        azimuth_deg=azimuth_deg_t,
        altitude_m=altitude_m_t,
    )


def angle_output_path(output_dir: Path, alpha_deg: float) -> Path:
    """Per-angle output CSV path, keyed by the surface/table zenith angle.

    Always keyed by ``alpha_deg`` (never ``theta_deg``), matching v1_0's
    own ``build_angle_output_path`` convention: the surface/table angle
    is what a generator actually solves/looks up at, so it is the
    natural file-identity axis regardless of which angle convention the
    caller originally supplied.

    Args:
        output_dir: Directory the file is written under.
        alpha_deg: Surface/table zenith angle in degrees.

    Returns:
        ``output_dir / "flux_alpha_<alpha_deg>deg.csv"``.
    """
    return output_dir / f"flux_alpha_{alpha_deg:06.3f}deg.csv"


def write_angle_flux_csv(
    path: Path,
    *,
    energy_GeV: torch.Tensor,
    altitude_km: torch.Tensor,
    per_particle_flux: Dict[str, torch.Tensor],
    cos_zenith: float,
) -> None:
    """Write one per-angle CSV, in the exact schema ``load_atmospheric_flux`` reads.

    One row per ``(particle, energy, altitude)`` triple -- every
    particle's flux is flattened from its own ``(n_energy, n_altitude)``
    grid into that many rows, all sharing this file's single
    ``cos_zenith`` value. The parent directory is created if needed.

    Args:
        path: Output CSV path.
        energy_GeV: 1-D energy grid, shape ``(n_energy,)``.
        altitude_km: 1-D production-altitude grid, shape ``(n_altitude,)``.
        per_particle_flux: Mapping from particle label to its
            height-differential flux, each shaped ``(n_energy,
            n_altitude)``, in whatever unit the generator's own metadata
            documents.
        cos_zenith: Cosine of the detector-frame zenith angle stored on
            every row of this file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    energy_list = energy_GeV.detach().cpu().tolist()
    altitude_list = altitude_km.detach().cpu().tolist()

    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["energy_GeV", "cos_zenith", "particle", "flux", "altitude_km"])
        for particle, flux_Eh in per_particle_flux.items():
            flux_list = flux_Eh.detach().cpu().tolist()
            for i_E, energy_value in enumerate(energy_list):
                row_flux = flux_list[i_E]
                for i_h, altitude_value in enumerate(altitude_list):
                    writer.writerow([energy_value, cos_zenith, particle, row_flux[i_h], altitude_value])
