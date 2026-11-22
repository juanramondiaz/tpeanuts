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

"""Reactor-neutrino source I/O: the antineutrino spectra and the reactor's own parameters.

Loaders for the tables a reactor source is built from, shipped under ``tpeanuts/data/reactor/dayabay/`` (the default
reactor is Daya Bay's cores): the per-isotope Huber-Mueller spectra, the non-equilibrium correction to them, and the
fission fractions, energy released per fission and thermal power of the reactor. They are properties of the
source, not of the Daya Bay analysis, so they live here; ``tpeanuts.experiments.dayabay`` reads them from this
module and keeps only what is specific to its detectors and cores (baselines, spent-fuel correction, backgrounds).

Every loader takes an optional ``path`` to replace the packaged file (same schema), so another reactor can be
described without touching the package. CSV is read with ``csv.DictReader`` and tensors are built with
``tpeanuts.util.tensor.as_tensor``, as in ``tpeanuts.source.solar.io``.

Module contents:
    reactor_data_path(...)
        Path of one packaged reactor data file.
    load_reactor_parameters(...)
        Fission fractions, energy per fission and thermal power.
    load_huber_mueller_spectra(...)
        Per-isotope antineutrino spectrum, ``(E_MeV, N_per_fission_per_MeV)`` pairs.
    load_nonequilibrium_correction(...)
        Per-isotope non-equilibrium flux correction ``C(E)`` (U235, Pu239, Pu241).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.tensor import as_tensor

Tensor = torch.Tensor

PARAMETERS_FILENAME = "reactor_parameters.csv"
THERMAL_POWER_FILENAME = "reactor_thermal_power_gw.txt"
SPECTRA_FILENAME = "huber_mueller_spectra.csv"
NONEQUILIBRIUM_FILENAME = "nonequilibrium_correction.csv"


def reactor_data_path(filename: str) -> Path:
    """Return the path of one packaged reactor data file under ``default.REACTOR_DATA_DIR``."""
    return default.REACTOR_DATA_DIR / filename


def _read_csv_rows(path: Union[str, Path]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV file into its own field names and rows, in file order.

    Raises:
        ValueError: If the file has no header row.
    """
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path!r}.")
        return list(reader.fieldnames), list(reader)


def _curves_by_isotope(
    rows: List[Dict[str, str]], value_column: str, context: RuntimeContext,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """Group ``rows`` by isotope into ``{isotope: (E_MeV, value)}`` tensors, each sorted by energy."""
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["isotope"], []).append(row)

    result: Dict[str, Tuple[Tensor, Tensor]] = {}
    for isotope, isotope_rows in grouped.items():
        isotope_rows = sorted(isotope_rows, key=lambda row: float(row["E_MeV"]))
        result[isotope] = (
            as_tensor([float(row["E_MeV"]) for row in isotope_rows], device=context.device, dtype=context.dtype),
            as_tensor([float(row[value_column]) for row in isotope_rows], device=context.device, dtype=context.dtype),
        )
    return result


def load_reactor_parameters(
    path: Optional[Union[str, Path]] = None, *, power_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Fission fractions, energy-per-fission, and nominal thermal power.

    Args:
        path: CSV with columns ``isotope, fission_fraction, energy_per_fission_MeV``; None reads the packaged file.
        power_path: Text file holding the thermal power in GW; None reads the packaged file.

    Returns:
        ``(fission_fractions, energy_per_fission_MeV, thermal_power_gw)``: the first two are ``{isotope: value}``
        dicts (isotope in ``tpeanuts.source.reactor.reactor.ISOTOPES``); the third is a scalar float, GW, shared by
        every core of the packaged reactor (Daya Bay's six).
    """
    _, rows = _read_csv_rows(Path(path) if path is not None else reactor_data_path(PARAMETERS_FILENAME))
    fission_fractions = {row["isotope"]: float(row["fission_fraction"]) for row in rows}
    energy_per_fission = {row["isotope"]: float(row["energy_per_fission_MeV"]) for row in rows}
    power_file = Path(power_path) if power_path is not None else reactor_data_path(THERMAL_POWER_FILENAME)
    thermal_power_gw = float(power_file.read_text())
    return fission_fractions, energy_per_fission, thermal_power_gw


def load_huber_mueller_spectra(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """Per-isotope Huber-Mueller antineutrino spectrum.

    Args:
        path: CSV with columns ``isotope, E_MeV, N_per_fission_per_MeV``; None reads the packaged file.
        context: RuntimeContext the tensors are built on.

    Returns:
        ``{isotope: (E_MeV, N_per_fission_per_MeV)}``, each a 1-D tensor, sorted by energy.
    """
    _, rows = _read_csv_rows(Path(path) if path is not None else reactor_data_path(SPECTRA_FILENAME))
    return _curves_by_isotope(rows, "N_per_fission_per_MeV", context)


def load_nonequilibrium_correction(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    """Per-isotope non-equilibrium flux correction ``C(E)`` (U235, Pu239, Pu241 only).

    Args:
        path: CSV with columns ``isotope, E_MeV, C``; None reads the packaged file.
        context: RuntimeContext the tensors are built on.

    Returns:
        ``{isotope: (E_MeV, C)}``, each a 1-D tensor, sorted by energy.
    """
    _, rows = _read_csv_rows(Path(path) if path is not None else reactor_data_path(NONEQUILIBRIUM_FILENAME))
    return _curves_by_isotope(rows, "C", context)
