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

"""Loader for the real Borexino Nature 2018 low-energy rate spectrum.

Ported from v1_0's ``experiments/borexino/io.py``, replacing its
pandas-based loader with ``csv.DictReader`` (this project's own
convention, e.g. ``tpeanuts.experiments.dayabay.io``,
``tpeanuts.source.solar.io``) and taking ``context: RuntimeContext``
instead of separate ``device``/``dtype`` keywords.

Module contents:
    load_low_energy_spectrum(...)
        ``data/experiments/borexino/observation/
        nature2018_low_energy_spectrum.csv`` -> Observation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.detector.common.observation import Observation
from tpeanuts.util.context import RuntimeContext

_RELATIVE_PATH = "observation/nature2018_low_energy_spectrum.csv"


def load_low_energy_spectrum(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Observation:
    """Load Borexino's real Nature 2018 low-energy rate spectrum.

    The published ordinate is differential in the hit estimator
    ``N_h``. Each tabulated row spans one hit bin and also provides the
    corresponding calibrated energy width ``dE``. This loader applies
    the local Jacobian ``dN_h/dE = 1/dE`` and returns a density per
    MeV, so multiplying ``value`` by ``bin_width_MeV`` recovers the
    published per-hit-bin rate. This conversion does not replace a full
    Borexino ``R(N_h|E)`` response or its background model, so it
    remains insufficient for a precision fit.

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/borexino/observation/
            nature2018_low_energy_spectrum.csv``
            (``config.default.BOREXINO_DATA_DIR``).
        context: RuntimeContext the returned tensors are built on.

    Returns:
        Observation with ``x_MeV`` = bin-center energy (converted from
        the table's keV), ``value``/``sigma_minus``/``sigma_plus`` =
        the table's rate and uncertainty converted from per-hit-bin to
        per MeV, and ``bin_width_MeV`` set to the calibrated energy
        width.
    """
    if path is None:
        path = default.BOREXINO_DATA_DIR / _RELATIVE_PATH

    labels = []
    energy_keV = []
    width_keV = []
    rate = []
    rate_error = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(f"bin{int(float(row['bin']))}")
            energy_keV.append(float(row["energy_keV"]))
            width_keV.append(float(row["width_keV"]))
            rate.append(float(row["rate"]))
            rate_error.append(float(row["rate_error"]))

    width_MeV = torch.tensor(width_keV, device=context.device, dtype=context.dtype) / 1000.0
    x_MeV = torch.tensor(energy_keV, device=context.device, dtype=context.dtype) / 1000.0
    value_per_hit = torch.tensor(rate, device=context.device, dtype=context.dtype)
    sigma_per_hit = torch.tensor(rate_error, device=context.device, dtype=context.dtype)
    value = value_per_hit / width_MeV
    sigma = sigma_per_hit / width_MeV

    return Observation.from_symmetric(tuple(labels), x_MeV, value, sigma, context=context, bin_width_MeV=width_MeV)
