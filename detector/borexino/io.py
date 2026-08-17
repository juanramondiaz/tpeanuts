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
Loader for the real Borexino Nature 2018 low-energy rate spectrum.

Module contents:
    load_low_energy_spectrum(...)
        data/detector/borexino/observation/nature2018_low_energy_spectrum.csv
        -> Observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.detector.common.observation import Observation
from tpeanuts.util.io import package_dir

_RELATIVE_PATH = "borexino/observation/nature2018_low_energy_spectrum.csv"


def load_low_energy_spectrum(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Observation:
    """Load Borexino's real Nature 2018 low-energy rate spectrum.

    The published ordinate is differential in the hit estimator ``N_h``.
    Each tabulated row spans one hit and also provides the corresponding
    calibrated energy width ``dE``. This loader applies the local Jacobian
    ``dN_h/dE = 1/dE`` and returns a density per MeV, so multiplying
    ``value`` by ``bin_width_MeV`` recovers the published per-hit-bin rate.
    This conversion does not replace a full Borexino ``R(N_h|E)`` response
    or its background model, so it remains insufficient for a precision fit.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/borexino/observation/
            nature2018_low_energy_spectrum.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        Observation with ``x_MeV`` = bin-center energy (converted from the
        table's keV), ``value``/``sigma_minus``/``sigma_plus`` = the
        table's rate and uncertainty converted from per-hit-bin to per MeV,
        and ``bin_width_MeV`` set to the calibrated energy width.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _RELATIVE_PATH

    table = pd.read_csv(path)

    labels = tuple(f"bin{int(b)}" for b in table["bin"])
    x_MeV = torch.tensor(table["energy_keV"].to_numpy(), device=device, dtype=dtype) / 1000.0
    width_MeV = torch.tensor(table["width_keV"].to_numpy(), device=device, dtype=dtype) / 1000.0
    value_per_hit = torch.tensor(table["rate"].to_numpy(), device=device, dtype=dtype)
    sigma_per_hit = torch.tensor(table["rate_error"].to_numpy(), device=device, dtype=dtype)
    value = value_per_hit / width_MeV
    sigma = sigma_per_hit / width_MeV

    return Observation.from_symmetric(labels, x_MeV, value, sigma, bin_width_MeV=width_MeV)
