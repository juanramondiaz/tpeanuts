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
Loaders for the real SNO Phase-I day/night data.

Module contents:
    load_day_night_spectrum(...)
        data/detector/sno/observation/day_night_spectrum.csv -> two Observations
        (day, night), real background-inclusive counts per electron-energy
        bin.
    load_backgrounds(...)
        data/detector/sno/observation/backgrounds.csv -> (day, night) real
        background counts per bin, same binning as the spectrum.
    load_coszenith_exposure(...)
        data/detector/sno/observation/coszenith_exposure.csv -> (eta, exposure)
        on the nadir-angle convention ``tpeanuts.medium.earth`` uses,
        converted with the same cos(zenith)->eta geometry as
        ``tpeanuts.medium.earth.exposure_io.nadir_exposure_from_csv``
        (re-derived here rather than imported, since that loader expects a
        column literally named "Exposure" and this file's real column is
        "livetime_s").
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

import tpeanuts.config.default as default
from tpeanuts.detector.common.observation import Observation
from tpeanuts.medium.earth.exposure_math import make_eta_grid
from tpeanuts.util.io import package_dir
from tpeanuts.util.math import interp1d_linear

_OBS_DIR = "sno/observation"


def load_day_night_spectrum(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Observation, Observation]:
    """Load the real SNO Phase-I day/night electron-energy spectrum.

    Counts are raw (background-inclusive); pair with
    ``load_backgrounds`` to isolate the CC(+ES) signal-like contribution.
    Poisson (``sqrt(N)``) uncertainties are used, since the source table
    gives no separate uncertainty column for the counts themselves.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno/observation/day_night_spectrum.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(day, night)`` Observations, ``bin_width_MeV`` set (a genuinely
        binned spectrum), ``value`` = raw counts, ``sigma_minus`` =
        ``sigma_plus`` = ``sqrt(value)`` (clamped at 1 to stay positive for
        empty bins).
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _OBS_DIR / "day_night_spectrum.csv"
    table = pd.read_csv(path)

    labels = tuple(f"bin{int(b)}" for b in table["bin"])
    energy_low = torch.tensor(table["energy_low_MeV"].to_numpy(), device=device, dtype=dtype)
    energy_high = torch.tensor(table["energy_high_MeV"].to_numpy(), device=device, dtype=dtype)
    x_MeV = 0.5 * (energy_low + energy_high)
    width_MeV = energy_high - energy_low

    day_counts = torch.tensor(table["day_counts"].to_numpy(), device=device, dtype=dtype)
    night_counts = torch.tensor(table["night_counts"].to_numpy(), device=device, dtype=dtype)
    day_sigma = torch.sqrt(day_counts.clamp_min(1.0))
    night_sigma = torch.sqrt(night_counts.clamp_min(1.0))

    day = Observation.from_symmetric(labels, x_MeV, day_counts, day_sigma, bin_width_MeV=width_MeV)
    night = Observation.from_symmetric(labels, x_MeV, night_counts, night_sigma, bin_width_MeV=width_MeV)
    return day, night


def load_backgrounds(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the real published day/night background counts per bin.

    Sums the "neutron" (NC neutron-capture leakage) and "low_energy"
    (radioactive) background categories -- the two published in
    ``backgrounds.csv`` -- into one background-count vector per period,
    same binning/order as ``load_day_night_spectrum``.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno/observation/backgrounds.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(day_background, night_background)``, each shape ``(n_bins,)``.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _OBS_DIR / "backgrounds.csv"
    table = pd.read_csv(path)

    day_bg = torch.tensor(
        (table["day_neutron"] + table["day_low_energy"]).to_numpy(), device=device, dtype=dtype,
    )
    night_bg = torch.tensor(
        (table["night_neutron"] + table["night_low_energy"]).to_numpy(), device=device, dtype=dtype,
    )
    return day_bg, night_bg


def load_coszenith_exposure(
    path: Optional[str | Path] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the real SNO cos(zenith) livetime distribution as (eta, exposure).

    Converts the raw cos(zenith)-binned livetime (seconds) into the nadir-
    angle convention ``tpeanuts.medium.earth`` uses throughout (eta=0 ->
    night/core-crossing, eta=pi -> day/no crossing), via the same geometric
    Jacobian ``tpeanuts.medium.earth.exposure_io.nadir_exposure_from_csv``
    applies internally for its "CosZenith" mode.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno/observation/coszenith_exposure.csv``.
        device: Target torch device.
        dtype: Target real dtype.

    Returns:
        ``(eta, exposure)``: nadir-angle grid in radians, shape ``(n,)``,
        strictly increasing 0 to pi, and the corresponding (non-negative,
        un-normalized) exposure weight, shape ``(n,)``, in seconds per
        radian of eta. Integrate with ``torch.trapezoid(exposure, x=eta)``
        to recover the total livetime in seconds.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _OBS_DIR / "coszenith_exposure.csv"
    table = pd.read_csv(path)

    raw = torch.tensor(table["livetime_s"].to_numpy(), device=device, dtype=dtype)
    ns = raw.numel()

    eta = make_eta_grid(ns, daynight=None, device=device, dtype=dtype)
    cz = torch.linspace(-1.0, 1.0, ns, device=device, dtype=dtype)
    dcz = cz[1] - cz[0]
    deta = math.pi / (ns - 1)

    exposure = interp1d_linear(
        x=-torch.cos(eta), xp=cz, fp=raw, left=0.0, right=0.0, device=device, dtype=dtype,
    )
    exposure = exposure * torch.sin(eta) * deta / dcz
    return eta, exposure.clamp_min(0.0)


def total_livetime_days(path: Optional[str | Path] = None) -> tuple[float, float]:
    """Real (day, night) total livetime in days, summed directly from the raw table.

    Deliberately independent of ``load_coszenith_exposure``'s eta-density
    conversion (whose day/night *ratio* is exact but whose absolute scale
    carries an overall constant Jacobian factor, irrelevant once the weights
    are renormalized per half as ``detector.sno.event_rate`` does): this
    reads the (day: cos_zenith > 0, night: cos_zenith < 0) livetime sums
    directly off the source column, so the reported exposure in days is
    exact.

    Args:
        path: Optional override path to the CSV. None loads the bundled
            table at ``data/detector/sno/observation/coszenith_exposure.csv``.

    Returns:
        ``(day_days, night_days)``.
    """
    if path is None:
        path = package_dir() / default.detector_data_dir / _OBS_DIR / "coszenith_exposure.csv"
    table = pd.read_csv(path)
    day_s = float(table.loc[table["cos_zenith"] > 0, "livetime_s"].sum())
    night_s = float(table.loc[table["cos_zenith"] < 0, "livetime_s"].sum())
    return day_s / 86400.0, night_s / 86400.0
