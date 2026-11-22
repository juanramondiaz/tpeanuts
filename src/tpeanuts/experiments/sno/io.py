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

"""Loaders for the real SNO Phase-I day/night data.

Ported from v1_0's ``experiments/sno/io.py``, replacing its pandas-based
loaders with ``csv.DictReader`` (this project's own convention, e.g.
``tpeanuts.experiments.dayabay.io``, ``tpeanuts.experiments.borexino.io``)
and taking ``context: RuntimeContext`` instead of separate
``device``/``dtype`` keywords.

Module contents:
    load_day_night_spectrum(...)
        ``data/experiments/sno/observation/day_night_spectrum.csv`` -> two
        Observations (day, night), real background-inclusive counts per
        electron-energy bin.
    load_backgrounds(...)
        ``data/experiments/sno/observation/backgrounds.csv`` -> (day, night)
        real background counts per bin, same binning as the spectrum.
    load_nadir_exposure_raw(...)
        ``data/experiments/sno/observation/coszenith_exposure.csv`` ->
        (eta_rad, exposure) on the nadir-angle convention
        ``tpeanuts.medium.earth`` uses, converted with the same
        cos(zenith) -> eta_rad geometry v1_0's own loader re-derived by
        hand (that geometry has no shared, importable helper in this
        project either -- see this function's own docstring).
    eta_exposure_from_uniform_coszenith(...)
        The cos(zenith) -> eta_rad conversion above, extracted so
        ``tpeanuts.experiments.sno_ii.io.load_zenith_exposure`` (a second
        real caller, a different CSV schema but the same underlying
        uniform-cos(zenith) structure) can reuse it without
        duplicating the geometry a second time.
    total_livetime_days(...)
        ``data/experiments/sno/observation/coszenith_exposure.csv`` ->
        exact (day, night) total livetime in days.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

import tpeanuts.config.default as default
from tpeanuts.detector.common.observation import Observation
from tpeanuts.medium.earth.exposure.time_average import make_eta_grid
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear

Tensor = torch.Tensor
_OBS_DIR = "observation"


def load_day_night_spectrum(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Tuple[Observation, Observation]:
    """Load the real SNO Phase-I day/night electron-energy spectrum.

    Counts are raw (background-inclusive); pair with
    ``load_backgrounds`` to isolate the CC(+ES) signal-like
    contribution. Poisson (``sqrt(N)``) uncertainties are used, since
    the source table gives no separate uncertainty column for the
    counts themselves.

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno/observation/
            day_night_spectrum.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(day, night)`` Observations, ``bin_width_MeV`` set (a
        genuinely binned spectrum), ``value`` = raw counts,
        ``sigma_minus`` = ``sigma_plus`` = ``sqrt(value)`` (clamped at
        1 to stay positive for empty bins).
    """
    if path is None:
        path = default.SNO_DATA_DIR / _OBS_DIR / "day_night_spectrum.csv"

    labels, energy_low, energy_high, day_counts, night_counts = [], [], [], [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(f"bin{int(float(row['bin']))}")
            energy_low.append(float(row["energy_low_MeV"]))
            energy_high.append(float(row["energy_high_MeV"]))
            day_counts.append(float(row["day_counts"]))
            night_counts.append(float(row["night_counts"]))

    energy_low_t = torch.tensor(energy_low, device=context.device, dtype=context.dtype)
    energy_high_t = torch.tensor(energy_high, device=context.device, dtype=context.dtype)
    x_MeV = 0.5 * (energy_low_t + energy_high_t)
    width_MeV = energy_high_t - energy_low_t

    day_counts_t = torch.tensor(day_counts, device=context.device, dtype=context.dtype)
    night_counts_t = torch.tensor(night_counts, device=context.device, dtype=context.dtype)
    day_sigma = torch.sqrt(day_counts_t.clamp_min(1.0))
    night_sigma = torch.sqrt(night_counts_t.clamp_min(1.0))

    labels_t = tuple(labels)
    day = Observation.from_symmetric(labels_t, x_MeV, day_counts_t, day_sigma, context=context, bin_width_MeV=width_MeV)
    night = Observation.from_symmetric(
        labels_t, x_MeV, night_counts_t, night_sigma, context=context, bin_width_MeV=width_MeV,
    )
    return day, night


def load_backgrounds(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Tuple[Tensor, Tensor]:
    """Load the real published day/night background counts per bin.

    Sums the "neutron" (NC neutron-capture leakage) and "low_energy"
    (radioactive) background categories -- the two published in
    ``backgrounds.csv`` -- into one background-count vector per
    period, same binning/order as ``load_day_night_spectrum``.

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno/observation/
            backgrounds.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(day_background, night_background)``, each shape ``(n_bins,)``.
    """
    if path is None:
        path = default.SNO_DATA_DIR / _OBS_DIR / "backgrounds.csv"

    day_bg, night_bg = [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            day_bg.append(float(row["day_neutron"]) + float(row["day_low_energy"]))
            night_bg.append(float(row["night_neutron"]) + float(row["night_low_energy"]))

    return (
        torch.tensor(day_bg, device=context.device, dtype=context.dtype),
        torch.tensor(night_bg, device=context.device, dtype=context.dtype),
    )


def load_nadir_exposure_raw(
    path: Optional[Union[str, Path]] = None, *, context: RuntimeContext,
) -> Tuple[Tensor, Tensor]:
    """Load the real SNO cos(zenith) livetime distribution as (eta_rad, exposure).

    Converts the raw cos(zenith)-binned livetime (seconds) into the
    nadir-angle convention ``tpeanuts.medium.earth`` uses throughout
    (eta_rad=0 -> night/core-crossing, eta_rad=pi -> day/no crossing), via the
    same geometric Jacobian v1_0's own ``load_coszenith_exposure``
    applies by hand -- this project has no shared, importable
    CSV-exposure-to-eta_rad helper either (``tpeanuts.medium.earth.exposure
    .table.build_nadir_exposure`` builds its own table analytically
    from a detector latitude, not from a real livetime CSV, so it is
    not a fit here).

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno/observation/
            coszenith_exposure.csv``.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(eta_rad, exposure)``: nadir-angle grid in radians, shape
        ``(n,)``, strictly increasing 0 to pi, and the corresponding
        (non-negative, un-normalized) exposure weight, shape ``(n,)``,
        in seconds per radian of eta_rad. Integrate with
        ``torch.trapezoid(exposure, x=eta_rad)`` to recover the total
        livetime in seconds. Not split into day/night halves -- see
        ``tpeanuts.experiments.sno.inference_model.SNOModel
        .from_real_exposure`` for that.
    """
    if path is None:
        path = default.SNO_DATA_DIR / _OBS_DIR / "coszenith_exposure.csv"

    raw = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            raw.append(float(row["livetime_s"]))

    raw_t = torch.tensor(raw, device=context.device, dtype=context.dtype)
    return eta_exposure_from_uniform_coszenith(raw_t, context=context)


def eta_exposure_from_uniform_coszenith(raw_livetime_s: Tensor, *, context: RuntimeContext) -> Tuple[Tensor, Tensor]:
    """Convert a uniform-cos(zenith)-binned livetime array into (eta_rad, exposure).

    Shared by ``load_nadir_exposure_raw`` above and
    ``tpeanuts.experiments.sno_ii.io.load_zenith_exposure`` (SNO Phase
    II's own cos(zenith) exposure table uses a different CSV schema --
    ``cos_zenith_low``/``cos_zenith_high``/``period`` columns instead
    of a bare ``cos_zenith`` -- but the same underlying structure: rows
    monotonically ordered over a uniform cos(zenith) grid spanning
    ``[-1, 1]``) -- extracted here once a second real caller needed the
    exact same conversion, not built as a speculative abstraction ahead
    of time.

    Args:
        raw_livetime_s: Livetime per uniform cos(zenith) bin, in
            seconds, shape ``(n,)``, ordered from cos(zenith)=-1 to +1.
        context: RuntimeContext the returned tensors are built on.

    Returns:
        ``(eta_rad, exposure)``: nadir-angle grid in radians, shape
        ``(n,)``, strictly increasing 0 to pi, and the corresponding
        (non-negative, un-normalized) exposure weight, shape ``(n,)``,
        in seconds per radian of eta_rad.
    """
    ns = raw_livetime_s.numel()

    eta_rad = make_eta_grid(ns, daynight=None, context=context)
    cz = torch.linspace(-1.0, 1.0, ns, device=context.device, dtype=context.dtype)
    dcz = cz[1] - cz[0]
    deta = math.pi / (ns - 1)

    zero = torch.zeros((), device=context.device, dtype=context.dtype)
    exposure = interp1d_linear(x=-torch.cos(eta_rad), xp=cz, fp=raw_livetime_s, left=zero, right=zero)
    exposure = exposure * torch.sin(eta_rad) * deta / dcz
    return eta_rad, exposure.clamp_min(0.0)


def total_livetime_days(path: Optional[Union[str, Path]] = None) -> Tuple[float, float]:
    """Real (day, night) total livetime in days, summed directly from the raw table.

    Deliberately independent of ``load_nadir_exposure_raw``'s eta_rad
    -density conversion (whose day/night *ratio* is exact but whose
    absolute scale carries an overall constant Jacobian factor,
    irrelevant once the weights are renormalized per half as
    ``experiments.sno.inference_model.SNOModel.from_real_exposure``
    does): this reads the (day: cos_zenith > 0, night: cos_zenith < 0)
    livetime sums directly off the source column, so the reported
    exposure in days is exact.

    Args:
        path: Optional override path to the CSV. None loads the
            bundled table at ``data/experiments/sno/observation/
            coszenith_exposure.csv``.

    Returns:
        ``(day_days, night_days)``.
    """
    if path is None:
        path = default.SNO_DATA_DIR / _OBS_DIR / "coszenith_exposure.csv"

    day_s, night_s = 0.0, 0.0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            cos_zenith = float(row["cos_zenith"])
            livetime = float(row["livetime_s"])
            if cos_zenith > 0:
                day_s += livetime
            elif cos_zenith < 0:
                night_s += livetime

    return day_s / 86_400.0, night_s / 86_400.0
