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
Sun-Earth distance modulation for solar neutrino fluxes.

Solar-model flux tables (``medium.solar.io.load_solar_fluxes``) are
normalized to the standard 1 AU reference distance. The physical flux
actually received at Earth varies by about +-3.4% over the year because
Earth's orbit is elliptical (perihelion around early January, aphelion
around early July):

    Phi(t) = Phi_1AU * (1 AU / R_sun_earth(t))^2

This module provides two ways to apply that modulation, both reading the
tabulated distance-vs-date table (``medium.solar.io.load_sun_earth_distance``):

    sun_earth_distance_factor(date)
        Instantaneous factor (1 AU / R(date))^2 for one calendar date.
    sun_earth_distance_factor_averaged(d1, d2)
        Factor averaged uniformly over an exposure-day window ``[d1, d2]``
        in the same day-of-year convention as
        ``medium.earth.exposure_table.ExposureParameters.exposure_d1/d2``
        (day 0 = the northern-hemisphere winter solstice, *not* January 1st
        -- see ``peanuts.time_average.IntegralDay``'s docstring), so it can
        be combined with the existing nadir-angle exposure integration
        without introducing a second, inconsistent notion of "day of year".

Day-of-year conventions
------------------------
Two distinct day-of-year origins are in play:

    "Jan-1 day": day 0 = January 1st. Used internally to index the bundled
        ``sun_earth_distance.csv`` table (whose ``date`` column are real
        calendar dates) and to interpret a single ISO ``date`` string.
    "exposure day": day 0 = the northern-hemisphere winter solstice, as
        defined by the legacy ``peanuts.time_average.IntegralDay`` and
        propagated to ``ExposureParameters.exposure_d1``/``exposure_d2``.

The winter solstice precedes January 1st by ``_WINTER_SOLSTICE_OFFSET_DAYS``
(~11 days). This fixed offset (rather than the exact per-year solstice
instant, which itself drifts by about +-1 day) is used to convert between
the two conventions; given the distance curve varies smoothly by only a few
percent over the year, a 1-day mismatch is far below any effect this module
introduces.

Module functions:
    sun_earth_distance_au(day_of_year, ...)
        Interpolate the Sun-Earth distance in AU at a Jan-1 day-of-year.
    sun_earth_distance_factor(date, ...)
        Instantaneous (1 AU / R)^2 modulation factor for one calendar date.
    sun_earth_distance_factor_averaged(d1, d2, ...)
        (1 AU / R)^2 factor averaged over an exposure-day window.
"""



from __future__ import annotations

import datetime
from functools import lru_cache
from typing import Optional

import torch

from tpeanuts.medium.solar.io import load_sun_earth_distance
from tpeanuts.util.math import interp1d_linear


Tensor = torch.Tensor

# Northern-hemisphere winter solstice precedes January 1st by ~11 days (see
# peanuts.time_average.IntegralDay: "The time origin day = 0 is the northern
# hemisphere winter solstice midnight").
_WINTER_SOLSTICE_OFFSET_DAYS = 11.0

_YEAR_DAYS = 365.0


@lru_cache(maxsize=8)
def _default_distance_table(device_str: str, dtype: torch.dtype) -> dict:
    return load_sun_earth_distance(device=torch.device(device_str), dtype=dtype)


def _day_of_year_from_date(date: str) -> float:
    """Convert an ISO ``"YYYY-MM-DD"`` date to a Jan-1-anchored day-of-year.

    January 1st is day 0. The calendar year encoded in ``date`` only fixes
    leap-year bookkeeping for that specific parse; the returned value is
    treated as a periodic day-of-year everywhere else in this module.
    """
    parsed = datetime.date.fromisoformat(date)
    return float((parsed - datetime.date(parsed.year, 1, 1)).days)


def _table_day_of_year(table: dict) -> Tensor:
    reference = table["distance_AU"]
    days = [_day_of_year_from_date(date_str) for date_str in table["date"]]
    return torch.tensor(days, device=reference.device, dtype=reference.dtype)


def _interp_periodic(query: Tensor, grid_days: Tensor, values: Tensor) -> Tensor:
    """Linearly interpolate a year-periodic quantity, wrapping at the boundary.

    ``grid_days`` is assumed sorted ascending and to already cover one full
    period (``_YEAR_DAYS``); the first/last samples are mirrored to
    ``grid_days[0] + _YEAR_DAYS`` / ``grid_days[-1] - _YEAR_DAYS`` so a query
    near the year boundary interpolates smoothly across it instead of
    clamping to the nearest tabulated endpoint.
    """
    padded_days = torch.cat(
        [grid_days[-1:] - _YEAR_DAYS, grid_days, grid_days[:1] + _YEAR_DAYS]
    )
    padded_values = torch.cat([values[-1:], values, values[:1]])
    query_wrapped = torch.remainder(query, _YEAR_DAYS)
    return interp1d_linear(
        query_wrapped, padded_days, padded_values,
        device=padded_days.device, dtype=padded_days.dtype,
    )


def sun_earth_distance_au(
    day_of_year: Tensor | float,
    *,
    table: Optional[dict] = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Interpolate the Sun-Earth distance in AU at a Jan-1 day-of-year.

    Args:
        day_of_year: Jan-1-anchored day-of-year (day 0 = January 1st),
            periodic with period 365. Scalar or tensor.
        table: Optional pre-loaded ``load_sun_earth_distance`` result. None
            loads (and caches) the bundled default table.
        device: Target torch device. None uses the table's own device.
        dtype: Target torch dtype.

    Returns:
        Sun-Earth distance in astronomical units, broadcast to
        ``day_of_year``'s shape.
    """
    if table is None:
        table = _default_distance_table(str(device or "cpu"), dtype)

    grid_days = _table_day_of_year(table)
    query = torch.as_tensor(day_of_year, device=grid_days.device, dtype=grid_days.dtype)
    return _interp_periodic(query, grid_days, table["distance_AU"])


def sun_earth_distance_factor(
    date: str,
    *,
    table: Optional[dict] = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Instantaneous flux modulation factor ``(1 AU / R(date))^2``.

    Args:
        date: ISO ``"YYYY-MM-DD"`` calendar date.
        table: Optional pre-loaded ``load_sun_earth_distance`` result. None
            loads (and caches) the bundled default table.
        device: Target torch device. None uses the table's own device.
        dtype: Target torch dtype.

    Returns:
        Scalar dimensionless factor to multiply a 1 AU-normalized solar
        flux by.
    """
    day_of_year = _day_of_year_from_date(date)
    distance_au = sun_earth_distance_au(day_of_year, table=table, device=device, dtype=dtype)
    return 1.0 / distance_au ** 2


def sun_earth_distance_factor_averaged(
    d1: float,
    d2: float,
    *,
    n_samples: int = 200,
    table: Optional[dict] = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Flux modulation factor averaged over an exposure-day window.

    ``d1``/``d2`` use the *exposure-day* convention (day 0 = the
    northern-hemisphere winter solstice), matching
    ``medium.earth.exposure_table.ExposureParameters.exposure_d1/exposure_d2``
    exactly, so this can be combined with the existing nadir-angle exposure
    integration over the same day-of-year window without a separate,
    inconsistent date range. The average is uniform over the day window
    (``mean`` of ``n_samples`` evenly spaced days), matching the uniform
    per-day weighting already used by the nadir-angle exposure integral
    (``peanuts.time_average.IntegralDay``): every day in ``[d1, d2]``
    contributes equally, with no additional live-time weighting.

    Args:
        d1: First exposure day of the integration window, in ``[0, 365]``
            (winter-solstice-anchored).
        d2: Last exposure day of the integration window, in ``[0, 365]``,
            with ``d2 >= d1``.
        n_samples: Number of evenly spaced days sampled over ``[d1, d2]``.
        table: Optional pre-loaded ``load_sun_earth_distance`` result. None
            loads (and caches) the bundled default table.
        device: Target torch device. None uses the table's own device.
        dtype: Target torch dtype.

    Returns:
        Scalar dimensionless factor to multiply a 1 AU-normalized solar
        flux/rate by.

    Raises:
        ValueError: If ``d1``/``d2`` lie outside ``[0, 365]`` or ``d1 > d2``,
            mirroring ``peanuts.time_average.IntegralDay``'s own validation.
    """
    if (not 0.0 <= d1 <= _YEAR_DAYS) or (not 0.0 <= d2 <= _YEAR_DAYS) or (d1 > d2):
        raise ValueError(
            "d1 and d2 must be between 0 and 365, with d2 >= d1 (matching "
            "peanuts.time_average.IntegralDay's exposure-day convention)."
        )

    if table is None:
        table = _default_distance_table(str(device or "cpu"), dtype)

    exposure_days = torch.linspace(
        d1, d2, n_samples,
        device=table["distance_AU"].device, dtype=table["distance_AU"].dtype,
    )
    jan1_days = exposure_days - _WINTER_SOLSTICE_OFFSET_DAYS
    distance_au = sun_earth_distance_au(jan1_days, table=table, device=device, dtype=dtype)
    return (1.0 / distance_au ** 2).mean()
