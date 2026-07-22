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

"""Pytest-compatible tests for tpeanuts.medium.solar.geometry."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.medium.solar.geometry import (
    sun_earth_distance_au,
    sun_earth_distance_factor,
    sun_earth_distance_factor_averaged,
)
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_table():
    # A single, clean sinusoidal cycle: distance minimum (perihelion) at
    # day 0 (Jan 1), maximum (aphelion) at day 182 (~July), back down by
    # day 364 -- daily resolution, matching the bundled table's shape.
    import datetime
    base = datetime.date(2026, 1, 1)
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(365)]
    day = torch.arange(365, dtype=DTYPE)
    # Minimum (perihelion) at day 0, matching Earth's real orbit.
    distance_au = 1.0 - 0.0167 * torch.cos(2 * torch.pi * day / 365.0)
    return {
        "date": dates,
        "distance_km": distance_au * 1.496e8,
        "distance_AU": distance_au.to(device=DEVICE, dtype=DTYPE),
    }


def test_sun_earth_distance_au_matches_table_at_exact_grid_point():
    table = make_synthetic_table()
    day0 = sun_earth_distance_au(0.0, table=table, device=DEVICE, dtype=DTYPE)
    day10 = sun_earth_distance_au(10.0, table=table, device=DEVICE, dtype=DTYPE)

    assert_close(day0, table["distance_AU"][0], name="distance at day 0")
    assert_close(day10, table["distance_AU"][10], name="distance at day 10")


def test_sun_earth_distance_au_wraps_smoothly_at_year_boundary():
    table = make_synthetic_table()
    # Query exactly at the boundary (day 365 == day 0) and just past it.
    at_zero = sun_earth_distance_au(0.0, table=table, device=DEVICE, dtype=DTYPE)
    at_year = sun_earth_distance_au(365.0, table=table, device=DEVICE, dtype=DTYPE)
    just_past = sun_earth_distance_au(365.5, table=table, device=DEVICE, dtype=DTYPE)

    assert_close(at_year, at_zero, name="periodic wrap at day 365 == day 0")
    # No discontinuity: half a day past the wrap should sit between day 364
    # and day 0/1, not jump to a clamped endpoint value.
    day364 = sun_earth_distance_au(364.0, table=table, device=DEVICE, dtype=DTYPE)
    assert min(float(day364), float(at_zero)) <= float(just_past) <= max(float(day364), float(at_zero)) + 1e-6


def test_sun_earth_distance_factor_perihelion_exceeds_aphelion():
    table = make_synthetic_table()
    perihelion = sun_earth_distance_factor("2026-01-01", table=table, device=DEVICE, dtype=DTYPE)
    aphelion = sun_earth_distance_factor("2026-07-02", table=table, device=DEVICE, dtype=DTYPE)

    # Earth is closest to the Sun in early January (perihelion): flux is
    # higher there, i.e. the (1 AU / R)^2 factor must be > 1, and larger
    # than the same factor at aphelion (where it must be < 1).
    assert float(perihelion) > 1.0
    assert float(aphelion) < 1.0
    assert float(perihelion) > float(aphelion)


def test_sun_earth_distance_factor_averaged_matches_manual_mean():
    table = make_synthetic_table()
    d1, d2 = 11.0, 41.0  # exposure-day window landing exactly on Jan 1 - Jan 31 (offset 11)
    n_samples = 31

    factor = sun_earth_distance_factor_averaged(
        d1, d2, n_samples=n_samples, table=table, device=DEVICE, dtype=DTYPE,
    )

    jan1_days = torch.linspace(d1, d2, n_samples, dtype=DTYPE) - 11.0
    expected = (1.0 / sun_earth_distance_au(jan1_days, table=table, device=DEVICE, dtype=DTYPE) ** 2).mean()
    assert_close(factor, expected, name="manual mean of (1AU/R)^2 over window")


def test_sun_earth_distance_factor_averaged_rejects_invalid_window():
    with pytest.raises(ValueError, match="between 0 and 365"):
        sun_earth_distance_factor_averaged(-1.0, 10.0, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError, match="between 0 and 365"):
        sun_earth_distance_factor_averaged(10.0, 400.0, device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError, match="between 0 and 365"):
        sun_earth_distance_factor_averaged(50.0, 10.0, device=DEVICE, dtype=DTYPE)


def test_sun_earth_distance_factor_averaged_full_year_is_close_to_one():
    # Averaging (1 AU / R)^2 over a full orbit should return very close to 1
    # for a near-circular orbit (eccentricity ~0.0167): the modulation must
    # not introduce a spurious net bias over a full year of exposure.
    table = make_synthetic_table()
    factor = sun_earth_distance_factor_averaged(
        0.0, 365.0, n_samples=365, table=table, device=DEVICE, dtype=DTYPE,
    )
    assert float(factor) == pytest.approx(1.0, abs=2e-3)


def test_default_bundled_distance_table_loads_and_gives_plausible_factor():
    # No synthetic override: exercises the real bundled
    # data/solar/geometry/sun_earth_distance.csv end-to-end.
    perihelion = sun_earth_distance_factor("2026-01-04", device=DEVICE, dtype=DTYPE)
    aphelion = sun_earth_distance_factor("2026-07-04", device=DEVICE, dtype=DTYPE)

    assert torch.isfinite(perihelion)
    assert torch.isfinite(aphelion)
    assert float(perihelion) > float(aphelion)
    # Known amplitude of Earth's orbital eccentricity effect: roughly +-3-4%
    # in flux over the year, i.e. the factor should stay within a modest
    # band around 1, not blow up or vanish.
    assert 0.9 < float(aphelion) < 1.0
    assert 1.0 < float(perihelion) < 1.1
