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

"""Pytest-compatible tests for tpeanuts.medium.solar.profile."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from tpeanuts.coherent.coordinates import (
    distance_to_solar_radius_fraction,
    production_to_surface_path_length,
    solar_path_grid,
    solar_radius_fraction_to_distance,
    solar_shell_widths,
)
from tpeanuts.medium.solar.profile import (
    SolarParameters,
    SolarProfile,
    build_solar_profile,
)
from tpeanuts.util.constant import R_SUN, R_SUN_KM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same_device(actual: torch.device, expected: torch.device) -> None:
    assert actual.type == torch.device(expected).type


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_profile(*, dtype: torch.dtype = DTYPE) -> SolarProfile:
    device = DEVICE
    radius = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=dtype)
    density = torch.tensor([100.0, 50.0, 10.0, 2.0, 1.0], device=device, dtype=dtype)
    fractions = {
        "pp": torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0], device=device, dtype=dtype),
        "8B": torch.tensor([4.0, 2.0, 0.5, 0.0, 0.0], device=device, dtype=dtype),
        "hep": torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0], device=device, dtype=dtype),
    }
    fluxes = {
        "pp": torch.tensor(6.0e10, device=device, dtype=dtype),
        "8B": torch.tensor(4.5e6, device=device, dtype=dtype),
        "hep": torch.tensor(8.0e3, device=device, dtype=dtype),
    }
    return SolarProfile(radius=radius, density=density, fractions=fractions, fluxes=fluxes)


def test_solar_profile_device_dtype_properties_and_string_summary():
    profile = make_profile()
    summary = str(profile)

    assert_same_device(profile.device, DEVICE)
    assert profile.dtype == DTYPE
    assert "SolarProfile" in summary
    assert "sources=" in summary
    assert "use_LZ=False" in summary


def test_electron_density_interpolates_linearly_and_clamps_edges():
    profile = make_profile()
    query = torch.tensor([-0.2, 0.0, 0.125, 0.375, 1.0, 1.2], device=DEVICE, dtype=DTYPE)

    density = profile.electron_density(query)
    expected = torch.tensor([100.0, 100.0, 75.0, 30.0, 1.0, 1.0], device=DEVICE, dtype=DTYPE)

    assert_close(density, expected, name="solar density interpolation")


def test_production_fraction_returns_grid_values_and_interpolates_query_points():
    profile = make_profile()
    query = torch.tensor([0.125, 0.375], device=DEVICE, dtype=DTYPE)

    grid_fraction = profile.production_fraction("pp")
    query_fraction = profile.production_fraction("pp", query)

    assert_close(grid_fraction, profile.fractions["pp"], name="production fraction on grid")
    assert_close(query_fraction, torch.tensor([0.5, 1.5], device=DEVICE, dtype=DTYPE), name="production fraction interpolation")


def test_production_fraction_unknown_source_raises():
    profile = make_profile()

    with pytest.raises(KeyError, match="Unknown solar source"):
        profile.production_fraction("unknown")


def test_source_fractions_single_and_multiple_sources():
    profile = make_profile()

    single = profile.source_fractions("pp")
    multiple = profile.source_fractions(["pp", "8B"])

    assert single.shape == profile.radius.shape
    assert multiple.shape == (2, profile.radius.numel())
    assert_close(multiple[0], profile.fractions["pp"], name="stacked pp")
    assert_close(multiple[1], profile.fractions["8B"], name="stacked 8B")


def test_normalized_fraction_integrates_to_one_and_is_nonnegative():
    profile = make_profile()

    normalized = profile.normalized_fraction("pp")
    area = torch.trapz(normalized, x=profile.radius)

    assert bool(torch.all(normalized >= 0.0))
    assert_close(area, torch.tensor(1.0, device=DEVICE, dtype=DTYPE), name="normalized production fraction area")


def test_flux_returns_scalar_and_unknown_source_raises():
    profile = make_profile()

    flux = profile.flux("8B")

    assert flux.shape == ()
    assert_close(flux, torch.tensor(4.5e6, device=DEVICE, dtype=DTYPE), name="8B flux")
    with pytest.raises(KeyError, match="Unknown solar flux source"):
        profile.flux("unknown")


def test_build_solar_profile_returns_existing_profile_when_context_matches():
    profile = make_profile()

    out = build_solar_profile(profile, context=RuntimeContext.resolve(profile.device, profile.dtype))

    assert out is profile


def test_build_solar_profile_casts_existing_profile_to_requested_dtype():
    profile = make_profile(dtype=torch.float64)
    ctx = make_context(dtype=torch.float32)

    out = build_solar_profile(profile, context=ctx)

    assert out is not profile
    assert out.radius.dtype == torch.float32
    assert out.density.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in out.fractions.values())
    assert all(value.dtype == torch.float32 for value in out.fluxes.values())


def test_solar_profile_default_loads_from_explicit_synthetic_paths(tmp_path):
    model_path = tmp_path / "model.csv"
    flux_path = tmp_path / "fluxes.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "density_log_10": [2.0, 1.0, 0.0],
            "pp fraction": [0.0, 1.0, 0.0],
            "8B fraction": [1.0, 0.0, 0.0],
        }
    ).to_csv(model_path, index=False)
    pd.DataFrame(
        {
            "fraction": ["pp", "8B"],
            "flux": [6.0e10, 4.5e6],
        }
    ).to_csv(flux_path, index=False)

    params = SolarParameters(model_path=str(model_path), fluxes_path=str(flux_path))
    profile = SolarProfile.default(params=params, context=make_context())

    assert_close(profile.radius, torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE), name="default synthetic radius")
    assert_close(profile.density, torch.tensor([100.0, 10.0, 1.0], device=DEVICE, dtype=DTYPE), name="default synthetic density")
    assert sorted(profile.fractions) == ["8B", "pp"]
    assert sorted(profile.fluxes) == ["8B", "pp"]


def test_build_solar_profile_loads_default_when_profile_is_none():
    profile = build_solar_profile(None, context=make_context())

    assert profile.radius.ndim == 1
    assert profile.density.shape == profile.radius.shape
    assert bool(torch.all(torch.diff(profile.radius) > 0.0))
    assert bool(torch.all(profile.radius >= 0.0))
    assert bool(torch.all(profile.radius <= 1.0))
    assert bool(torch.all(profile.density > 0.0))
    assert {"pp", "8B", "7Be", "hep"}.issubset(profile.fractions)
    assert {"pp", "8B", "7Be", "hep"}.issubset(profile.fluxes)


def test_solar_radius_fraction_distance_roundtrip_m_and_km():
    ctx = make_context()
    rho = torch.tensor([0.0, 0.05, 0.25, 0.70, 1.0], device=DEVICE, dtype=DTYPE)

    radius_m = solar_radius_fraction_to_distance(rho, unit="m", context=ctx)
    radius_km = solar_radius_fraction_to_distance(rho, unit="km", context=ctx)
    rho_from_m = distance_to_solar_radius_fraction(radius_m, unit="m", context=ctx)
    rho_from_km = distance_to_solar_radius_fraction(radius_km, unit="km", context=ctx)

    assert_close(radius_m / 1.0e3, radius_km, name="solar radius m/km conversion", atol=1.0e-8, rtol=1.0e-12)
    assert_close(rho_from_m, rho, name="rho roundtrip through meters")
    assert_close(rho_from_km, rho, name="rho roundtrip through kilometers")


def test_production_to_surface_path_length_matches_one_minus_rho():
    ctx = make_context()
    rho0 = torch.tensor([0.0, 0.10, 0.50, 0.95, 1.0], device=DEVICE, dtype=DTYPE)

    length_m = production_to_surface_path_length(rho0, unit="m", context=ctx)
    length_km = production_to_surface_path_length(rho0, unit="km", context=ctx)
    expected_m = (1.0 - rho0) * R_SUN

    assert_close(length_m, expected_m, name="production-to-surface path length")
    assert_close(length_m / 1.0e3, length_km, name="production path m/km conversion", atol=1.0e-8, rtol=1.0e-12)
    assert_close(length_km[0], torch.tensor(R_SUN_KM, device=DEVICE, dtype=DTYPE), name="central production length")
    assert_close(length_km[-1], torch.tensor(0.0, device=DEVICE, dtype=DTYPE), name="surface production length")


def test_solar_path_grid_aligned_with_profile_radius_and_widths_sum_to_path():
    profile = make_profile()
    ctx = make_context()
    rho0 = torch.tensor(0.30, device=DEVICE, dtype=DTYPE)

    grid = solar_path_grid(rho0, profile_radius=profile.radius, context=ctx)
    widths_km = solar_shell_widths(grid, unit="km", context=ctx)
    total_km = torch.sum(widths_km)
    expected_km = production_to_surface_path_length(rho0, unit="km", context=ctx)

    assert_close(grid, torch.tensor([0.30, 0.50, 0.75, 1.00], device=DEVICE, dtype=DTYPE), name="profile-aligned solar path grid")
    assert bool(torch.all(widths_km >= 0.0))
    assert_close(total_km, expected_km, name="profile-aligned shell widths sum to path", atol=1.0e-8, rtol=1.0e-12)


def test_solar_path_grid_even_spacing_and_validation_errors():
    ctx = make_context()
    grid = solar_path_grid(0.25, nsteps=3, context=ctx)

    assert_close(grid, torch.tensor([0.25, 0.50, 0.75, 1.00], device=DEVICE, dtype=DTYPE), name="even solar path grid")

    with pytest.raises(ValueError, match="rho0"):
        solar_path_grid(torch.tensor([0.1, 0.2], device=DEVICE, dtype=DTYPE), context=ctx)

    with pytest.raises(ValueError, match="nsteps"):
        solar_path_grid(0.1, nsteps=0, context=ctx)

    with pytest.raises(ValueError, match="monotonically increasing"):
        solar_shell_widths(torch.tensor([0.0, 0.5, 0.4], device=DEVICE, dtype=DTYPE), context=ctx)
