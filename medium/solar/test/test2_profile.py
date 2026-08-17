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

"""Pytest-compatible tests for tpeanuts.medium.solar.profile (SolarMediumProfile).

Production-distribution/flux/spectrum tests now live in
``source.solar.test.test2_model`` -- see that module.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from tpeanuts.medium.solar.profile import (
    SolarMediumParameters,
    SolarMediumProfile,
    build_solar_medium,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_same_device(actual: torch.device, expected: torch.device) -> None:
    assert actual.type == torch.device(expected).type


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_medium(*, dtype: torch.dtype = DTYPE) -> SolarMediumProfile:
    device = DEVICE
    radius = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device, dtype=dtype)
    density = torch.tensor([100.0, 50.0, 10.0, 2.0, 1.0], device=device, dtype=dtype)
    return SolarMediumProfile(radius=radius, density=density)


def test_solar_medium_device_dtype_properties_and_string_summary():
    medium = make_medium()
    summary = str(medium)

    assert_same_device(medium.device, DEVICE)
    assert medium.dtype == DTYPE
    assert "SolarMediumProfile" in summary
    assert "n_e=" in summary


def test_electron_density_interpolates_linearly_and_clamps_edges():
    medium = make_medium()
    query = torch.tensor([-0.2, 0.0, 0.125, 0.375, 1.0, 1.2], device=DEVICE, dtype=DTYPE)

    density = medium.electron_density(query)
    expected = torch.tensor([100.0, 100.0, 75.0, 30.0, 1.0, 1.0], device=DEVICE, dtype=DTYPE)

    assert_close(density, expected, name="solar density interpolation")


def test_density_n_defaults_to_none_on_manually_constructed_medium():
    medium = make_medium()

    assert medium.density_n is None


def test_neutron_density_interpolates_linearly_and_clamps_edges():
    medium = make_medium()
    medium.density_n = torch.tensor([50.0, 25.0, 5.0, 1.0, 0.5], device=DEVICE, dtype=DTYPE)
    query = torch.tensor([-0.2, 0.0, 0.125, 0.375, 1.0, 1.2], device=DEVICE, dtype=DTYPE)

    neutron_density = medium.neutron_density(query)
    expected = torch.tensor([50.0, 50.0, 37.5, 15.0, 0.5, 0.5], device=DEVICE, dtype=DTYPE)

    assert_close(neutron_density, expected, name="solar neutron-density interpolation")


def test_neutron_density_raises_when_density_n_is_not_set():
    medium = make_medium()

    with pytest.raises(ValueError, match="density_n is not set"):
        medium.neutron_density(medium.radius)


@pytest.mark.parametrize(
    ("radius", "density", "message"),
    [
        ([0.0, 0.5, 0.5], [10.0, 5.0, 1.0], "strictly increasing"),
        ([0.0, 0.5, 1.0], [10.0, float("nan"), 1.0], "finite and non-negative"),
        ([0.0, 0.5, 1.0], [10.0, -1.0, 1.0], "finite and non-negative"),
    ],
)
def test_construction_rejects_invalid_grid_or_values(radius, density, message):
    with pytest.raises(ValueError, match=message):
        SolarMediumProfile(
            radius=torch.tensor(radius, device=DEVICE, dtype=DTYPE),
            density=torch.tensor(density, device=DEVICE, dtype=DTYPE),
        )


def test_build_solar_medium_returns_existing_medium_when_context_matches():
    medium = make_medium()

    out = build_solar_medium(medium, context=RuntimeContext.resolve(medium.device, medium.dtype))

    assert out is medium


def test_build_solar_medium_casts_existing_medium_to_requested_dtype():
    medium = make_medium(dtype=torch.float64)
    ctx = make_context(dtype=torch.float32)

    out = build_solar_medium(medium, context=ctx)

    assert out is not medium
    assert out.radius.dtype == torch.float32
    assert out.density.dtype == torch.float32
    assert out.density_n is None


def test_build_solar_medium_casts_density_n_when_present():
    medium = make_medium(dtype=torch.float64)
    medium.density_n = torch.tensor(
        [50.0, 25.0, 5.0, 1.0, 0.5], device=DEVICE, dtype=torch.float64,
    )
    ctx = make_context(dtype=torch.float32)

    out = build_solar_medium(medium, context=ctx)

    assert out.density_n is not None
    assert out.density_n.dtype == torch.float32
    assert_close(
        out.density_n,
        medium.density_n.to(dtype=torch.float32),
        name="cast density_n",
    )


def test_solar_medium_default_loads_from_explicit_synthetic_path(tmp_path):
    density_path = tmp_path / "density.csv"
    pd.DataFrame(
        {
            "radius": [0.0, 0.5, 1.0],
            "electron_density_mol_cm3": [100.0, 10.0, 1.0],
            "neutron_density_mol_cm3": [80.0, 8.0, 0.8],
        }
    ).to_csv(density_path, index=False)

    params = SolarMediumParameters(density_path=str(density_path))
    medium = SolarMediumProfile.default(params=params, context=make_context())

    assert_close(medium.radius, torch.tensor([0.0, 0.5, 1.0], device=DEVICE, dtype=DTYPE), name="default synthetic radius")
    assert_close(medium.density, torch.tensor([100.0, 10.0, 1.0], device=DEVICE, dtype=DTYPE), name="default synthetic density")


def test_build_solar_medium_loads_default_when_medium_is_none():
    medium = build_solar_medium(None, context=make_context())

    assert medium.radius.ndim == 1
    assert medium.density.shape == medium.radius.shape
    assert bool(torch.all(torch.diff(medium.radius) > 0.0))
    assert bool(torch.all(medium.radius >= 0.0))
    assert bool(torch.all(medium.radius <= 1.0))
    assert bool(torch.all(medium.density > 0.0))
    # The default Zenodo provider supplies density_n directly. Near the
    # photosphere n_n can exceed the free-electron density because the plasma
    # is no longer fully ionized, so no global n_n <= n_e inequality applies.
    assert medium.density_n is not None
    assert medium.density_n.shape == medium.radius.shape
    assert torch.isfinite(medium.density_n).all()
    assert bool(torch.all(medium.density_n > 0.0))
