#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.profile."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.medium.atmosphere.geometry import atmosphere_path_length
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters, AtmosphereProfile
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_params(**overrides) -> AtmosphereParameters:
    kwargs = {
        "atmosphere_density_source": "exponential",
        "atmosphere_density_kwargs": {"rho0_gcm3": 1.2e-3, "scale_height_km": 7.5},
        "nsteps": 16,
        "method": "midpoint",
        "matter": True,
        "evolution_scale_m": R_E,
    }
    kwargs.update(overrides)
    return AtmosphereParameters(**kwargs)


def test_atmosphere_profile_shapes_and_metadata():
    context = make_context()
    params = make_params(nsteps=20)

    profile = AtmosphereProfile(20.0, 30.0, 2.0, params=params, context=context)

    assert profile.x.shape == (21,)
    assert profile.dx_evolution.shape == (20,)
    assert profile.altitude_km.shape == (20,)
    assert profile.n_e_molcm3.shape == (20,)
    assert profile.trajectory.meta["kind"] == "atmosphere"
    assert profile.atmosphere_density_source == "exponential"
    assert profile.device.type == DEVICE.type
    assert profile.dtype == DTYPE


def test_atmosphere_profile_path_grid_matches_geometry_length():
    context = make_context()
    params = make_params(nsteps=12)
    profile = AtmosphereProfile(15.0, 45.0, 1.0, params=params, context=context)
    expected_L_atm = atmosphere_path_length(15.0, 45.0, 1.0, device=DEVICE, dtype=DTYPE)
    expected_x_end = expected_L_atm / (torch.as_tensor(R_E, device=DEVICE, dtype=DTYPE) / 1.0e3)

    torch.testing.assert_close(profile.L_atm_km, expected_L_atm, rtol=1.0e-13, atol=1.0e-10)
    torch.testing.assert_close(profile.x[0], torch.tensor(0.0, device=DEVICE, dtype=DTYPE), rtol=0.0, atol=0.0)
    torch.testing.assert_close(profile.x[-1], expected_x_end, rtol=1.0e-13, atol=1.0e-13)
    torch.testing.assert_close(profile.dx_evolution.sum(), profile.x[-1] - profile.x[0], rtol=1.0e-13, atol=1.0e-13)


def test_atmosphere_profile_altitude_samples_are_inside_atmosphere_segment():
    context = make_context()
    params = make_params(nsteps=32, method="midpoint")
    h_km = torch.tensor(25.0, device=DEVICE, dtype=DTYPE)

    profile = AtmosphereProfile(h_km, 60.0, 2.0, params=params, context=context)

    assert torch.all(profile.altitude_km > 0.0)
    assert torch.all(profile.altitude_km < h_km)
    assert torch.all(torch.diff(profile.altitude_km) > 0.0)


@pytest.mark.parametrize("method", ["left", "midpoint", "right", None])
def test_atmosphere_profile_sampling_methods_have_expected_sample_counts(method):
    context = make_context()
    params = make_params(nsteps=8, method=method)

    profile = AtmosphereProfile(12.0, 20.0, 0.0, params=params, context=context)

    assert profile.trajectory.sample_x.shape == (8,)
    assert profile.altitude_km.shape == (8,)


def test_atmosphere_profile_matter_false_returns_zero_electron_density():
    context = make_context()
    params = make_params(matter=False)

    profile = AtmosphereProfile(20.0, 30.0, 2.0, params=params, context=context)

    torch.testing.assert_close(profile.n_e_molcm3, torch.zeros_like(profile.n_e_molcm3), rtol=0.0, atol=0.0)


def test_atmosphere_profile_exponential_density_decreases_with_altitude():
    context = make_context()
    params = make_params(nsteps=64)

    profile = AtmosphereProfile(40.0, 0.0, 0.0, params=params, context=context)

    assert torch.all(profile.n_e_molcm3 > 0.0)
    assert torch.all(torch.diff(profile.n_e_molcm3) < 0.0)


def test_atmosphere_profile_broadcasts_angle_and_height_inputs():
    context = make_context()
    params = make_params(nsteps=10)
    h = torch.tensor([10.0, 20.0], device=DEVICE, dtype=DTYPE)
    theta = torch.tensor([0.0, 60.0], device=DEVICE, dtype=DTYPE)

    profile = AtmosphereProfile(h, theta, 0.0, params=params, context=context)

    assert profile.x.shape == (2, 11)
    assert profile.altitude_km.shape == (2, 10)
    assert profile.n_e_molcm3.shape == (2, 10)


def test_atmosphere_profile_neutron_density_defaults_to_none():
    context = make_context()
    params = make_params(nsteps=10)

    profile = AtmosphereProfile(20.0, 30.0, 2.0, params=params, context=context)

    assert profile.include_matter_nc is False
    assert profile.n_n_molcm3 is None


def test_atmosphere_profile_neutron_density_matches_complementary_fraction():
    context = make_context()
    params = make_params(nsteps=16, include_matter_nc=True, atmosphere_density_kwargs={"rho0_gcm3": 1.2e-3, "scale_height_km": 7.5, "Ye": 0.494})

    profile = AtmosphereProfile(20.0, 30.0, 2.0, params=params, context=context)

    assert profile.n_n_molcm3 is not None
    assert profile.n_n_molcm3.shape == profile.n_e_molcm3.shape
    total = profile.n_e_molcm3 / 0.494
    torch.testing.assert_close(profile.n_n_molcm3, total * (1.0 - 0.494), rtol=1.0e-12, atol=1.0e-16)


def test_atmosphere_profile_neutron_density_zero_when_matter_false():
    context = make_context()
    params = make_params(matter=False, include_matter_nc=True)

    profile = AtmosphereProfile(20.0, 30.0, 2.0, params=params, context=context)

    assert profile.n_n_molcm3 is not None
    torch.testing.assert_close(profile.n_n_molcm3, torch.zeros_like(profile.n_n_molcm3), rtol=0.0, atol=0.0)


def test_atmosphere_profile_rejects_invalid_nsteps_and_scale():
    context = make_context()

    with pytest.raises(ValueError):
        AtmosphereProfile(10.0, 0.0, 0.0, params=make_params(nsteps=0), context=context)
    with pytest.raises(ValueError):
        AtmosphereProfile(10.0, 0.0, 0.0, params=make_params(evolution_scale_m=0.0), context=context)
