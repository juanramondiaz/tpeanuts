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

"""Pytest-compatible tests for solar Landau-Zener corrections."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.solar.landau_zener import density_gradient, plz, resonance_radius
from tpeanuts.medium.solar.matter_mixing import Vk
from tpeanuts.medium.solar.profile import build_solar_profile
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class SimpleProfile:
    radius: torch.Tensor
    density: torch.Tensor


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(
    *,
    antinu: bool | torch.Tensor = False,
    theta12: float = 0.59,
    dtype: torch.dtype = DTYPE,
) -> OscillationParameters:
    ctx = make_context(dtype)
    pmns = PMNS_SM(PMNSParams(theta12=theta12, theta13=0.15, theta23=0.78, delta=1.20, context=ctx))
    mass_spectrum = MassSpectrum_SM(
        DeltamSq21=torch.as_tensor(7.42e-5, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(2.517e-3, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=antinu)


def resonance_density(oscillation: OscillationParameters, energy_mev: float) -> torch.Tensor:
    energy = torch.tensor(energy_mev, device=DEVICE, dtype=DTYPE)
    unit_density = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)
    cos2theta12 = torch.cos(2.0 * oscillation.pmns.params.theta12)
    vk_per_density = Vk(oscillation.mass_spectrum.DeltamSq21, energy, unit_density)
    return cos2theta12 / vk_per_density


def make_linear_resonance_profile(
    oscillation: OscillationParameters,
    *,
    energy_mev: float = 10.0,
    resonance_r: float = 0.4,
    slope_factor: float = 1.0,
) -> SimpleProfile:
    radius = torch.linspace(0.0, 1.0, 101, device=DEVICE, dtype=DTYPE)
    ne_res = resonance_density(oscillation, energy_mev)
    density = ne_res + slope_factor * ne_res * (resonance_r - radius)
    return SimpleProfile(radius=radius, density=density)


def test_density_gradient_matches_linear_profile_derivative():
    radius = torch.linspace(0.0, 1.0, 11, device=DEVICE, dtype=DTYPE)
    density = 12.0 - 5.0 * radius
    profile = SimpleProfile(radius=radius, density=density)

    gradient = density_gradient(profile)

    expected = torch.full_like(radius, -5.0)
    torch.testing.assert_close(gradient, expected, rtol=1.0e-14, atol=1.0e-14)


def test_resonance_radius_recovers_known_linear_crossing_for_scalar_and_grid():
    oscillation = make_oscillation()
    profile = make_linear_resonance_profile(oscillation, energy_mev=10.0, resonance_r=0.4)

    scalar_radius = resonance_radius(oscillation, torch.tensor(10.0, device=DEVICE, dtype=DTYPE), profile)
    energy_grid = torch.tensor([5.0, 10.0, 20.0], device=DEVICE, dtype=DTYPE)
    grid_radius = resonance_radius(oscillation, energy_grid, profile)

    expected_grid = torch.tensor([float("nan"), 0.4, 0.9], device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(scalar_radius, torch.tensor(0.4, device=DEVICE, dtype=DTYPE), rtol=1.0e-13, atol=1.0e-13)
    assert torch.isnan(grid_radius[0])
    torch.testing.assert_close(grid_radius[1:], expected_grid[1:], rtol=1.0e-13, atol=1.0e-13)


def test_resonance_radius_returns_nan_without_physical_solar_resonance():
    oscillation = make_oscillation()
    profile = make_linear_resonance_profile(oscillation, energy_mev=10.0, resonance_r=0.4)
    low_energy = torch.tensor(0.01, device=DEVICE, dtype=DTYPE)

    no_res_low_energy = resonance_radius(oscillation, low_energy, profile)
    no_res_antinu = resonance_radius(make_oscillation(antinu=True), torch.tensor(10.0, device=DEVICE, dtype=DTYPE), profile)
    no_res_lma_dark = resonance_radius(make_oscillation(theta12=1.0), torch.tensor(10.0, device=DEVICE, dtype=DTYPE), profile)

    assert torch.isnan(no_res_low_energy)
    assert torch.isnan(no_res_antinu)
    assert torch.isnan(no_res_lma_dark)


def test_plz_is_zero_when_no_resonance_exists():
    oscillation = make_oscillation()
    profile = make_linear_resonance_profile(oscillation, energy_mev=10.0, resonance_r=0.4)
    energies = torch.tensor([0.001, 0.01], device=DEVICE, dtype=DTYPE)

    probabilities = plz(oscillation, energies, profile)

    torch.testing.assert_close(probabilities, torch.zeros_like(energies), rtol=0.0, atol=0.0)


def test_plz_shape_bounds_and_scalar_vector_consistency():
    oscillation = make_oscillation()
    profile = make_linear_resonance_profile(
        oscillation,
        energy_mev=10.0,
        resonance_r=0.4,
        slope_factor=1.0e15,
    )
    scalar_energy = torch.tensor(10.0, device=DEVICE, dtype=DTYPE)
    energy_grid = torch.tensor([5.0, 10.0], device=DEVICE, dtype=DTYPE)

    scalar_probability = plz(oscillation, scalar_energy, profile)
    grid_probability = plz(oscillation, energy_grid, profile)

    assert scalar_probability.ndim == 0
    assert grid_probability.shape == energy_grid.shape
    assert torch.isfinite(grid_probability).all()
    assert torch.all((grid_probability >= 0.0) & (grid_probability <= 1.0))
    torch.testing.assert_close(scalar_probability, grid_probability[1], rtol=1.0e-14, atol=1.0e-14)


def test_standard_solar_profile_lma_is_fully_adiabatic_to_float_precision():
    context = make_context()
    oscillation = PropagationConfig.oscillation_parameters_from_preset("_SM_NUFIT52_NO", context=context)
    profile = build_solar_profile(None, context=context)
    energies = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probabilities = plz(oscillation, energies, profile)

    assert probabilities.shape == energies.shape
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities, torch.zeros_like(probabilities), rtol=0.0, atol=0.0)
