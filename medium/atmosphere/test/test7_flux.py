#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytest-compatible tests for tpeanuts.medium.atmosphere.flux."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.atmosphere.flux import (
    atmosphere_flux_state,
    atmosphere_flux_integrated,
    atmosphere_flux_integrated_angular,
    atmosphere_flux_integrated_height,
)
from tpeanuts.medium.atmosphere.probability import atmosphere_probability_state
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(context: RuntimeContext | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset("_SM_NUFIT52_NO", context=context or make_context())


def make_atmosphere(**overrides) -> AtmosphereParameters:
    values = {
        "atmosphere_density_source": "exponential",
        "nsteps": 48,
        "method": "midpoint",
        "matter": False,
    }
    values.update(overrides)
    return AtmosphereParameters(**values)


def test_atmosphere_flux_equals_probability_times_scalar_flux():
    context = make_context()
    oscillation = make_oscillation(context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)

    probability = atmosphere_probability_state(
        state_mu,
        oscillation,
        E_MeV=1000.0,
        h_km=20.0,
        theta_deg=45.0,
        atmosphere=make_atmosphere(),
        context=context,
    )
    flux = atmosphere_flux_state(
        state_mu,
        oscillation,
        E_MeV=1000.0,
        h_km=20.0,
        theta_deg=45.0,
        flux=2.5,
        atmosphere=make_atmosphere(),
        context=context,
    )

    torch.testing.assert_close(flux, probability * 2.5, atol=1.0e-13, rtol=1.0e-13)


def test_atmosphere_flux_applies_spectrum_factor():
    context = make_context()
    oscillation = make_oscillation(context)
    state_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)

    probability = atmosphere_probability_state(
        state_e,
        oscillation,
        E_MeV=2000.0,
        h_km=30.0,
        theta_deg=60.0,
        atmosphere=make_atmosphere(),
        context=context,
    )
    flux = atmosphere_flux_state(
        state_e,
        oscillation,
        E_MeV=2000.0,
        h_km=30.0,
        theta_deg=60.0,
        flux=2.0,
        spectrum=3.0,
        atmosphere=make_atmosphere(),
        context=context,
    )

    torch.testing.assert_close(flux, probability * 6.0, atol=1.0e-13, rtol=1.0e-13)


def test_atmosphere_flux_broadcasts_energy_flux_and_spectrum():
    context = make_context()
    oscillation = make_oscillation(context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    energy = torch.tensor([500.0, 1000.0, 3000.0], device=DEVICE, dtype=DTYPE)[:, None]
    flux_norm = torch.tensor([10.0, 20.0, 30.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.1, 0.2, 0.4], device=DEVICE, dtype=DTYPE)

    probability = atmosphere_probability_state(
        state_mu,
        oscillation,
        E_MeV=energy,
        h_km=20.0,
        theta_deg=45.0,
        atmosphere=make_atmosphere(),
        context=context,
    )
    flux = atmosphere_flux_state(
        state_mu,
        oscillation,
        E_MeV=energy,
        h_km=20.0,
        theta_deg=45.0,
        flux=flux_norm,
        spectrum=spectrum,
        atmosphere=make_atmosphere(),
        context=context,
    )

    assert flux.shape == (3, 3)
    torch.testing.assert_close(flux, probability * flux_norm[:, None] * spectrum[:, None], atol=1.0e-13, rtol=1.0e-13)


def test_atmosphere_flux_conserves_total_flux_for_normalized_state():
    context = make_context()
    oscillation = make_oscillation(context)
    state_tau = torch.tensor([0.0, 0.0, 1.0], device=DEVICE, dtype=CDTYPE)
    flux_norm = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    flux = atmosphere_flux_state(
        state_tau,
        oscillation,
        E_MeV=torch.tensor([700.0, 1200.0, 2500.0], device=DEVICE, dtype=DTYPE)[:, None],
        h_km=40.0,
        theta_deg=70.0,
        flux=flux_norm,
        atmosphere=make_atmosphere(nsteps=64),
        context=context,
    )

    torch.testing.assert_close(flux.sum(dim=-1), flux_norm, atol=1.0e-12, rtol=1.0e-12)


def test_atmosphere_flux_massbasis_conserves_incoherent_weight_sum():
    context = make_context()
    oscillation = make_oscillation(context)
    weights = torch.tensor([0.2, 0.5, 0.3], device=DEVICE, dtype=DTYPE)

    flux = atmosphere_flux_state(
        weights,
        oscillation,
        E_MeV=1500.0,
        h_km=25.0,
        theta_deg=55.0,
        flux=4.0,
        massbasis=True,
        atmosphere=make_atmosphere(),
        context=context,
    )

    assert flux.shape == (3,)
    torch.testing.assert_close(flux.sum(), torch.tensor(4.0, device=DEVICE, dtype=DTYPE), atol=1.0e-12, rtol=1.0e-12)


def test_atmosphere_flux_zero_normalization_returns_zero_flux():
    context = make_context()
    oscillation = make_oscillation(context)
    state_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)

    flux = atmosphere_flux_state(
        state_e,
        oscillation,
        E_MeV=1000.0,
        h_km=20.0,
        theta_deg=30.0,
        flux=0.0,
        atmosphere=make_atmosphere(),
        context=context,
    )

    torch.testing.assert_close(flux, torch.zeros(3, device=DEVICE, dtype=DTYPE), atol=0.0, rtol=0.0)


def test_atmosphere_flux_rejects_invalid_state_dimension():
    with pytest.raises(ValueError, match="last dimension"):
        atmosphere_flux_state(
            torch.ones(2, device=DEVICE, dtype=CDTYPE),
            make_oscillation(),
            E_MeV=1000.0,
            h_km=20.0,
            theta_deg=30.0,
            flux=1.0,
            atmosphere=make_atmosphere(),
            context=make_context(),
        )


def test_atmosphere_flux_integrated_matches_manual_energy_trapezoid():
    context = make_context()
    oscillation = make_oscillation(context)
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    theta = torch.tensor(65.0, device=DEVICE, dtype=DTYPE)
    h = torch.tensor(20.0, device=DEVICE, dtype=DTYPE)
    E = torch.tensor([500.0, 1000.0, 3000.0], device=DEVICE, dtype=DTYPE)
    flux_norm = torch.tensor([10.0, 20.0, 30.0], device=DEVICE, dtype=DTYPE)
    atmosphere = make_atmosphere(nsteps=16)

    flux_grid = atmosphere_flux_state(
        weights, oscillation, E_MeV=E, h_km=h, theta_deg=theta, flux=flux_norm,
        massbasis=True, atmosphere=atmosphere, context=context, method="analytical",
    )
    expected = torch.trapezoid(flux_grid, x=E, dim=-2)

    result = atmosphere_flux_integrated(
        weights, oscillation, E, h, theta, flux_norm,
        massbasis=True, atmosphere=atmosphere, context=context, method="analytical",
    )

    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)


def test_atmosphere_flux_integrated_angular_matches_manual_trapezoid():
    context = make_context()
    oscillation = make_oscillation(context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    theta = torch.tensor([30.0, 60.0, 90.0], device=DEVICE, dtype=DTYPE)
    atmosphere = make_atmosphere(nsteps=16)

    flux_grid = atmosphere_flux_state(
        state_mu, oscillation, E_MeV=1000.0, h_km=20.0, theta_deg=theta, flux=5.0,
        atmosphere=atmosphere, context=context, method="analytical",
    )
    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)
    expected = 2.0 * torch.pi * torch.trapezoid(flux_grid * sin_theta[:, None], x=theta_rad, dim=-2)

    result = atmosphere_flux_integrated_angular(
        state_mu, oscillation, E_MeV=1000.0, h_km=20.0, theta_deg=theta, flux=5.0,
        atmosphere=atmosphere, context=context, method="analytical",
    )

    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)


def test_atmosphere_flux_integrated_height_matches_weighted_average_times_total_flux():
    context = make_context()
    oscillation = make_oscillation(context)
    state_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=CDTYPE)
    h = torch.tensor([5.0, 20.0, 40.0], device=DEVICE, dtype=DTYPE)
    production_flux = torch.tensor([1.0, 3.0, 2.0], device=DEVICE, dtype=DTYPE)
    atmosphere = make_atmosphere(nsteps=16)

    P = atmosphere_probability_state(
        state_mu, oscillation, E_MeV=1000.0, h_km=h, theta_deg=45.0,
        atmosphere=atmosphere, context=context, method="analytical",
    )
    expected = torch.trapezoid(P * production_flux[:, None], x=h, dim=-2)

    result = atmosphere_flux_integrated_height(
        state_mu, oscillation, E_MeV=1000.0, h_km=h, theta_deg=45.0, production_flux=production_flux,
        atmosphere=atmosphere, context=context, method="analytical",
    )

    torch.testing.assert_close(result, expected, atol=1.0e-12, rtol=1.0e-12)
