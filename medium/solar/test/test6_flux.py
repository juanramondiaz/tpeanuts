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

"""Pytest-compatible tests for solar flux helpers."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.flux import solar_flux
from tpeanuts.medium.solar.profile import build_solar_profile
from tpeanuts.medium.solar.probability import psolar
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation() -> OscillationParameters:
    return OscillationParameters.from_preset("_SM_NUFIT52_NO", context=make_context())


def make_profile():
    return build_solar_profile(None, context=make_context())


def test_solar_flux_single_source_equals_probability_times_total_flux():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probability = psolar(oscillation, energy, profile, "8B")
    flux = solar_flux("8B", profile, oscillation, energy)
    expected = probability * profile.flux("8B")

    assert flux.shape == (3, 3)
    assert torch.isfinite(flux).all()
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_single_source_with_spectrum_broadcasts_over_energy():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    probability = psolar(oscillation, energy, profile, "8B")
    flux = solar_flux("8B", profile, oscillation, energy, source_spectrum=spectrum)
    expected = probability * profile.flux("8B") * spectrum[:, None]

    assert flux.shape == (3, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_multiple_sources_preserves_source_order_and_flux_normalization():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "8B", "hep")

    probability = psolar(oscillation, energy, profile, sources)
    fluxes = torch.stack([profile.flux(source) for source in sources], dim=0)
    flux = solar_flux(sources, profile, oscillation, energy)
    expected = probability * fluxes[:, None, None]

    assert flux.shape == (3, 2, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_multiple_sources_with_source_energy_spectrum():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)
    sources = ("8B", "hep")
    spectrum = torch.tensor(
        [
            [0.25, 0.10],
            [0.02, 0.04],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    probability = psolar(oscillation, energy, profile, sources)
    fluxes = torch.stack([profile.flux(source) for source in sources], dim=0)
    flux = solar_flux(sources, profile, oscillation, energy, source_spectrum=spectrum)
    expected = probability * fluxes[:, None, None] * spectrum[:, :, None]

    assert flux.shape == (2, 2, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_sums_to_source_flux_when_no_spectrum_is_supplied():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    flux = solar_flux("8B", profile, oscillation, energy)

    expected_total = profile.flux("8B").expand_as(energy)
    torch.testing.assert_close(flux.sum(dim=-1), expected_total, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_unknown_source_raises_key_error():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor(5.0, device=DEVICE, dtype=DTYPE)

    with pytest.raises(KeyError):
        solar_flux("not_a_source", profile, oscillation, energy)
