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

from tpeanuts.core.common.flux import flux_integrated
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.medium.solar.flux import solar_flux_integrated, solar_flux_state
from tpeanuts.medium.solar.geometry import sun_earth_distance_factor
from tpeanuts.medium.solar.profile import build_solar_profile
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation() -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset("_SM_NUFIT52_NO", context=make_context())


def make_profile():
    return build_solar_profile(None, context=make_context())


def test_solar_flux_single_source_uses_profile_spectrum_by_default():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probability = solar_probability_state(oscillation, energy, profile, "8B")
    flux = solar_flux_state("8B", profile, oscillation, energy)
    expected = probability * profile.flux("8B") * profile.spectrum("8B", energy)[:, None]

    assert flux.shape == (3, 3)
    assert torch.isfinite(flux).all()
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_single_source_with_spectrum_broadcasts_over_energy():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    probability = solar_probability_state(oscillation, energy, profile, "8B")
    flux = solar_flux_state("8B", profile, oscillation, energy, source_spectrum=spectrum)
    expected = probability * profile.flux("8B") * spectrum[:, None]

    assert flux.shape == (3, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_multiple_sources_uses_ordered_profile_spectra():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "8B", "hep")

    probability = solar_probability_state(oscillation, energy, profile, sources)
    fluxes = torch.stack([profile.flux(source) for source in sources], dim=0)
    flux = solar_flux_state(sources, profile, oscillation, energy)
    expected = probability * fluxes[:, None, None] * profile.spectrum(sources, energy)[:, :, None]

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

    probability = solar_probability_state(oscillation, energy, profile, sources)
    fluxes = torch.stack([profile.flux(source) for source in sources], dim=0)
    flux = solar_flux_state(sources, profile, oscillation, energy, source_spectrum=spectrum)
    expected = probability * fluxes[:, None, None] * spectrum[:, :, None]

    assert flux.shape == (2, 2, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_sums_to_differential_source_flux_by_default():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    flux = solar_flux_state("8B", profile, oscillation, energy)

    expected_total = profile.flux("8B") * profile.spectrum("8B", energy)
    torch.testing.assert_close(flux.sum(dim=-1), expected_total, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_unknown_source_raises_key_error():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor(5.0, device=DEVICE, dtype=DTYPE)

    with pytest.raises(KeyError):
        solar_flux_state("not_a_source", profile, oscillation, energy)


def test_solar_flux_state_date_none_leaves_flux_at_1au_reference():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    no_date = solar_flux_state("8B", profile, oscillation, energy)
    explicit_none = solar_flux_state("8B", profile, oscillation, energy, date=None)

    torch.testing.assert_close(no_date, explicit_none, rtol=1.0e-14, atol=1.0e-14)


def test_solar_flux_state_date_applies_sun_earth_distance_factor():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    reference = solar_flux_state("8B", profile, oscillation, energy)
    on_date = solar_flux_state("8B", profile, oscillation, energy, date="2026-01-04")
    factor = sun_earth_distance_factor("2026-01-04", device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(on_date, reference * factor, rtol=1.0e-13, atol=1.0e-13)
    # Perihelion (early January): Earth is closer to the Sun, so the flux
    # received must be higher than the 1 AU reference.
    assert bool(torch.all(on_date > reference))


def test_solar_flux_integrated_date_applies_sun_earth_distance_factor():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    reference = solar_flux_integrated("8B", profile, oscillation, energy, spectrum)
    on_date = solar_flux_integrated("8B", profile, oscillation, energy, spectrum, date="2026-01-04")
    factor = sun_earth_distance_factor("2026-01-04", device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(on_date, reference * factor, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_integrated_uses_profile_spectrum_by_default():
    # The normalized spectral density is resolved from SolarProfile when the
    # caller does not provide an explicit override.
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    automatic = solar_flux_integrated("8B", profile, oscillation, energy)
    explicit = solar_flux_integrated(
        "8B", profile, oscillation, energy, profile.spectrum("8B", energy)
    )
    torch.testing.assert_close(automatic, explicit)


def test_solar_flux_integrated_matches_manual_energy_integration():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    rate = solar_flux_integrated("8B", profile, oscillation, energy, spectrum)

    flux_grid = solar_flux_state("8B", profile, oscillation, energy, spectrum)
    expected = flux_integrated(flux_grid, energy, energy_dim=0)

    assert rate.shape == (3,)
    torch.testing.assert_close(rate, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_integrated_is_stable_under_energy_grid_refinement():
    # Regression test for the bug this replaces: omitting source_spectrum
    # made solar_flux_integrated multiply the *total* source flux by P(E)
    # and integrate that over the energy grid, so the result scaled with the
    # arbitrary grid spacing/range instead of converging to a fixed physical
    # rate. With an explicit, normalized spectrum, a coarse and a fine grid
    # over the same physical energy range should agree.
    oscillation = make_oscillation()
    profile = make_profile()

    def flat_normalized_spectrum(energy: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(energy)
        return weights / torch.trapezoid(weights, x=energy)

    energy_coarse = torch.linspace(1.0, 10.0, 5, device=DEVICE, dtype=DTYPE)
    energy_fine = torch.linspace(1.0, 10.0, 41, device=DEVICE, dtype=DTYPE)

    rate_coarse = solar_flux_integrated(
        "8B", profile, oscillation, energy_coarse,
        flat_normalized_spectrum(energy_coarse),
    )
    rate_fine = solar_flux_integrated(
        "8B", profile, oscillation, energy_fine,
        flat_normalized_spectrum(energy_fine),
    )

    torch.testing.assert_close(rate_coarse, rate_fine, rtol=5.0e-2, atol=1.0e-6)
