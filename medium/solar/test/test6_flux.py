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
from tpeanuts.medium.vacuum.solar_geometry import sun_earth_distance_factor
from tpeanuts.medium.solar.profile import build_solar_medium
from tpeanuts.medium.solar.probability import solar_probability_state
from tpeanuts.source.solar import build_solar_source
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation() -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset("_SM_NUFIT52_NO", context=make_context())


def make_medium_source():
    context = make_context()
    return build_solar_medium(None, context=context), build_solar_source(None, context=context)


def test_solar_flux_single_source_uses_source_spectrum_by_default():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    probability = solar_probability_state(oscillation, energy, medium, source, "8B")
    flux = solar_flux_state(oscillation, energy, medium, source, "8B")
    expected = probability * source.total_flux("8B") * source.spectrum("8B", energy)[:, None]

    assert flux.shape == (3, 3)
    assert torch.isfinite(flux).all()
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_single_source_with_spectrum_broadcasts_over_energy():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    probability = solar_probability_state(oscillation, energy, medium, source, "8B")
    flux = solar_flux_state(oscillation, energy, medium, source, "8B", spectrum)
    expected = probability * source.total_flux("8B") * spectrum[:, None]

    assert flux.shape == (3, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_multiple_sources_uses_ordered_source_spectra():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "8B", "hep")

    probability = solar_probability_state(oscillation, energy, medium, source, sources)
    fluxes = torch.stack([source.total_flux(key) for key in sources], dim=0)
    flux = solar_flux_state(oscillation, energy, medium, source, sources)
    expected = probability * fluxes[:, None, None] * source.spectrum(sources, energy)[:, :, None]

    assert flux.shape == (3, 2, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_multiple_sources_with_source_energy_spectrum():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
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

    probability = solar_probability_state(oscillation, energy, medium, source, sources)
    fluxes = torch.stack([source.total_flux(key) for key in sources], dim=0)
    flux = solar_flux_state(oscillation, energy, medium, source, sources, spectrum)
    expected = probability * fluxes[:, None, None] * spectrum[:, :, None]

    assert flux.shape == (2, 2, 3)
    torch.testing.assert_close(flux, expected, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_sums_to_differential_source_flux_by_default():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    flux = solar_flux_state(oscillation, energy, medium, source, "8B")

    expected_total = source.total_flux("8B") * source.spectrum("8B", energy)
    torch.testing.assert_close(flux.sum(dim=-1), expected_total, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_unknown_source_raises_key_error():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor(5.0, device=DEVICE, dtype=DTYPE)

    with pytest.raises(KeyError):
        solar_flux_state(oscillation, energy, medium, source, "not_a_source")


def test_solar_flux_state_date_none_leaves_flux_at_1au_reference():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    no_date = solar_flux_state(oscillation, energy, medium, source, "8B")
    explicit_none = solar_flux_state(oscillation, energy, medium, source, "8B", date=None)

    torch.testing.assert_close(no_date, explicit_none, rtol=1.0e-14, atol=1.0e-14)


def test_solar_flux_state_date_applies_sun_earth_distance_factor():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    reference = solar_flux_state(oscillation, energy, medium, source, "8B")
    on_date = solar_flux_state(oscillation, energy, medium, source, "8B", date="2026-01-04")
    factor = sun_earth_distance_factor("2026-01-04", device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(on_date, reference * factor, rtol=1.0e-13, atol=1.0e-13)
    # Perihelion (early January): Earth is closer to the Sun, so the flux
    # received must be higher than the 1 AU reference.
    assert bool(torch.all(on_date > reference))


def test_solar_flux_integrated_date_applies_sun_earth_distance_factor():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    reference = solar_flux_integrated(oscillation, energy, medium, source, "8B", spectrum)
    on_date = solar_flux_integrated(oscillation, energy, medium, source, "8B", spectrum, date="2026-01-04")
    factor = sun_earth_distance_factor("2026-01-04", device=DEVICE, dtype=DTYPE)

    torch.testing.assert_close(on_date, reference * factor, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_integrated_uses_source_spectrum_by_default():
    # The normalized spectral density is resolved from SolarNeutrinoSource
    # when the caller does not provide an explicit override.
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    automatic = solar_flux_integrated(oscillation, energy, medium, source, "8B")
    table = source.spectrum_table("8B")
    explicit = solar_flux_integrated(
        oscillation, table.energy_MeV, medium, source, "8B", table.density_MeV_inverse
    )
    torch.testing.assert_close(automatic, explicit)


def test_solar_flux_integrated_matches_manual_energy_integration():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    spectrum = torch.tensor([0.2, 0.5, 0.1], device=DEVICE, dtype=DTYPE)

    rate = solar_flux_integrated(oscillation, energy, medium, source, "8B", spectrum)

    flux_grid = solar_flux_state(oscillation, energy, medium, source, "8B", spectrum)
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
    medium, source = make_medium_source()

    def flat_normalized_spectrum(energy: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(energy)
        return weights / torch.trapezoid(weights, x=energy)

    energy_coarse = torch.linspace(1.0, 10.0, 5, device=DEVICE, dtype=DTYPE)
    energy_fine = torch.linspace(1.0, 10.0, 41, device=DEVICE, dtype=DTYPE)

    rate_coarse = solar_flux_integrated(
        oscillation, energy_coarse, medium, source, "8B",
        flat_normalized_spectrum(energy_coarse),
    )
    rate_fine = solar_flux_integrated(
        oscillation, energy_fine, medium, source, "8B",
        flat_normalized_spectrum(energy_fine),
    )

    torch.testing.assert_close(rate_coarse, rate_fine, rtol=5.0e-2, atol=1.0e-6)


def test_line_flux_integral_is_a_weighted_sum_independent_of_external_grid():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    lines = source.spectrum_table("7Be")
    result = solar_flux_integrated(
        oscillation,
        torch.linspace(0.0, 2.0, 17, device=DEVICE, dtype=DTYPE),
        medium,
        source,
        "7Be",
    )
    probability = solar_probability_state(
        oscillation, lines.energy_MeV, medium, source, "7Be",
    )
    expected = source.total_flux("7Be") * (
        probability * lines.weights[:, None]
    ).sum(dim=0)
    torch.testing.assert_close(result, expected)


def test_solar_flux_state_auto_detects_line_source_at_its_own_energies():
    # solar_flux_state used to unconditionally call source.spectrum(...),
    # which raises TypeError for a line source. It must now auto-detect the
    # line spectrum the same way solar_flux_integrated already does, and
    # summing the per-energy (per-line) result must reproduce
    # solar_flux_integrated exactly (same physics, just not yet summed).
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    lines = source.spectrum_table("7Be")

    flux_per_line = solar_flux_state(oscillation, lines.energy_MeV, medium, source, "7Be")
    rate = solar_flux_integrated(oscillation, lines.energy_MeV, medium, source, "7Be")

    assert flux_per_line.shape == (2, 3)
    assert torch.isfinite(flux_per_line).all()
    torch.testing.assert_close(flux_per_line.sum(dim=0), rate, rtol=1.0e-13, atol=1.0e-13)


def test_solar_flux_state_line_source_rejects_mismatched_energy_grid():
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    lines = source.spectrum_table("7Be")

    with pytest.raises(ValueError, match="exact line energies"):
        solar_flux_state(oscillation, lines.energy_MeV + 0.01, medium, source, "7Be")


def test_solar_flux_state_line_source_accepts_explicit_spectrum_override():
    # An explicit source_spectrum override bypasses the auto-detection
    # entirely, so it works even on an energy grid that is not the source's
    # own tabulated line energies.
    oscillation = make_oscillation()
    medium, source = make_medium_source()
    energy = torch.tensor([1.0, 5.0], device=DEVICE, dtype=DTYPE)
    override = torch.tensor([0.3, 0.7], device=DEVICE, dtype=DTYPE)

    flux = solar_flux_state(oscillation, energy, medium, source, "7Be", override)

    assert flux.shape == (2, 3)
    assert torch.isfinite(flux).all()
