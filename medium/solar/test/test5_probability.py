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

"""Pytest-compatible tests for solar adiabatic probabilities."""

from __future__ import annotations

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.profile import build_solar_profile
from tpeanuts.medium.solar.probability import Tei, psolar, solar_probability_mass
from tpeanuts.medium.solar.validation import compare_psolar_with_legacy
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, context: RuntimeContext | None = None) -> OscillationParameters:
    return OscillationParameters.from_preset(
        "_SM_NUFIT52_NO",
        context=context or make_context(),
    )


def make_profile(*, use_lz: bool = False):
    context = make_context()
    profile = build_solar_profile(None, context=context)
    profile.use_LZ = use_lz
    return profile


def test_tei_returns_normalized_finite_weights_for_energy_density_grid():
    oscillation = make_oscillation()
    energy = torch.tensor([0.1, 1.0, 10.0], device=DEVICE, dtype=DTYPE)[:, None]
    density = torch.tensor([0.0, 1.0, 100.0], device=DEVICE, dtype=DTYPE)[None, :]

    weights = Tei(oscillation, energy, density)

    assert weights.shape == (3, 3, 3)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), rtol=1.0e-14, atol=1.0e-14)


def test_solar_probability_mass_single_source_shape_and_normalization():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights = solar_probability_mass(oscillation, energy, profile, "8B")

    assert weights.shape == (3, 3)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0.0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-13, atol=1.0e-13)


def test_solar_probability_mass_multiple_sources_preserves_source_order():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "8B", "hep")

    multi = solar_probability_mass(oscillation, energy, profile, sources)
    stacked = torch.stack(
        [solar_probability_mass(oscillation, energy, profile, source) for source in sources],
        dim=0,
    )

    assert multi.shape == (3, 2, 3)
    torch.testing.assert_close(multi, stacked, rtol=1.0e-14, atol=1.0e-14)


def test_psolar_probabilities_are_normalized_and_match_mass_projection():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    weights = solar_probability_mass(oscillation, energy, profile, "8B")
    probabilities = psolar(oscillation, energy, profile, "8B")
    pmns_projection = oscillation.pmns.pmns_matrix().abs() ** 2
    expected = torch.einsum("ei,ni->ne", pmns_projection, weights)

    assert probabilities.shape == (3, 3)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, device=DEVICE, dtype=DTYPE), rtol=1.0e-13, atol=1.0e-13)
    torch.testing.assert_close(probabilities, expected, rtol=1.0e-13, atol=1.0e-13)


def test_psolar_multiple_sources_matches_single_source_stack():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([0.5, 5.0], device=DEVICE, dtype=DTYPE)
    sources = ("pp", "7Be", "8B")

    multi = psolar(oscillation, energy, profile, sources)
    stacked = torch.stack(
        [psolar(oscillation, energy, profile, source) for source in sources],
        dim=0,
    )

    assert multi.shape == (3, 2, 3)
    torch.testing.assert_close(multi, stacked, rtol=1.0e-14, atol=1.0e-14)


def test_electron_survival_decreases_from_low_to_high_energy_for_8b():
    oscillation = make_oscillation()
    profile = make_profile()
    energy = torch.tensor([0.1, 10.0], device=DEVICE, dtype=DTYPE)

    pee = psolar(oscillation, energy, profile, "8B")[:, 0]

    assert pee[0] > pee[1]
    assert 0.45 < float(pee[0]) < 0.65
    assert 0.20 < float(pee[1]) < 0.40


def test_lz_enabled_standard_lma_matches_adiabatic_result_to_float_precision():
    oscillation = make_oscillation()
    profile_ad = make_profile(use_lz=False)
    profile_lz = make_profile(use_lz=True)
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)

    p_ad = psolar(oscillation, energy, profile_ad, "8B")
    p_lz = psolar(oscillation, energy, profile_lz, "8B")

    torch.testing.assert_close(p_lz, p_ad, rtol=0.0, atol=0.0)


def test_psolar_matches_legacy_reference_for_b16_8b_at_5mev_on_cpu():
    context = RuntimeContext.resolve("cpu", DTYPE)
    oscillation = make_oscillation(context=context)

    result = compare_psolar_with_legacy(
        "8B",
        oscillation,
        5.0,
        context=context,
        legacy_precision=True,
    )

    assert result["max_abs"] < 1.0e-10
