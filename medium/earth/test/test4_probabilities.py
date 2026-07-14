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

"""
Pytest-compatible tests for tpeanuts.medium.earth.probability.

Combines the legacy ``test6_probabilities`` (pearth correctness) and
``test6b_prob_compare`` (analytical-vs-numerical agreement) backup scripts.
The diagnostic plots from those historical backup tests live in notebooks;
this file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.medium.earth.probability import pearth, pearth_analytical, pearth_numerical
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close, build_pmns


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3
DEPTH_SURFACE_M = 0.0
DEPTH_UNDERGROUND_M = 1000.0
NSTEPS_COMPARE = 200


def _oscillation(antinu=False) -> OscillationParameters:
    return OscillationParameters(
        pmns=build_pmns(),
        DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        antinu=antinu,
    )


def _two_shell_profile() -> EarthProfile:
    """Synthetic two-shell profile: constant density 2.0 for r<0.5, 1.0 for 0.5<r<=1.0."""
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    params = EarthParameters(
        profile_perturbative_name="even_power",
        profile_perturbative_kwargs={"rj": rj, "coefficients": coefficients},
    )
    return EarthProfile(params=params, context=RuntimeContext.resolve(DEVICE, DTYPE))


def _assert_probability_vector(P: torch.Tensor, atol: float = 1.0e-8) -> None:
    row_sum = torch.sum(P, dim=-1)
    assert torch.all(torch.isfinite(P))
    assert torch.all(P >= -atol)
    assert_close(row_sum, torch.ones_like(row_sum), atol=atol, rtol=atol, name="probabilities sum to one")


def _analytical(state, profile, oscillation, E, eta, depth_m, *, massbasis):
    return pearth(
        state, profile, oscillation, E, eta, depth_m,
        method="analytical", massbasis=massbasis, reunitarize=True,
    )


def _numerical(state, profile, oscillation, E, eta, depth_m, *, massbasis, nsteps=NSTEPS_COMPARE, full_oscillation=False):
    return pearth(
        state, profile, oscillation, E, eta, depth_m,
        method="numerical", massbasis=massbasis, full_oscillation=full_oscillation,
        nsteps=nsteps, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE),
    )


def test_massbasis_scalar_probabilities_are_normalized():
    profile = _two_shell_profile()
    nustate = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    P = _analytical(nustate, profile, _oscillation(), torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
                     torch.tensor(0.40, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, massbasis=True)

    assert P.shape == (3,)
    _assert_probability_vector(P)


def test_flavourbasis_scalar_probabilities_are_normalized():
    profile = _two_shell_profile()
    psi_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)

    P = _analytical(psi_e, profile, _oscillation(), torch.tensor(1200.0, device=DEVICE, dtype=DTYPE),
                     torch.tensor(0.60, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, massbasis=False)

    assert P.shape == (3,)
    _assert_probability_vector(P)


def test_above_horizon_identity_limits():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    mass_state_1 = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    flavour_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    eta_above = torch.tensor(2.20, device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    P_mass = _analytical(mass_state_1, profile, oscillation, E, eta_above, DEPTH_SURFACE_M, massbasis=True)
    P_flavour = _analytical(flavour_e, profile, oscillation, E, eta_above, DEPTH_SURFACE_M, massbasis=False)

    expected_mass = torch.abs(oscillation.pmns.pmns_matrix()[:, 0]) ** 2
    expected_flavour = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    assert_close(P_mass, expected_mass.real, name="above-horizon mass-basis reduces to PMNS column")
    assert_close(P_flavour, expected_flavour, name="above-horizon flavour-basis keeps input flavour")


def test_energy_eta_grid_probabilities_shape_and_normalization():
    profile = _two_shell_profile()
    weights = torch.tensor([0.55, 0.30, 0.15], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 2000.0, 6000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.30, 1.10], device=DEVICE, dtype=DTYPE)

    P = _analytical(weights, profile, _oscillation(), E, eta, DEPTH_SURFACE_M, massbasis=True)

    assert P.shape == (3, 2, 3)
    _assert_probability_vector(P)


def test_antineutrino_probabilities_differ_and_valid():
    profile = _two_shell_profile()
    weights = torch.tensor([0.20, 0.50, 0.30], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([1000.0, 5000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.35, 1.00], device=DEVICE, dtype=DTYPE)

    P_nu = _analytical(weights, profile, _oscillation(antinu=False), E, eta, DEPTH_SURFACE_M, massbasis=True)
    P_antinu = _analytical(weights, profile, _oscillation(antinu=True), E, eta, DEPTH_SURFACE_M, massbasis=True)

    assert P_antinu.shape == (2, 3)
    _assert_probability_vector(P_antinu)
    assert torch.max(torch.abs(P_nu - P_antinu)) > 0.0


def test_case_a_flavourbasis_analytical_vs_numerical():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_analytical = _analytical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False)
    P_numerical = _numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False)

    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.max(torch.abs(P_analytical - P_numerical)) < 5.0e-3


def test_case_b_flavourbasis_analytical_vs_numerical():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(2.40, device=DEVICE, dtype=DTYPE)

    P_analytical = _analytical(state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M, massbasis=False)
    P_numerical = _numerical(state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M, massbasis=False)

    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.max(torch.abs(P_analytical - P_numerical)) < 5.0e-6


def test_numerical_full_oscillation_final_matches_final_only():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    evolution, x = _numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False, full_oscillation=True)
    final_only = _numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False, full_oscillation=False)

    assert evolution.shape == (NSTEPS_COMPARE + 1, 3)
    assert x.shape == (NSTEPS_COMPARE + 1,)
    _assert_probability_vector(evolution[-1])
    assert_close(evolution[-1], final_only, atol=1.0e-12, rtol=1.0e-12, name="full-path final state matches final-only result")


def test_massbasis_diagnostic_difference_is_finite():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_analytical = _analytical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=True)
    P_numerical = _numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=True)

    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.isfinite(torch.max(torch.abs(P_analytical - P_numerical)))


def test_pearth_dispatch_matches_direct_analytical_call():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.5, device=DEVICE, dtype=DTYPE)

    P_dispatch = pearth(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    P_direct = pearth_analytical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=True, reunitarize=True)

    assert_close(P_dispatch, P_direct, atol=0.0, rtol=0.0, name="pearth(method='analytical') matches pearth_analytical")


def test_pearth_dispatch_matches_direct_numerical_call():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.5, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    P_dispatch = pearth(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="numerical", massbasis=True, nsteps=50, context=ctx)
    P_direct = pearth_numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=True, nsteps=50, context=ctx)

    assert_close(P_dispatch, P_direct, atol=0.0, rtol=0.0, name="pearth(method='numerical') matches pearth_numerical")


def test_pearth_rejects_invalid_method():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="method must be either"):
        pearth(state, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
               torch.tensor(0.4, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, method="bogus")


def test_pearth_numerical_rejects_non_scalar_antinu():
    profile = _two_shell_profile()
    oscillation = OscillationParameters(
        pmns=build_pmns(),
        DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        antinu=torch.tensor([False, True], device=DEVICE),
    )
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="only supports scalar antinu"):
        pearth_numerical(state, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
                          torch.tensor(0.4, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, massbasis=True,
                          context=RuntimeContext.resolve(DEVICE, DTYPE))
