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

Combines the legacy ``test6_probabilities`` (earth_probability_state
correctness) and ``test6b_prob_compare`` (analytical-vs-numerical agreement)
backup scripts. The diagnostic plots from those historical backup tests live
in notebooks; this file keeps only fast numerical sanity checks that can run
automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.medium.earth.probability import (
    earth_probability_state,
    earth_probability_state_analytical,
    earth_probability_state_numerical,
    earth_probability_transition,
)
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close, build_pmns


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3
DEPTH_SURFACE_M = 0.0
DEPTH_UNDERGROUND_M = 1000.0
NSTEPS_COMPARE = 200


def _oscillation(antinu=False, nsi=None) -> OscillationParameters:
    return OscillationParameters(
        pmns=build_pmns(),
        mass_spectrum=MassSpectrum_SM(
            DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
            DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        ),
        antinu=antinu,
        nsi=nsi,
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


def _two_shell_even_power_neutron_profile() -> EarthProfile:
    """Synthetic two-shell even_power profile with constant n_e/n_n per shell."""
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE, dtype=DTYPE)
    coefficients_n = torch.tensor([[1.6, 0.0, 0.0], [0.8, 0.0, 0.0]], device=DEVICE, dtype=DTYPE)
    params = EarthParameters(
        profile_perturbative_name="even_power",
        profile_perturbative_kwargs={
            "rj": rj, "coefficients": coefficients, "coefficients_n": coefficients_n,
        },
    )
    return EarthProfile(params=params, context=RuntimeContext.resolve(DEVICE, DTYPE))


def _two_shell_prem_profile() -> EarthProfile:
    """Synthetic two-shell PREM profile with constant n_e/n_n per shell."""
    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor([[2.0, 0.0], [1.0, 0.0]], device=DEVICE, dtype=DTYPE)
    coefficients_n = torch.tensor([[1.6, 0.0], [0.8, 0.0]], device=DEVICE, dtype=DTYPE)
    params = EarthParameters(
        profile_perturbative_name="prem",
        profile_perturbative_kwargs={
            "rj": rj, "coefficients": coefficients, "coefficients_n": coefficients_n,
        },
    )
    return EarthProfile(params=params, context=RuntimeContext.resolve(DEVICE, DTYPE))


def _assert_probability_vector(P: torch.Tensor, atol: float = 1.0e-8) -> None:
    row_sum = torch.sum(P, dim=-1)
    assert torch.all(torch.isfinite(P))
    assert torch.all(P >= -atol)
    assert_close(row_sum, torch.ones_like(row_sum), atol=atol, rtol=atol, name="probabilities sum to one")


def _analytical(state, profile, oscillation, E, eta, depth_m, *, massbasis):
    return earth_probability_state(
        state, profile, oscillation, E, eta, depth_m,
        method="analytical", massbasis=massbasis, reunitarize=True,
    )


def _numerical(state, profile, oscillation, E, eta, depth_m, *, massbasis, nsteps=NSTEPS_COMPARE, full_oscillation=False):
    return earth_probability_state(
        state, profile, oscillation, E, eta, depth_m,
        method="numerical", massbasis=massbasis, full_oscillation=full_oscillation,
        nsteps=nsteps, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE),
    )


def _sterile_oscillation(antinu=False, deltamsq41=1.7, delta14=0.3, nsi=None) -> OscillationParameters:
    """4-flavour (3+1 sterile) oscillation object for Fase 6 N=4 checks.

    ``deltamsq41`` defaults to a physically realistic sterile splitting
    (Giunti et al. 2017 best fit scale), but the analytical-vs-numerical
    differential tests below override it to a smaller value: at 1.7 eV^2 the
    sterile oscillation length is a few km, far shorter than the ~200-step
    numerical trajectory grid can resolve, so the numerical path would be
    aliased rather than actually disagreeing with the analytical one.

    ``delta14`` (the active-sterile CP phase) defaults to a non-zero value on
    purpose: it exercises the genuinely complex reduced mixing matrix case
    for both Case A and Case B (see the ``conjugate_right`` fix in
    ``core/BSM/hamiltonian.py``/``core/perturbative/evolutor.py`` and the
    direct far-side segment recomputation in
    ``medium/earth/evolutor.py::_earth_evolutor_case_a_batched``).
    """
    from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
    from tpeanuts.core.common.pmns import PMNSParams

    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    sm_params = PMNSParams(theta12=0.5836, theta13=0.1498, theta23=0.8552, delta=3.438, context=ctx)
    sterile_params = PMNSSterileParams(
        theta14=0.15, theta24=0.10, theta34=0.05,
        delta14=delta14, delta24=-0.2, delta34=0.0,
        context=ctx,
    )
    pmns4 = PMNS_sterile(sm_params, sterile_params)
    mass_spectrum = MassSpectrum_BSM(
        DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq41=torch.tensor(deltamsq41, device=DEVICE, dtype=DTYPE),
    )
    return OscillationParameters(pmns=pmns4, mass_spectrum=mass_spectrum, antinu=antinu, nsi=nsi)


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


def test_sterile_case_a_flavourbasis_analytical_vs_numerical():
    profile = _two_shell_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_analytical = _analytical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False)
    P_numerical = _numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False)

    assert P_analytical.shape == (4,)
    assert P_numerical.shape == (4,)
    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.max(torch.abs(P_analytical - P_numerical)) < 5.0e-3


def test_sterile_case_b_flavourbasis_analytical_vs_numerical():
    profile = _two_shell_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(2.40, device=DEVICE, dtype=DTYPE)

    P_analytical = _analytical(state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M, massbasis=False)
    P_numerical = _numerical(state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M, massbasis=False)

    assert P_analytical.shape == (4,)
    assert P_numerical.shape == (4,)
    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.max(torch.abs(P_analytical - P_numerical)) < 5.0e-6


def test_numerical_include_matter_nc_changes_sterile_result_with_prem_profile():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_state_numerical(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
        nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=False,
    )
    P_nc = earth_probability_state_numerical(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
        nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=True,
    )

    assert P_cc.shape == (4,)
    assert P_nc.shape == (4,)
    _assert_probability_vector(P_cc)
    _assert_probability_vector(P_nc)
    assert torch.max(torch.abs(P_nc - P_cc)) > 0.0


def test_numerical_include_matter_nc_changes_sterile_result_with_even_power_profile():
    profile = _two_shell_even_power_neutron_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_state_numerical(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
        nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=False,
    )
    P_nc = earth_probability_state_numerical(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
        nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=True,
    )

    assert P_cc.shape == (4,)
    assert P_nc.shape == (4,)
    _assert_probability_vector(P_cc)
    _assert_probability_vector(P_nc)
    assert torch.max(torch.abs(P_nc - P_cc)) > 0.0


def test_numerical_include_matter_nc_is_noop_for_three_flavour():
    profile = _two_shell_prem_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_state_numerical(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
        nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=False,
    )
    P_nc = earth_probability_state_numerical(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
        nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=True,
    )

    assert_close(P_nc, P_cc, atol=1.0e-13, rtol=1.0e-13, name="3-flavour ignores include_matter_nc")


def test_numerical_include_matter_nc_raises_for_even_power_without_include_neutron():
    profile = _two_shell_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="neutron-density coefficients"):
        earth_probability_state_numerical(
            state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False,
            nsteps=NSTEPS_COMPARE, context=RuntimeContext.resolve(DEVICE, DTYPE),
            include_matter_nc=True,
        )


def test_analytical_include_matter_nc_changes_sterile_result_case_a():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=False,
    )
    P_nc = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=True,
    )

    assert P_cc.shape == (4,)
    assert P_nc.shape == (4,)
    _assert_probability_vector(P_cc)
    _assert_probability_vector(P_nc)
    assert torch.max(torch.abs(P_nc - P_cc)) > 0.0


def test_analytical_include_matter_nc_changes_sterile_result_case_b():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(2.40, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=False,
    )
    P_nc = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=True,
    )

    assert P_cc.shape == (4,)
    assert P_nc.shape == (4,)
    _assert_probability_vector(P_cc)
    _assert_probability_vector(P_nc)
    assert torch.max(torch.abs(P_nc - P_cc)) > 0.0


def test_analytical_vs_numerical_include_matter_nc_agree_case_a():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_analytical = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=True,
    )
    P_numerical = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
        method="numerical", massbasis=False, nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=True,
    )

    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.max(torch.abs(P_analytical - P_numerical)) < 5.0e-3


def test_analytical_vs_numerical_include_matter_nc_agree_case_b():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(2.40, device=DEVICE, dtype=DTYPE)

    P_analytical = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=True,
    )
    P_numerical = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_UNDERGROUND_M,
        method="numerical", massbasis=False, nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE), include_matter_nc=True,
    )

    _assert_probability_vector(P_analytical)
    _assert_probability_vector(P_numerical)
    assert torch.max(torch.abs(P_analytical - P_numerical)) < 5.0e-6


def test_analytical_include_matter_nc_raises_for_even_power_without_include_neutron():
    profile = _two_shell_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="neutron-density coefficients"):
        earth_probability_state(
            state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
            method="analytical", massbasis=False, include_matter_nc=True,
        )


def test_analytical_include_matter_nc_is_noop_for_three_flavour():
    profile = _two_shell_prem_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=False,
    )
    P_nc = earth_probability_state(
        state, profile, oscillation, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True, include_matter_nc=True,
    )

    assert_close(P_nc, P_cc, atol=1.0e-13, rtol=1.0e-13, name="3-flavour analytical ignores include_matter_nc")


def test_nsi_case_a_flavourbasis_analytical_forwards_epsilon_and_matches_numerical():
    """Regression test: ``earth_probability_state``/``earth_probability_state_analytical`` used to silently
    drop ``epsilon`` for ``method="analytical"`` (never forwarded to
    ``earth_evolutor``, even though ``earth_evolutor`` itself already
    supported NSI). This also exercises a genuinely varying (non-constant)
    density profile together with NSI on a multi-shell Case A trajectory,
    which used to crash ``evolutor_first_order``'s NSI sandwich term (see
    the ``core/perturbative/evolutor.py`` P-broadcasting fix).
    """
    from tpeanuts.core.BSM.bsm_nsi import NSIConfig

    rj = torch.tensor([0.5, 1.0], device=DEVICE, dtype=DTYPE)
    coefficients = torch.tensor(
        [[2.0, 0.3, 0.0], [1.0, 0.1, 0.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    profile = EarthProfile(
        params=EarthParameters(
            profile_perturbative_name="even_power",
            profile_perturbative_kwargs={"rj": rj, "coefficients": coefficients},
        ),
        context=RuntimeContext.resolve(DEVICE, DTYPE),
    )
    nsi_cfg = NSIConfig.from_preset("nsi_globalfit_esteban2018", device=DEVICE, real_dtype=DTYPE)
    oscillation_nsi = _oscillation(nsi=nsi_cfg)
    oscillation_sm = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.30, device=DEVICE, dtype=DTYPE)  # crosses both shells -> Case A, multi-segment

    P_analytical_nsi = earth_probability_state(
        state, profile, oscillation_nsi, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True,
    )
    P_analytical_sm = earth_probability_state(
        state, profile, oscillation_sm, E, eta, DEPTH_SURFACE_M,
        method="analytical", massbasis=False, reunitarize=True,
    )
    P_numerical_nsi = earth_probability_state(
        state, profile, oscillation_nsi, E, eta, DEPTH_SURFACE_M,
        method="numerical", massbasis=False, nsteps=NSTEPS_COMPARE, ode_method="midpoint",
        context=RuntimeContext.resolve(DEVICE, DTYPE),
    )

    _assert_probability_vector(P_analytical_nsi)
    _assert_probability_vector(P_numerical_nsi)
    assert torch.max(torch.abs(P_analytical_nsi - P_analytical_sm)) > 1.0e-4, (
        "epsilon must not be silently dropped by earth_probability_state(method='analytical')"
    )
    assert torch.max(torch.abs(P_analytical_nsi - P_numerical_nsi)) < 5.0e-3


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

    P_dispatch = earth_probability_state(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    P_direct = earth_probability_state_analytical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=True, reunitarize=True)

    assert_close(P_dispatch, P_direct, atol=0.0, rtol=0.0, name="earth_probability_state(method='analytical') matches earth_probability_state_analytical")


def test_pearth_dispatch_matches_direct_numerical_call():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.5, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    P_dispatch = earth_probability_state(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="numerical", massbasis=True, nsteps=50, context=ctx)
    P_direct = earth_probability_state_numerical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=True, nsteps=50, context=ctx)

    assert_close(P_dispatch, P_direct, atol=0.0, rtol=0.0, name="earth_probability_state(method='numerical') matches earth_probability_state_numerical")


def test_pearth_rejects_invalid_method():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="method must be either"):
        earth_probability_state(state, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
               torch.tensor(0.4, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, method="bogus")


def test_pearth_numerical_rejects_non_scalar_antinu():
    profile = _two_shell_profile()
    oscillation = OscillationParameters(
        pmns=build_pmns(),
        mass_spectrum=MassSpectrum_SM(
            DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
            DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        ),
        antinu=torch.tensor([False, True], device=DEVICE),
    )
    state = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="only supports scalar antinu"):
        earth_probability_state_numerical(state, profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
                          torch.tensor(0.4, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, massbasis=True,
                          context=RuntimeContext.resolve(DEVICE, DTYPE))


def test_earth_probability_transition_is_doubly_stochastic():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P = earth_probability_transition(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True)

    assert P.shape == (3, 3)
    assert torch.all(torch.isfinite(P))
    assert torch.all(P >= -1.0e-10)
    torch.testing.assert_close(P.sum(dim=-2), torch.ones(3, device=DEVICE, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10)


def test_earth_probability_transition_matches_probability_state_column():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_matrix = earth_probability_transition(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True)

    for initial_index in range(3):
        state = torch.zeros(3, device=DEVICE, dtype=torch.complex128)
        state[initial_index] = 1.0
        P_coherent = earth_probability_state_analytical(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, massbasis=False, reunitarize=True)
        assert_close(P_coherent, P_matrix[:, initial_index], name=f"earth_probability_transition column matches earth_probability_state for initial index {initial_index}")


def test_earth_probability_transition_include_matter_nc_changes_sterile_result():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    P_cc = earth_probability_transition(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True, include_matter_nc=False)
    P_nc = earth_probability_transition(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True, include_matter_nc=True)

    assert P_cc.shape == (4, 4)
    assert P_nc.shape == (4, 4)
    assert torch.max(torch.abs(P_nc - P_cc)) > 0.0
