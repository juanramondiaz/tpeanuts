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
Pytest-compatible tests for tpeanuts.medium.earth.flux.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import torch

from tpeanuts.medium.earth.exposure_integration import earth_probability_exposure
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.flux import earth_flux_state, earth_flux_exposure
from tpeanuts.medium.earth.probability import earth_probability_state
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


def _oscillation() -> OscillationParameters:
    return OscillationParameters(
        pmns=build_pmns(),
        mass_spectrum=MassSpectrum_SM(
            DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
            DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        ),
        antinu=False,
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


def _sterile_oscillation(deltamsq41=1.7) -> OscillationParameters:
    from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
    from tpeanuts.core.common.pmns import PMNSParams

    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    sm_params = PMNSParams(theta12=0.5836, theta13=0.1498, theta23=0.8552, delta=3.438, context=ctx)
    sterile_params = PMNSSterileParams(
        theta14=0.15, theta24=0.10, theta34=0.05,
        delta14=0.3, delta24=-0.2, delta34=0.0,
        context=ctx,
    )
    pmns4 = PMNS_sterile(sm_params, sterile_params)
    mass_spectrum = MassSpectrum_BSM(
        DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq41=torch.tensor(deltamsq41, device=DEVICE, dtype=DTYPE),
    )
    return OscillationParameters(pmns=pmns4, mass_spectrum=mass_spectrum, antinu=False)


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


def test_earth_flux_matches_probability_times_flux_scalar():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.6, 0.3, 0.1], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.4, device=DEVICE, dtype=DTYPE)

    P = earth_probability_state(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    flux_out = earth_flux_state(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=2.5,
                           method="analytical", massbasis=True, reunitarize=True)

    assert_close(flux_out, P * 2.5, name="earth_flux_state matches earth_probability_state * flux")


def test_earth_flux_applies_spectrum():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.6, 0.3, 0.1], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.4, device=DEVICE, dtype=DTYPE)

    P = earth_probability_state(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    flux_out = earth_flux_state(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=2.0, spectrum=3.0,
                           method="analytical", massbasis=True, reunitarize=True)

    assert_close(flux_out, P * 2.0 * 3.0, name="earth_flux applies flux and spectrum")


def test_earth_flux_flavourbasis_matches_probability():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    psi_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.6, device=DEVICE, dtype=DTYPE)

    P = earth_probability_state(psi_e, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=False, reunitarize=True)
    flux_out = earth_flux_state(psi_e, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=1.0,
                           method="analytical", massbasis=False, reunitarize=True)

    assert_close(flux_out, P, name="earth_flux_state with flux=1.0 matches earth_probability_state")


def test_earth_flux_numerical_full_oscillation_returns_path():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    psi_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.6, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    flux_path, x = earth_flux_state(psi_e, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=4.0,
                               method="numerical", massbasis=False, full_oscillation=True,
                               nsteps=40, context=ctx)

    assert flux_path.shape == (41, 3)
    assert x.shape == (41,)
    assert torch.all(torch.isfinite(flux_path))
    assert_close(torch.sum(flux_path[-1]), torch.tensor(4.0, dtype=DTYPE), name="final flux sums to flux normalization")


def test_earth_flux_broadcast_flux_over_grid():
    # flux broadcasts against the *leading* probability dimensions (here the
    # energy axis), not the trailing flavour axis -- see flux_state.
    # E and eta must have unequal lengths to get an independent (NE, Neta)
    # outer-product grid rather than paired element-wise broadcasting.
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 1500.0, 3000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.3, 1.1], device=DEVICE, dtype=DTYPE)
    flux_per_energy = torch.tensor([10.0, 20.0, 30.0], device=DEVICE, dtype=DTYPE)

    P = earth_probability_state(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    flux_out = earth_flux_state(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=flux_per_energy,
                           method="analytical", massbasis=True, reunitarize=True)

    assert P.shape == (3, 2, 3)
    assert flux_out.shape == (3, 2, 3)
    assert_close(flux_out, P * flux_per_energy[:, None, None], name="earth_flux broadcasts per-energy flux over the leading E axis")


def test_earth_flux_exposure_matches_earth_probability_exposure_times_flux():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    P_int = earth_probability_exposure(weights, profile, oscillation, E, DEPTH_SURFACE_M,
                               method="analytical", massbasis=True, exposure=exposure, context=ctx)
    flux_out = earth_flux_exposure(weights, profile, oscillation, E, DEPTH_SURFACE_M, flux=2.5,
                                      method="analytical", massbasis=True, exposure=exposure, context=ctx)

    assert flux_out.shape == (3,)
    assert_close(flux_out, P_int * 2.5, name="earth_flux_exposure matches earth_probability_exposure * flux")


def test_earth_flux_exposure_vector_energy_shape():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 2000.0], device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    flux_out = earth_flux_exposure(weights, profile, oscillation, E, DEPTH_SURFACE_M, flux=1.0,
                                      method="analytical", massbasis=True, exposure=exposure, context=ctx)

    assert flux_out.shape == (2, 3)
    assert torch.all(torch.isfinite(flux_out))


def test_earth_flux_state_include_matter_nc_changes_sterile_result():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.60, device=DEVICE, dtype=DTYPE)

    flux_cc = earth_flux_state(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=2.0,
                                method="analytical", massbasis=False, reunitarize=True, include_matter_nc=False)
    flux_nc = earth_flux_state(state, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=2.0,
                                method="analytical", massbasis=False, reunitarize=True, include_matter_nc=True)

    assert flux_cc.shape == (4,)
    assert flux_nc.shape == (4,)
    assert torch.max(torch.abs(flux_nc - flux_cc)) > 0.0


def test_earth_probability_exposure_sterile_n4_shape():
    """Regression test: earth_probability_exposure used to hardcode a
    3-flavour output accumulator, which would fail to accumulate a 4-flavour
    (3+1 sterile) probability chunk."""
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    P_int = earth_probability_exposure(weights, profile, oscillation, E, DEPTH_SURFACE_M,
                                        method="numerical", massbasis=True, exposure=exposure,
                                        normalized_exposure=True, nsteps=20, context=ctx)

    assert P_int.shape == (4,)
    assert torch.all(torch.isfinite(P_int))
    assert_close(P_int.sum(), torch.tensor(1.0, device=DEVICE, dtype=DTYPE), atol=1.0e-8, rtol=1.0e-8, name="sterile exposure-integrated probabilities sum to one")


def test_earth_flux_exposure_include_matter_nc_changes_sterile_result():
    profile = _two_shell_prem_profile()
    oscillation = _sterile_oscillation(deltamsq41=DM3L_EV2 * 2.0)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    flux_cc = earth_flux_exposure(weights, profile, oscillation, E, DEPTH_SURFACE_M, flux=1.0,
                                   method="numerical", massbasis=True, exposure=exposure,
                                   nsteps=20, context=ctx, include_matter_nc=False)
    flux_nc = earth_flux_exposure(weights, profile, oscillation, E, DEPTH_SURFACE_M, flux=1.0,
                                   method="numerical", massbasis=True, exposure=exposure,
                                   nsteps=20, context=ctx, include_matter_nc=True)

    assert flux_cc.shape == (4,)
    assert flux_nc.shape == (4,)
    assert torch.max(torch.abs(flux_nc - flux_cc)) > 0.0
