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

from tpeanuts.medium.earth.exposure_integration import pearth_integrated
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.flux import earth_flux, earth_flux_integrated
from tpeanuts.medium.earth.probability import pearth
from tpeanuts.medium.earth.profile import EarthParameters, EarthProfile
from tpeanuts.core.common.oscillation import OscillationParameters
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
        DeltamSq21=torch.tensor(DM21_EV2, device=DEVICE, dtype=DTYPE),
        DeltamSq3l=torch.tensor(DM3L_EV2, device=DEVICE, dtype=DTYPE),
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


def test_earth_flux_matches_probability_times_flux_scalar():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.6, 0.3, 0.1], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.4, device=DEVICE, dtype=DTYPE)

    P = pearth(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    flux_out = earth_flux(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=2.5,
                           method="analytical", massbasis=True, reunitarize=True)

    assert_close(flux_out, P * 2.5, name="earth_flux matches pearth * flux")


def test_earth_flux_applies_spectrum():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.6, 0.3, 0.1], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.4, device=DEVICE, dtype=DTYPE)

    P = pearth(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    flux_out = earth_flux(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=2.0, spectrum=3.0,
                           method="analytical", massbasis=True, reunitarize=True)

    assert_close(flux_out, P * 2.0 * 3.0, name="earth_flux applies flux and spectrum")


def test_earth_flux_flavourbasis_matches_probability():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    psi_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.6, device=DEVICE, dtype=DTYPE)

    P = pearth(psi_e, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=False, reunitarize=True)
    flux_out = earth_flux(psi_e, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=1.0,
                           method="analytical", massbasis=False, reunitarize=True)

    assert_close(flux_out, P, name="earth_flux with flux=1.0 matches pearth")


def test_earth_flux_numerical_full_oscillation_returns_path():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    psi_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.6, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    flux_path, x = earth_flux(psi_e, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=4.0,
                               method="numerical", massbasis=False, full_oscillation=True,
                               nsteps=40, context=ctx)

    assert flux_path.shape == (41, 3)
    assert x.shape == (41,)
    assert torch.all(torch.isfinite(flux_path))
    assert_close(torch.sum(flux_path[-1]), torch.tensor(4.0, dtype=DTYPE), name="final flux sums to flux normalization")


def test_earth_flux_broadcast_flux_over_grid():
    # flux broadcasts against the *leading* probability dimensions (here the
    # energy axis), not the trailing flavour axis -- see flux_from_probability.
    # E and eta must have unequal lengths to get an independent (NE, Neta)
    # outer-product grid rather than paired element-wise broadcasting.
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 1500.0, 3000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.3, 1.1], device=DEVICE, dtype=DTYPE)
    flux_per_energy = torch.tensor([10.0, 20.0, 30.0], device=DEVICE, dtype=DTYPE)

    P = pearth(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, method="analytical", massbasis=True, reunitarize=True)
    flux_out = earth_flux(weights, profile, oscillation, E, eta, DEPTH_SURFACE_M, flux=flux_per_energy,
                           method="analytical", massbasis=True, reunitarize=True)

    assert P.shape == (3, 2, 3)
    assert flux_out.shape == (3, 2, 3)
    assert_close(flux_out, P * flux_per_energy[:, None, None], name="earth_flux broadcasts per-energy flux over the leading E axis")


def test_earth_flux_integrated_matches_pearth_integrated_times_flux():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    P_int = pearth_integrated(weights, profile, oscillation, E, DEPTH_SURFACE_M,
                               method="analytical", massbasis=True, exposure=exposure, context=ctx)
    flux_out = earth_flux_integrated(weights, profile, oscillation, E, DEPTH_SURFACE_M, flux=2.5,
                                      method="analytical", massbasis=True, exposure=exposure, context=ctx)

    assert flux_out.shape == (3,)
    assert_close(flux_out, P_int * 2.5, name="earth_flux_integrated matches pearth_integrated * flux")


def test_earth_flux_integrated_vector_energy_shape():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([800.0, 2000.0], device=DEVICE, dtype=DTYPE)
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    exposure = ExposureParameters(detector_latitude_rad=0.5, exposure_ns=9, exposure_use_cache=False)

    flux_out = earth_flux_integrated(weights, profile, oscillation, E, DEPTH_SURFACE_M, flux=1.0,
                                      method="analytical", massbasis=True, exposure=exposure, context=ctx)

    assert flux_out.shape == (2, 3)
    assert torch.all(torch.isfinite(flux_out))
