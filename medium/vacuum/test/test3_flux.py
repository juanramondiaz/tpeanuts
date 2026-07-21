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
Pytest-compatible tests for tpeanuts.medium.vacuum.flux.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.vacuum.flux import vacuum_flux_state
from tpeanuts.medium.vacuum.probability import vacuum_probability_state
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3


def _oscillation() -> OscillationParameters:
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    pmns = PMNS_SM(PMNSParams(theta12=0.59, theta13=0.15, theta23=0.78, delta=1.20, context=ctx))
    mass_spectrum = MassSpectrum_SM(
        DeltamSq21=torch.as_tensor(DM21_EV2, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(DM3L_EV2, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=False)


def test_vacuum_flux_matches_probability_times_flux_scalar():
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(weights, oscillation, E, L, massbasis=True)
    flux_out = vacuum_flux_state(weights, oscillation, E, L, flux=2.5, massbasis=True)

    assert_close(flux_out, P * 2.5, name="vacuum_flux_state_state matches vacuum_probability_state * flux")


def test_vacuum_flux_applies_spectrum():
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(weights, oscillation, E, L, massbasis=True)
    flux_out = vacuum_flux_state(weights, oscillation, E, L, flux=2.0, spectrum=3.0, massbasis=True)

    assert_close(flux_out, P * 2.0 * 3.0, name="vacuum_flux_state applies flux and spectrum")


def test_vacuum_flux_flavourbasis_matches_probability():
    oscillation = _oscillation()
    psi_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(850.0, device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(psi_mu, oscillation, E, L, massbasis=False)
    flux_out = vacuum_flux_state(psi_mu, oscillation, E, L, flux=1.0, massbasis=False)

    assert_close(flux_out, P, name="vacuum_flux_state_state with flux=1.0 matches vacuum_probability_state")


def test_vacuum_flux_broadcast_flux_over_leading_energy_axis():
    # flux broadcasts against the *leading* probability dimensions (here the
    # energy axis), not the trailing flavour axis -- see flux_state.
    oscillation = _oscillation()
    weights = torch.tensor([0.5, 0.3, 0.2], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([500.0, 1500.0, 4000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)
    flux_per_energy = torch.tensor([10.0, 20.0, 30.0], device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(weights, oscillation, E, L, massbasis=True)
    flux_out = vacuum_flux_state(weights, oscillation, E, L, flux=flux_per_energy, massbasis=True)

    assert P.shape == (3, 3)
    assert flux_out.shape == (3, 3)
    assert_close(flux_out, P * flux_per_energy[:, None], name="vacuum_flux_state broadcasts per-energy flux over the leading E axis")


def test_vacuum_flux_massbasis_defaults_true():
    oscillation = _oscillation()
    weights = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    flux_default = vacuum_flux_state(weights, oscillation, E, L, flux=1.0)
    flux_explicit = vacuum_flux_state(weights, oscillation, E, L, flux=1.0, massbasis=True)

    assert_close(flux_default, flux_explicit, atol=0.0, rtol=0.0, name="vacuum_flux_state massbasis defaults to True")
