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
Pytest-compatible tests for tpeanuts.medium.earth.evolutor.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from tpeanuts.medium.earth.evolutor import earth_evolutor, earth_evolutor_from_zenith
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


def _unitarity_error(U: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(3, device=U.device, dtype=U.dtype)
    left = U.conj().transpose(-1, -2) @ U
    return torch.amax(torch.abs(left - identity), dim=(-2, -1))


def test_earth_evolutor_is_callable():
    assert callable(earth_evolutor)
    assert callable(earth_evolutor_from_zenith)


def test_above_horizon_is_identity_at_surface():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    eta = torch.tensor([math.pi / 2.0, 2.0, 3.0], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([1000.0, 2000.0, 3000.0], device=DEVICE, dtype=DTYPE)

    U = earth_evolutor(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True)

    identity = torch.eye(3, device=DEVICE, dtype=U.dtype).expand(3, 3, 3)
    assert_close(U, identity, atol=1.0e-12, rtol=1.0e-12, name="above-horizon identity")


def test_case_a_scalar_evolutor_is_finite_and_unitary():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    U = earth_evolutor(
        profile,
        oscillation,
        torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(0.40, device=DEVICE, dtype=DTYPE),
        DEPTH_SURFACE_M,
        reunitarize=True,
    )

    assert U.shape == (3, 3)
    assert torch.all(torch.isfinite(U.real)) and torch.all(torch.isfinite(U.imag))
    assert torch.max(_unitarity_error(U)) < 5.0e-10


def test_case_b_underground_evolutor_is_finite_and_unitary():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    U = earth_evolutor(
        profile,
        oscillation,
        torch.tensor(2500.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(2.40, device=DEVICE, dtype=DTYPE),
        DEPTH_UNDERGROUND_M,
        reunitarize=True,
    )

    assert U.shape == (3, 3)
    assert torch.all(torch.isfinite(U.real)) and torch.all(torch.isfinite(U.imag))
    assert torch.max(_unitarity_error(U)) < 5.0e-10


def test_energy_eta_grid_output_shape_and_identity_region():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    E = torch.tensor([800.0, 2000.0, 6000.0], device=DEVICE, dtype=DTYPE)
    eta = torch.tensor([0.15, 2.20], device=DEVICE, dtype=DTYPE)

    U = earth_evolutor(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True)

    assert U.shape == (3, 2, 3, 3)
    assert torch.max(_unitarity_error(U)) < 5.0e-10

    identity = torch.eye(3, device=DEVICE, dtype=U.dtype).expand(3, 3, 3)
    assert_close(U[:, 1], identity, atol=1.0e-12, rtol=1.0e-12, name="above-horizon eta column is identity")


def test_antineutrino_path_differs_and_is_unitary():
    profile = _two_shell_profile()

    eta = torch.tensor([0.25, 1.10], device=DEVICE, dtype=DTYPE)
    E = torch.tensor([1000.0, 4000.0], device=DEVICE, dtype=DTYPE)

    U_nu = earth_evolutor(profile, _oscillation(antinu=False), E, eta, DEPTH_SURFACE_M, reunitarize=True)
    U_antinu = earth_evolutor(profile, _oscillation(antinu=True), E, eta, DEPTH_SURFACE_M, reunitarize=True)

    assert U_antinu.shape == (2, 3, 3)
    assert torch.max(_unitarity_error(U_antinu)) < 5.0e-10
    assert torch.max(torch.abs(U_nu - U_antinu)) > 0.0


def test_invalid_eta_raises_value_error():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    with pytest.raises(ValueError):
        earth_evolutor(
            profile,
            oscillation,
            torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
            torch.tensor(-0.01, device=DEVICE, dtype=DTYPE),
            DEPTH_SURFACE_M,
        )


def test_earth_evolutor_from_zenith_matches_nadir_convention():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    theta_deg = torch.tensor([170.0, 120.0, 45.0], device=DEVICE, dtype=DTYPE)
    eta_equiv = math.pi - torch.deg2rad(theta_deg)
    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)

    U_zenith = earth_evolutor_from_zenith(profile, oscillation, E, theta_deg, DEPTH_SURFACE_M, reunitarize=True)
    U_nadir = earth_evolutor(profile, oscillation, E, eta_equiv, DEPTH_SURFACE_M, reunitarize=True)

    assert_close(U_zenith, U_nadir, name="earth_evolutor_from_zenith matches eta=pi-theta convention")


def test_reunitarize_reduces_unitarity_error():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    E = torch.tensor(2000.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.55, device=DEVICE, dtype=DTYPE)

    U_raw = earth_evolutor(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=False)
    U_projected = earth_evolutor(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True)

    err_raw = float(torch.max(_unitarity_error(U_raw)))
    err_projected = float(torch.max(_unitarity_error(U_projected)))

    assert err_projected <= err_raw + 1.0e-14


def test_legacy_precision_flag_runs_and_is_unitary():
    profile = _two_shell_profile()
    oscillation = _oscillation()

    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)
    eta = torch.tensor(0.45, device=DEVICE, dtype=DTYPE)

    U = earth_evolutor(profile, oscillation, E, eta, DEPTH_SURFACE_M, reunitarize=True, legacy_precision=True)

    assert torch.all(torch.isfinite(U.real)) and torch.all(torch.isfinite(U.imag))
    assert torch.max(_unitarity_error(U)) < 5.0e-10


def test_oscillation_object_is_not_mutated():
    profile = _two_shell_profile()
    oscillation = _oscillation()
    antinu_before = oscillation.antinu

    earth_evolutor(profile, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
                    torch.tensor(0.4, device=DEVICE, dtype=DTYPE), DEPTH_SURFACE_M, reunitarize=True)

    assert oscillation.antinu == antinu_before
    assert dataclasses.is_dataclass(oscillation)
