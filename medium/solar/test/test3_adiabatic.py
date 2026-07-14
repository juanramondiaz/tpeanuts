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

"""Pytest-compatible tests for solar adiabatic matter mixing utilities."""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.matter_mixing import DeltamSqee, Vk, th12_M, th13_M
from tpeanuts.medium.solar.probability import Tei
from tpeanuts.util.context import RuntimeContext


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VK_FACTOR_LEGACY = 3.868e-7 / 2.533


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(
    *,
    antinu: bool | torch.Tensor = False,
    DeltamSq3l: float = 2.517e-3,
    dtype: torch.dtype = DTYPE,
) -> OscillationParameters:
    return OscillationParameters.build(
        theta12=0.59,
        theta13=0.15,
        theta23=0.78,
        delta=1.20,
        DeltamSq21=7.42e-5,
        DeltamSq3l=DeltamSq3l,
        antinu=antinu,
        context=make_context(dtype),
    )


def test_vk_legacy_prefactor_matches_original_formula_and_broadcasts():
    energy = torch.tensor([[1.0], [5.0], [10.0]], device=DEVICE, dtype=DTYPE)
    density = torch.tensor([1.0, 10.0, 100.0], device=DEVICE, dtype=DTYPE)
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)

    actual = Vk(dm21, energy, density, legacy_precision=True)
    expected = VK_FACTOR_LEGACY * density * energy / dm21

    assert actual.shape == (3, 3)
    torch.testing.assert_close(actual, expected, rtol=1.0e-14, atol=0.0)


def test_vk_full_precision_antineutrino_sign_and_dtype_device():
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    density = torch.tensor([100.0, 50.0, 10.0], device=DEVICE, dtype=DTYPE)
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)

    vk_nu = Vk(dm21, energy, density)
    vk_antinu = Vk(dm21, energy, density, antinu=True)

    assert vk_nu.device.type == DEVICE.type
    assert vk_nu.dtype == DTYPE
    torch.testing.assert_close(vk_antinu, -vk_nu, rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.parametrize(
    ("dm3l", "dm31", "dm32"),
    [
        (2.517e-3, 2.517e-3, 2.517e-3 - 7.42e-5),
        (-2.498e-3, -2.498e-3 + 7.42e-5, -2.498e-3),
    ],
)
def test_deltamsqee_matches_normal_and_inverted_ordering_formula(dm3l, dm31, dm32):
    oscillation = make_oscillation(DeltamSq3l=dm3l)
    theta12 = oscillation.pmns.params.theta12

    expected = torch.cos(theta12) ** 2 * dm31 + torch.sin(theta12) ** 2 * dm32

    torch.testing.assert_close(DeltamSqee(oscillation), expected, rtol=1.0e-14, atol=0.0)


def test_matter_angles_return_vacuum_limit_at_zero_density():
    oscillation = make_oscillation()
    energy = torch.tensor([0.1, 1.0, 10.0], device=DEVICE, dtype=DTYPE)
    density = torch.zeros_like(energy)

    theta13_m = th13_M(oscillation, energy, density)
    theta12_m = th12_M(oscillation, energy, density)

    torch.testing.assert_close(theta13_m, torch.full_like(energy, 0.15), rtol=1.0e-14, atol=1.0e-14)
    torch.testing.assert_close(theta12_m, torch.full_like(energy, 0.59), rtol=1.0e-14, atol=1.0e-14)


def test_matter_angles_are_finite_bounded_and_broadcast_over_energy_density_grid():
    oscillation = make_oscillation()
    energy = torch.logspace(-1, 1.2, 9, device=DEVICE, dtype=DTYPE)[:, None]
    density = torch.logspace(-4, 2.2, 11, device=DEVICE, dtype=DTYPE)[None, :]

    theta13_m = th13_M(oscillation, energy, density)
    theta12_m = th12_M(oscillation, energy, density)

    assert theta13_m.shape == (9, 11)
    assert theta12_m.shape == (9, 11)
    assert torch.isfinite(theta13_m).all()
    assert torch.isfinite(theta12_m).all()
    assert torch.all((theta13_m >= 0.0) & (theta13_m <= math.pi / 2.0))
    assert torch.all((theta12_m >= 0.0) & (theta12_m <= math.pi / 2.0))


def test_neutrino_matter_theta12_increases_with_density_for_lma_parameters():
    oscillation = make_oscillation()
    energy = torch.tensor(10.0, device=DEVICE, dtype=DTYPE)
    density = torch.logspace(-6, 2.2, 80, device=DEVICE, dtype=DTYPE)

    theta12_m = th12_M(oscillation, energy, density)

    assert torch.all(theta12_m[1:] >= theta12_m[:-1] - 1.0e-12)
    assert theta12_m[-1] > theta12_m[0]
    assert theta12_m[-1] > math.pi / 4.0


def test_antineutrino_matter_theta12_moves_below_vacuum_angle():
    antinu_oscillation = make_oscillation(antinu=True)
    energy = torch.tensor([1.0, 5.0, 10.0], device=DEVICE, dtype=DTYPE)
    density = torch.full_like(energy, 100.0)

    theta12_m = th12_M(antinu_oscillation, energy, density)

    assert torch.all(theta12_m < antinu_oscillation.pmns.params.theta12)


def test_tei_weights_are_normalized_and_match_matter_angles():
    oscillation = make_oscillation()
    energy = torch.tensor([1.0, 10.0], device=DEVICE, dtype=DTYPE)[:, None]
    density = torch.tensor([0.0, 10.0, 100.0], device=DEVICE, dtype=DTYPE)[None, :]

    weights = Tei(oscillation, energy, density)
    theta13_m = th13_M(oscillation, energy, density)
    theta12_m = th12_M(oscillation, energy, density)
    expected = torch.stack(
        [
            torch.cos(theta13_m) ** 2 * torch.cos(theta12_m) ** 2,
            torch.cos(theta13_m) ** 2 * torch.sin(theta12_m) ** 2,
            torch.sin(theta13_m) ** 2,
        ],
        dim=-1,
    )

    assert weights.shape == (2, 3, 3)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), rtol=1.0e-14, atol=1.0e-14)
    torch.testing.assert_close(weights, expected, rtol=1.0e-14, atol=1.0e-14)


def test_tei_landau_zener_probability_swaps_first_two_weights_at_full_jump():
    oscillation = make_oscillation()
    energy = torch.tensor(10.0, device=DEVICE, dtype=DTYPE)
    density = torch.tensor([0.0, 50.0, 100.0], device=DEVICE, dtype=DTYPE)

    adiabatic = Tei(oscillation, energy, density)
    full_jump = Tei(oscillation, energy, density, p_lz=torch.ones_like(density))

    torch.testing.assert_close(full_jump[..., 0], adiabatic[..., 1], rtol=1.0e-14, atol=1.0e-14)
    torch.testing.assert_close(full_jump[..., 1], adiabatic[..., 0], rtol=1.0e-14, atol=1.0e-14)
    torch.testing.assert_close(full_jump[..., 2], adiabatic[..., 2], rtol=1.0e-14, atol=1.0e-14)
    torch.testing.assert_close(full_jump.sum(dim=-1), torch.ones_like(density), rtol=1.0e-14, atol=1.0e-14)
