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
Pytest-compatible tests for tpeanuts.core.common.potential.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.potential import kinetic_potential, matter_potential
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _matter_factor_full_precision() -> float:
    return (
        math.sqrt(2.0)
        * constant.G_F_MEV_M2
        * constant.N_A
        * 1.0e6
        * constant.HBARC_MeV_m ** 2
    )


def test_matter_potential_full_precision_formula():
    n = torch.tensor([0.0, 1.0, 2.0, 5.0], device=DEVICE, dtype=DTYPE)

    V = matter_potential(n, antinu=False)
    expected = constant.R_E * _matter_factor_full_precision() * n

    assert_close(V, expected, name="full-precision matter potential formula")


def test_matter_potential_legacy_formula():
    n = torch.tensor([0.0, 1.0, 2.0, 5.0], device=DEVICE, dtype=DTYPE)

    V = matter_potential(n, antinu=False, legacy_precision=True)
    expected = constant.R_E * 3.868e-7 * n

    assert_close(V, expected, name="legacy matter potential formula")


def test_matter_potential_antinu_sign_scalar_and_tensor():
    n = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        device=DEVICE,
        dtype=DTYPE,
    )

    V_nu = matter_potential(n, antinu=False)
    V_anti = matter_potential(n, antinu=True)
    assert_close(V_anti, -V_nu, name="scalar antinu flips matter sign")

    antinu = torch.tensor([False, True, False], device=DEVICE)
    V_mixed = matter_potential(n, antinu=antinu)
    expected_sign = torch.tensor([[1.0], [-1.0], [1.0]], device=DEVICE, dtype=DTYPE)
    assert_close(V_mixed, expected_sign * V_nu, name="tensor antinu broadcasts")


def test_matter_potential_shape_dtype_device():
    n = torch.ones((4, 5), device=DEVICE, dtype=DTYPE)

    V = matter_potential(n, antinu=False)

    assert V.shape == n.shape
    assert V.dtype == n.dtype
    assert V.device == n.device


def test_potentials_accept_tensorlike_with_context():
    context = RuntimeContext.resolve(DEVICE, torch.float32)

    V = matter_potential([0.0, 1.0, 2.0], antinu=False, context=context)
    k = kinetic_potential([0.0, 7.42e-5, 2.517e-3], 1000.0, context=context)

    assert V.dtype == torch.float32
    assert k.dtype == torch.float32
    assert V.device.type == context.device.type
    assert k.device.type == context.device.type
    assert V.shape == (3,)
    assert k.shape == (3,)


def test_kinetic_potential_scalar_energy_formula():
    mSq = torch.tensor([0.0, 7.42e-5, 2.517e-3], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    k = kinetic_potential(mSq, E)
    expected = constant.R_E * 0.5 * 1.0e-12 * mSq / E / constant.HBARC_MeV_m

    assert k.shape == (3,)
    assert_close(k, expected, name="kinetic potential scalar-energy formula")


def test_kinetic_potential_energy_grid_broadcasting():
    mSq = torch.tensor([0.0, 7.42e-5, 2.517e-3], device=DEVICE, dtype=DTYPE)
    E = torch.logspace(2.0, 5.0, 8, device=DEVICE, dtype=DTYPE)

    k = kinetic_potential(mSq, E)
    expected = constant.R_E * 0.5 * 1.0e-12 * mSq[None, :] / E[:, None] / constant.HBARC_MeV_m

    assert k.shape == (E.numel(), 3)
    assert_close(k, expected, name="kinetic potential energy-grid broadcasting")


def test_kinetic_potential_batched_mass_and_energy():
    mSq = torch.tensor(
        [
            [0.0, 7.42e-5, 2.517e-3],
            [0.0, 7.42e-5, 2.517e-3],
            [0.0, 7.42e-5, 2.517e-3],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    E = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)

    k = kinetic_potential(mSq, E)
    expected = constant.R_E * 0.5 * 1.0e-12 * mSq / E[:, None] / constant.HBARC_MeV_m

    assert k.shape == (3, 3)
    assert_close(k, expected, name="kinetic potential batched formula")


def test_kinetic_potential_inverse_energy_scaling():
    mSq = torch.tensor([0.0, 7.42e-5, 2.517e-3], device=DEVICE, dtype=DTYPE)

    k1 = kinetic_potential(mSq, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE))
    k2 = kinetic_potential(mSq, torch.tensor(2000.0, device=DEVICE, dtype=DTYPE))

    assert_close(k1[1:] / k2[1:], torch.full_like(k1[1:], 2.0), name="k scales as 1/E")


def test_evolution_scale_scales_matter_and_kinetic_terms():
    n = torch.tensor([1.0, 2.0], device=DEVICE, dtype=DTYPE)
    mSq = torch.tensor([7.42e-5, 2.517e-3], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    scale = torch.tensor(1234.0, device=DEVICE, dtype=DTYPE)
    V = matter_potential(n, antinu=False, evolution_scale_m=scale)
    V2 = matter_potential(n, antinu=False, evolution_scale_m=2.0 * scale)
    k = kinetic_potential(mSq, E, evolution_scale_m=scale)
    k2 = kinetic_potential(mSq, E, evolution_scale_m=2.0 * scale)

    assert_close(V2, 2.0 * V, name="matter potential scales linearly with length scale")
    assert_close(k2, 2.0 * k, name="kinetic potential scales linearly with length scale")


def test_non_positive_evolution_scale_raises():
    n = torch.tensor([1.0], device=DEVICE, dtype=DTYPE)
    mSq = torch.tensor([7.42e-5], device=DEVICE, dtype=DTYPE)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="evolution_scale_m must be positive"):
        matter_potential(n, antinu=False, evolution_scale_m=torch.tensor(0.0, device=DEVICE, dtype=DTYPE))

    with pytest.raises(ValueError, match="evolution_scale_m must be positive"):
        kinetic_potential(mSq, E, evolution_scale_m=torch.tensor(-1.0, device=DEVICE, dtype=DTYPE))
