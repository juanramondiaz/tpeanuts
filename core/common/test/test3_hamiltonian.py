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
Pytest-compatible checks for the Standard Model core Hamiltonian builders.
"""

from __future__ import annotations

import pytest
import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_flavour,
    hamiltonian_kinetic_reduced,
    hamiltonian_matter_reduced,
    hamiltonian_reduced,
    kinetic_mass_squared_vector,
    kinetic_mass_vector,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.potential import kinetic_potential, matter_potential
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def make_context(dtype: torch.dtype = DTYPE) -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, dtype)


def make_oscillation(*, antinu=False, context: RuntimeContext | None = None) -> OscillationParameters:
    return OscillationParameters.from_preset(
        "_SM_NUFIT52_NO",
        antinu=antinu,
        context=context or make_context(),
    )


def test_kinetic_mass_squared_vector_normal_ordering():
    ctx = make_context()
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)

    mass_vector = kinetic_mass_squared_vector(dm21, dm3l, context=ctx)
    expected = torch.stack([torch.zeros_like(dm21), dm21, dm3l])

    assert mass_vector.shape == (3,)
    assert_close(mass_vector, expected, name="normal-ordering mass vector")


def test_kinetic_mass_squared_vector_inverted_ordering():
    ctx = make_context()
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(-2.498e-3, device=DEVICE, dtype=DTYPE)

    mass_vector = kinetic_mass_squared_vector(dm21, dm3l, context=ctx)
    expected = torch.stack([-dm21, torch.zeros_like(dm21), dm3l])

    assert mass_vector.shape == (3,)
    assert_close(mass_vector, expected, name="inverted-ordering mass vector")


def test_kinetic_mass_squared_vector_batched_ordering():
    ctx = make_context()
    dm21 = torch.tensor([7.42e-5, 7.42e-5], device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor([2.517e-3, -2.498e-3], device=DEVICE, dtype=DTYPE)

    mass_vector = kinetic_mass_squared_vector(dm21, dm3l, context=ctx)
    expected = torch.tensor(
        [
            [0.0, 7.42e-5, 2.517e-3],
            [-7.42e-5, 0.0, -2.498e-3],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    assert mass_vector.shape == (2, 3)
    assert_close(mass_vector, expected, name="batched ordering mass vector")


def test_kinetic_mass_vector_matches_kinetic_potential():
    ctx = make_context()
    dm21 = torch.tensor(7.42e-5, device=DEVICE, dtype=DTYPE)
    dm3l = torch.tensor(2.517e-3, device=DEVICE, dtype=DTYPE)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)

    mass_vector = kinetic_mass_squared_vector(dm21, dm3l, context=ctx)
    kinetic = kinetic_mass_vector(dm21, dm3l, energy, context=ctx)
    expected = kinetic_potential(mass_vector, energy, context=ctx)

    assert kinetic.shape == (3,)
    assert_close(kinetic, expected, name="kinetic mass vector")


def test_kinetic_mass_vector_energy_grid_broadcasting():
    ctx = make_context()
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)

    kinetic = kinetic_mass_vector(7.42e-5, 2.517e-3, energy, context=ctx)

    assert kinetic.shape == (3, 3)
    assert torch.isfinite(kinetic).all()
    assert_close(kinetic[0, 1:] / kinetic[1, 1:], torch.full((2,), 2.0, device=DEVICE, dtype=DTYPE))


def test_hamiltonian_kinetic_reduced_formula():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Ured = osc.pmns.reduced()

    Hkin, ki = hamiltonian_kinetic_reduced(
        osc.DeltamSq21,
        osc.DeltamSq3l,
        energy,
        Ured,
        return_ki=True,
    )
    expected = (Ured * ki.to(dtype=Ured.dtype)[..., None, :]) @ Ured.transpose(-1, -2)

    assert Hkin.shape == (3, 3)
    assert ki.shape == (3,)
    assert Hkin.dtype == CDTYPE
    assert_close(Hkin, expected, name="Hkin = Ured diag(ki) Ured^T")


def test_hamiltonian_kinetic_reduced_batched_energy():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)

    Hkin, ki = hamiltonian_kinetic_reduced(
        osc.DeltamSq21,
        osc.DeltamSq3l,
        energy,
        osc.pmns.reduced(),
        return_ki=True,
    )

    assert Hkin.shape == (3, 3, 3)
    assert ki.shape == (3, 3)
    assert torch.isfinite(Hkin.real).all()
    assert torch.isfinite(Hkin.imag).all()


def test_hamiltonian_kinetic_reduced_invalid_shape_raises():
    with pytest.raises(ValueError, match="Ured must have final dimensions"):
        hamiltonian_kinetic_reduced(
            7.42e-5,
            2.517e-3,
            1000.0,
            torch.eye(2, device=DEVICE, dtype=CDTYPE),
        )


def test_hamiltonian_matter_reduced_formula():
    V = torch.tensor(1.2345, device=DEVICE, dtype=DTYPE)

    Hmat = hamiltonian_matter_reduced(V, context=make_context())
    expected = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    expected[0, 0] = V.to(dtype=CDTYPE)

    assert Hmat.shape == (3, 3)
    assert_close(Hmat, expected, name="Hmat = diag(V, 0, 0)")


def test_hamiltonian_matter_reduced_batched():
    V = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)

    Hmat = hamiltonian_matter_reduced(V, context=make_context())

    assert Hmat.shape == (3, 3, 3)
    assert_close(Hmat[:, 0, 0].real, V, name="batched matter diagonal")
    assert_close(Hmat[:, 1:, :], torch.zeros((3, 2, 3), device=DEVICE, dtype=CDTYPE))


def test_hamiltonian_reduced_equals_kinetic_plus_matter():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    V = matter_potential(n_e, antinu=osc.antinu, context=ctx)
    Hkin = hamiltonian_kinetic_reduced(
        osc.DeltamSq21,
        osc.DeltamSq3l,
        energy,
        osc.pmns.reduced(antinu=osc.antinu),
    )
    Hmat = hamiltonian_matter_reduced(V, context=ctx)

    assert H.shape == (3, 3)
    assert_close(H, Hkin + Hmat, name="H_reduced = Hkin + Hmat")


def test_hamiltonian_reduced_batched_energy_and_density():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor([0.5, 1.0, 1.5], device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx)

    assert H.shape == (3, 3, 3)
    assert H.dtype == CDTYPE
    assert torch.isfinite(H.real).all()
    assert torch.isfinite(H.imag).all()


def test_hamiltonian_reduced_context_none_uses_pmns_device_dtype():
    ctx = make_context()
    osc = make_oscillation(context=ctx)

    H = hamiltonian_reduced(osc, 1000.0, 1.0)

    assert H.shape == (3, 3)
    assert H.device.type == ctx.device.type
    assert H.dtype == CDTYPE


def test_hamiltonian_flavour_matches_pmns_basis_transform():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    Hred = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    Hflav = hamiltonian_flavour(osc, energy, n_e, context=ctx)
    expected = osc.pmns.H_flavour_basis(
        Hred,
        antinu=osc.antinu,
        device=Hred.device,
        dtype=Hred.dtype,
    )

    assert Hflav.shape == (3, 3)
    assert_close(Hflav, expected, name="flavour-basis Hamiltonian transform")


def test_hamiltonian_antinu_matches_manual_sign_and_conjugation():
    ctx = make_context()
    osc = make_oscillation(antinu=True, context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    V = matter_potential(n_e, antinu=True, context=ctx)
    Hkin = hamiltonian_kinetic_reduced(
        osc.DeltamSq21,
        osc.DeltamSq3l,
        energy,
        osc.pmns.reduced(antinu=True),
    )
    Hmat = hamiltonian_matter_reduced(V, context=ctx)

    assert_close(H, Hkin + Hmat, name="antinu reduced Hamiltonian")


def test_evolution_scale_scales_reduced_hamiltonian_linearly():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    scale = torch.tensor(1234.0, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx, evolution_scale_m=scale)
    H2 = hamiltonian_reduced(osc, energy, n_e, context=ctx, evolution_scale_m=2.0 * scale)

    assert_close(H2, 2.0 * H, name="Hamiltonian scales linearly with evolution scale")


def test_hamiltonian_reduced_legacy_precision_changes_only_matter_term():
    ctx = make_context()
    osc = make_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_full = hamiltonian_reduced(osc, energy, n_e, context=ctx, legacy_precision=False)
    H_legacy = hamiltonian_reduced(osc, energy, n_e, context=ctx, legacy_precision=True)
    V_full = matter_potential(n_e, antinu=False, context=ctx, legacy_precision=False)
    V_legacy = matter_potential(n_e, antinu=False, context=ctx, legacy_precision=True)
    expected_delta = hamiltonian_matter_reduced(V_legacy - V_full, context=ctx)

    assert_close(H_legacy - H_full, expected_delta, name="legacy precision matter-only delta")
