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

"""Pytest-compatible checks for perturbative segment evolutors."""

from __future__ import annotations

import torch

from tpeanuts.core.common.hamiltonian import (
    hamiltonian_kinetic_reduced,
    hamiltonian_matter_reduced,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.potential import matter_potential
from tpeanuts.core.perturbative.evolutor import (
    evolutor_first_order,
    evolutor_perturbative_from_H,
    evolutor_perturbative_segment,
    evolutor_zero_order,
)
from tpeanuts.core.perturbative.spectral import hamiltonian_spectral_data
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


class DummyProfileModel:
    """Minimal profile model exposing the evolutor perturbative interface."""

    def __init__(
        self,
        *,
        average: torch.Tensor,
        potential: torch.Tensor,
        length: torch.Tensor,
        zero_mask: torch.Tensor,
        residual_scale: torch.Tensor | float = 0.0,
        perturbation: torch.Tensor | bool = False,
    ) -> None:
        self.average = average
        self.potential = potential
        self.length = length
        self.zero_mask = zero_mask
        self.residual_scale = torch.as_tensor(residual_scale, device=average.device, dtype=average.dtype)
        self.perturbation = torch.as_tensor(perturbation, device=average.device, dtype=torch.bool)

    def residual_integral(self, la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        del la, lb
        shape = torch.broadcast_shapes(self.length.shape, self.residual_scale.shape)
        scale = torch.broadcast_to(self.residual_scale, shape)
        return scale[..., None, None] * torch.ones((*shape, 3, 3), device=self.average.device, dtype=self.average.dtype)

    def has_perturbation(self) -> torch.Tensor:
        shape = torch.broadcast_shapes(self.length.shape, self.perturbation.shape)
        return torch.broadcast_to(self.perturbation, shape)


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_oscillation(*, antinu=False) -> OscillationParameters:
    return OscillationParameters.from_preset("_SM_NUFIT52_NO", antinu=antinu, context=make_context())


def make_hamiltonian() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0 + 0.0j, 0.2 + 0.1j, 0.05 - 0.03j],
            [0.2 - 0.1j, 2.0 + 0.0j, 0.3 + 0.2j],
            [0.05 + 0.03j, 0.3 - 0.2j, 3.0 + 0.0j],
        ],
        device=DEVICE,
        dtype=CDTYPE,
    )


def identity3(batch_shape=()) -> torch.Tensor:
    return torch.eye(3, device=DEVICE, dtype=CDTYPE).expand(*batch_shape, 3, 3)


def assert_unitary(U: torch.Tensor, *, name: str) -> None:
    identity = torch.eye(3, device=U.device, dtype=U.dtype).expand(U.shape)
    assert_close(U.conj().transpose(-2, -1) @ U, identity, name=f"{name} Udag U", atol=1e-10, rtol=1e-10)


def test_evolutor_zero_order_matches_matrix_exp():
    H = make_hamiltonian()
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)

    U0 = evolutor_zero_order(H, length)
    expected = torch.matrix_exp(-1j * H * length)

    assert U0.shape == (3, 3)
    assert_close(U0, expected, name="zero-order evolutor equals matrix_exp", atol=1e-10, rtol=1e-10)
    assert_unitary(U0, name="zero-order evolutor")


def test_evolutor_zero_order_returns_reusable_spectral_data():
    H = make_hamiltonian()
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)

    U0, spectral = evolutor_zero_order(H, length, return_spectral=True)
    U0_from_trace = evolutor_zero_order(H, length, trace_H=spectral["trace"])

    assert set(spectral) == {"T", "trace", "lam", "c1", "M"}
    assert_close(U0_from_trace, U0, name="zero-order with supplied trace")


def test_evolutor_zero_order_enforces_identity_for_zero_length():
    H = make_hamiltonian()
    length = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    zero_mask = torch.tensor(True, device=DEVICE)

    U0 = evolutor_zero_order(H, length, zero_mask=zero_mask)

    assert_close(U0, identity3(), name="zero-length zero-order evolutor")


def test_evolutor_zero_order_batched_matches_matrix_exp():
    H0 = make_hamiltonian()
    H = torch.stack([H0, 0.5 * H0], dim=0)
    length = torch.tensor([0.25, 0.75], device=DEVICE, dtype=DTYPE)

    U0 = evolutor_zero_order(H, length)
    expected = torch.matrix_exp(-1j * H * length[..., None, None])

    assert U0.shape == (2, 3, 3)
    assert_close(U0, expected, name="batched zero-order evolutor", atol=1e-10, rtol=1e-10)
    assert_unitary(U0, name="batched zero-order evolutor")


def test_evolutor_first_order_zero_residual_is_zero():
    H = make_hamiltonian()
    spectral = hamiltonian_spectral_data(H)
    profile = DummyProfileModel(
        average=torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        potential=torch.tensor(0.2, device=DEVICE, dtype=DTYPE),
        length=torch.tensor(0.37, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=0.0,
        perturbation=True,
    )

    U1 = evolutor_first_order(
        spectral["M"],
        spectral["lam"],
        spectral["trace"],
        profile,
    )

    assert_close(U1, torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE), name="zero residual first order")


def test_evolutor_first_order_nonzero_residual_is_finite():
    H = make_hamiltonian()
    spectral = hamiltonian_spectral_data(H)
    profile = DummyProfileModel(
        average=torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        potential=torch.tensor(0.2, device=DEVICE, dtype=DTYPE),
        length=torch.tensor(0.37, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
    )

    U1 = evolutor_first_order(
        spectral["M"],
        spectral["lam"],
        spectral["trace"],
        profile,
    )

    assert U1.shape == (3, 3)
    assert torch.isfinite(U1.real).all()
    assert torch.isfinite(U1.imag).all()
    assert torch.max(torch.abs(U1)) > 0.0


def test_perturbative_from_H_equals_zero_order_when_no_perturbation():
    H = make_hamiltonian()
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        potential=torch.tensor(0.2, device=DEVICE, dtype=DTYPE),
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=False,
    )

    U = evolutor_perturbative_from_H(H, length, profile)
    U0 = evolutor_zero_order(H, length)

    assert_close(U, U0, name="no perturbation mask returns U0")


def test_perturbative_from_H_adds_first_order_when_perturbed():
    H = make_hamiltonian()
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        potential=torch.tensor(0.2, device=DEVICE, dtype=DTYPE),
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
    )

    U = evolutor_perturbative_from_H(H, length, profile)
    U0 = evolutor_zero_order(H, length)

    assert U.shape == (3, 3)
    assert torch.isfinite(U.real).all()
    assert torch.isfinite(U.imag).all()
    assert torch.max(torch.abs(U - U0)) > 0.0


def test_perturbative_from_H_zero_length_is_identity():
    H = make_hamiltonian()
    length = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        potential=torch.tensor(0.2, device=DEVICE, dtype=DTYPE),
        length=length,
        zero_mask=torch.tensor(True, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
    )

    U = evolutor_perturbative_from_H(H, length, profile, zero_mask=profile.zero_mask)

    assert_close(U, identity3(), name="zero-length perturbative evolutor")


def test_perturbative_from_H_batched_masking_and_zero_length():
    H0 = make_hamiltonian()
    H = torch.stack([H0, H0, H0], dim=0)
    length = torch.tensor([0.0, 0.25, 0.5], device=DEVICE, dtype=DTYPE)
    zero_mask = length == 0.0
    profile = DummyProfileModel(
        average=torch.ones(3, device=DEVICE, dtype=DTYPE),
        potential=torch.full((3,), 0.2, device=DEVICE, dtype=DTYPE),
        length=length,
        zero_mask=zero_mask,
        residual_scale=torch.tensor([1.0e-4, 1.0e-4, 1.0e-4], device=DEVICE, dtype=DTYPE),
        perturbation=torch.tensor([True, False, True], device=DEVICE),
    )

    U = evolutor_perturbative_from_H(H, length, profile, zero_mask=zero_mask)
    U0 = evolutor_zero_order(H, length)

    assert U.shape == (3, 3, 3)
    assert_close(U[0], identity3(), name="batched zero-length identity")
    assert_close(U[1], U0[1], name="batched unperturbed entry")
    assert torch.max(torch.abs(U[2] - U0[2])) > 0.0


def test_evolutor_perturbative_segment_constant_profile_equals_manual_U0():
    ctx = make_context()
    osc = make_oscillation()
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential(average, antinu=False, context=ctx)
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=0.0,
        perturbation=False,
    )

    U = evolutor_perturbative_segment(osc, energy, profile)
    Hkin, ki = hamiltonian_kinetic_reduced(
        osc.DeltamSq21,
        osc.DeltamSq3l,
        energy,
        osc.pmns.reduced(),
        return_ki=True,
    )
    H = Hkin + hamiltonian_matter_reduced(potential, context=ctx)
    trace_H = (ki[..., 0] + ki[..., 1] + ki[..., 2] + potential).to(dtype=CDTYPE)
    U0 = evolutor_zero_order(H, length, trace_H=trace_H)

    assert_close(U, U0, name="segment constant profile equals U0", atol=1e-10, rtol=1e-10)


def test_evolutor_perturbative_segment_zero_length_identity():
    ctx = make_context()
    osc = make_oscillation()
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential(average, antinu=False, context=ctx)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=torch.tensor(0.0, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(True, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
    )

    U = evolutor_perturbative_segment(osc, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), profile)

    assert_close(U, identity3(), name="segment zero length identity")
