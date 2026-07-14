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

"""Pytest-compatible checks for medium-independent numerical evolutors."""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.evolutor import compose_segment_evolutors
from tpeanuts.core.common.hamiltonian import hamiltonian_flavour
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.evolutor import (
    evolutor_numerical,
    evolutor_numerical_segment,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_oscillation(*, antinu=False) -> OscillationParameters:
    return OscillationParameters.from_preset("_SM_NUFIT52_NO", antinu=antinu, context=make_context())


def identity3(batch_shape=()) -> torch.Tensor:
    return torch.eye(3, device=DEVICE, dtype=CDTYPE).expand(*batch_shape, 3, 3)


def manual_segment_evolutors(
    oscillation: OscillationParameters,
    E_MeV: torch.Tensor,
    n_e: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    E = E_MeV
    while E.ndim < n_e.ndim:
        E = E.unsqueeze(-1)
    H = hamiltonian_flavour(
        oscillation,
        E,
        n_e,
        context=make_context(),
    )
    return torch.linalg.matrix_exp(-1j * H * dx[..., None, None].to(CDTYPE))


def assert_unitary(U: torch.Tensor, *, name: str, atol=1.0e-10) -> None:
    identity = torch.eye(3, device=U.device, dtype=U.dtype).expand(U.shape)
    residual = U.conj().transpose(-2, -1) @ U - identity
    assert_close(residual, torch.zeros_like(residual), name=name, atol=atol, rtol=atol)


def test_evolutor_numerical_segment_matches_manual_matrix_exp():
    osc = make_oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.05, 0.03, 0.02], device=DEVICE, dtype=DTYPE)

    U_steps = evolutor_numerical_segment(osc, E, n_e, dx, device=DEVICE, dtype=DTYPE)
    expected = manual_segment_evolutors(osc, E, n_e, dx)

    assert U_steps.shape == (3, 3, 3)
    assert_close(U_steps, expected, name="numerical segment matrix exponentials", atol=1.0e-11, rtol=1.0e-11)
    assert_unitary(U_steps, name="segment unitarity")


def test_evolutor_numerical_segment_broadcasts_energy_and_dx():
    osc = make_oscillation()
    E = torch.tensor([500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(
        [
            [1.0, 1.1, 1.2],
            [1.4, 1.5, 1.6],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    U_steps = evolutor_numerical_segment(osc, E, n_e, dx, device=DEVICE, dtype=DTYPE)
    expected = manual_segment_evolutors(osc, E, n_e, dx.broadcast_to(n_e.shape))

    assert U_steps.shape == (2, 3, 3, 3)
    assert_close(U_steps, expected, name="batched numerical segment evolutors", atol=1.0e-11, rtol=1.0e-11)


def test_evolutor_numerical_segment_zero_dx_returns_identity_steps():
    osc = make_oscillation()
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.zeros_like(n_e)

    U_steps = evolutor_numerical_segment(osc, 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    assert_close(U_steps, identity3((3,)), name="zero dx segment identity")


def test_evolutor_numerical_segment_rejects_scalar_density_without_segment_axis():
    osc = make_oscillation()

    with pytest.raises(ValueError, match="segment dimension"):
        evolutor_numerical_segment(osc, 1000.0, torch.tensor(1.0, device=DEVICE, dtype=DTYPE), 0.1, device=DEVICE, dtype=DTYPE)


def test_evolutor_numerical_composes_segments_like_common_evolutor():
    osc = make_oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor([1.0, 1.2, 1.4, 1.1], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04, 0.01], device=DEVICE, dtype=DTYPE)

    U_steps = evolutor_numerical_segment(osc, E, n_e, dx, device=DEVICE, dtype=DTYPE)
    U_total = evolutor_numerical(osc, E, n_e, dx, device=DEVICE, dtype=DTYPE)
    expected = compose_segment_evolutors(U_steps, segment_dim=-3, multiply="left")

    assert U_total.shape == (3, 3)
    assert_close(U_total, expected, name="numerical total composition", atol=1.0e-11, rtol=1.0e-11)
    assert_unitary(U_total, name="total numerical unitarity")


def test_evolutor_numerical_history_starts_with_identity_and_ends_with_total():
    osc = make_oscillation()
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    history = evolutor_numerical(osc, 1000.0, n_e, dx, return_history=True, device=DEVICE, dtype=DTYPE)
    total = evolutor_numerical(osc, 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    assert history.shape == (4, 3, 3)
    assert_close(history[0], identity3(), name="history starts with identity")
    assert_close(history[-1], total, name="history ends with total evolutor", atol=1.0e-11, rtol=1.0e-11)


def test_evolutor_numerical_history_matches_sequential_left_accumulation():
    osc = make_oscillation()
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)
    U_steps = evolutor_numerical_segment(osc, 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    history = evolutor_numerical(osc, 1000.0, n_e, dx, return_history=True, device=DEVICE, dtype=DTYPE)
    sequential = [identity3().clone()]
    running = identity3().clone()
    for j in range(U_steps.shape[-3]):
        running = U_steps[j] @ running
        sequential.append(running.clone())
    expected = torch.stack(sequential, dim=0)

    assert_close(history, expected, name="history sequential accumulation", atol=1.0e-11, rtol=1.0e-11)


def test_evolutor_numerical_batched_history_shape_and_final_slice():
    osc = make_oscillation()
    E = torch.tensor([500.0, 1000.0], device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(
        [
            [1.0, 1.1, 1.2],
            [1.4, 1.5, 1.6],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    history = evolutor_numerical(osc, E, n_e, dx, return_history=True, device=DEVICE, dtype=DTYPE)
    total = evolutor_numerical(osc, E, n_e, dx, device=DEVICE, dtype=DTYPE)

    assert history.shape == (2, 4, 3, 3)
    assert_close(history[:, 0], identity3((2,)), name="batched history identity")
    assert_close(history[:, -1], total, name="batched history final slice", atol=1.0e-11, rtol=1.0e-11)


def test_evolutor_numerical_antineutrino_is_finite_and_differs_from_neutrino():
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    U_nu = evolutor_numerical(make_oscillation(antinu=False), 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)
    U_anti = evolutor_numerical(make_oscillation(antinu=True), 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    assert torch.isfinite(U_nu.real).all()
    assert torch.isfinite(U_nu.imag).all()
    assert torch.isfinite(U_anti.real).all()
    assert torch.isfinite(U_anti.imag).all()
    assert torch.max(torch.abs(U_nu - U_anti)) > 0.0


def test_evolutor_numerical_accepts_tensor_antinu_steps():
    antinu = torch.tensor([False, True, False], device=DEVICE)
    osc = make_oscillation(antinu=antinu)
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    U_steps = evolutor_numerical_segment(osc, 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    assert U_steps.shape == (3, 3, 3)
    assert torch.isfinite(U_steps.real).all()
    assert torch.isfinite(U_steps.imag).all()
