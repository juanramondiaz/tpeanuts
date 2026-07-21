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
Pytest-compatible tests for tpeanuts.medium.vacuum.evolutor.

The diagnostic plots from the historical backup tests live in notebooks; this
file keeps only fast numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.common.hamiltonian import kinetic_eigenvalue_vector
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.vacuum.evolutor import vacuum_evolutor, vacuum_evolved_state
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3


def _oscillation(*, delta: float = 1.20, antinu: bool = False, context: RuntimeContext | None = None) -> OscillationParameters:
    ctx = context if context is not None else RuntimeContext.resolve(DEVICE, DTYPE)
    pmns = PMNS_SM(PMNSParams(theta12=0.59, theta13=0.15, theta23=0.78, delta=delta, context=ctx))
    mass_spectrum = MassSpectrum_SM(
        DeltamSq21=torch.as_tensor(DM21_EV2, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(DM3L_EV2, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=antinu)


def _sterile_oscillation(*, antinu: bool = False, context: RuntimeContext | None = None) -> OscillationParameters:
    """4-flavour (3+1 sterile) oscillation for N=4 vacuum propagation checks."""
    from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
    from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM

    ctx = context if context is not None else RuntimeContext.resolve(DEVICE, DTYPE)
    sm_params = PMNSParams(theta12=0.5836, theta13=0.1498, theta23=0.8552, delta=3.438, context=ctx)
    sterile_params = PMNSSterileParams(
        theta14=0.15, theta24=0.10, theta34=0.05,
        delta14=0.3, delta24=-0.2, delta34=0.0,
        context=ctx,
    )
    pmns4 = PMNS_sterile(sm_params, sterile_params)
    mass_spectrum = MassSpectrum_BSM(
        DeltamSq21=torch.as_tensor(DM21_EV2, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(DM3L_EV2, device=ctx.device, dtype=ctx.dtype),
        DeltamSq41=torch.as_tensor(1.7, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns4, mass_spectrum=mass_spectrum, antinu=antinu)


def _unitarity_error(S: torch.Tensor) -> torch.Tensor:
    n = S.shape[-1]
    identity = torch.eye(n, device=S.device, dtype=S.dtype)
    left = S.conj().transpose(-1, -2) @ S
    return torch.amax(torch.abs(left - identity), dim=(-2, -1))


def test_zero_baseline_is_identity():
    oscillation = _oscillation()

    S = vacuum_evolutor(
        oscillation,
        torch.tensor(1000.0, device=DEVICE, dtype=DTYPE),
        torch.tensor(0.0, device=DEVICE, dtype=DTYPE),
    )

    identity = torch.eye(3, device=DEVICE, dtype=S.dtype)
    assert_close(S, identity, atol=1.0e-12, rtol=1.0e-12, name="vacuum evolutor is identity at L=0")


def test_evolutor_is_exactly_unitary():
    oscillation = _oscillation()
    E = torch.tensor([250.0, 1000.0, 8000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    S = vacuum_evolutor(oscillation, E, L)

    assert torch.max(_unitarity_error(S)) < 1.0e-13


def test_evolutor_matches_documented_formula():
    oscillation = _oscillation()
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    E = torch.tensor([500.0, 2000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(735.0, device=DEVICE, dtype=DTYPE)

    S = vacuum_evolutor(oscillation, E, L, context=ctx)

    ki = kinetic_eigenvalue_vector(oscillation, E, context=ctx, evolution_scale_m=R_E)
    x = L * 1.0e3 / R_E
    U = oscillation.pmns.pmns_matrix(antinu=oscillation.antinu)
    phase = torch.exp(-1j * ki.to(U.dtype) * x.to(U.dtype))
    expected = (U * phase[..., None, :]) @ U.conj().transpose(-2, -1)

    assert_close(S, expected, name="vacuum_evolutor matches U diag(exp(-i k_i x)) U^dagger")


def test_evolutor_energy_and_baseline_broadcasting():
    oscillation = _oscillation()

    S_paired = vacuum_evolutor(
        oscillation,
        torch.tensor([500.0, 1000.0, 2000.0], device=DEVICE, dtype=DTYPE),
        torch.tensor([100.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE),
    )
    assert S_paired.shape == (3, 3, 3)

    S_scalar_L = vacuum_evolutor(
        oscillation,
        torch.tensor([500.0, 1000.0, 2000.0, 4000.0], device=DEVICE, dtype=DTYPE),
        torch.tensor(1300.0, device=DEVICE, dtype=DTYPE),
    )
    assert S_scalar_L.shape == (4, 3, 3)

    E_grid = torch.tensor([500.0, 1000.0], device=DEVICE, dtype=DTYPE)[:, None]
    L_grid = torch.tensor([100.0, 500.0, 1000.0], device=DEVICE, dtype=DTYPE)[None, :]
    S_outer = vacuum_evolutor(oscillation, E_grid, L_grid)
    assert S_outer.shape == (2, 3, 3, 3)


def test_evolved_state_matches_manual_matrix_multiply():
    oscillation = _oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)
    state = torch.tensor([0.6, 0.8, 0.0], device=DEVICE, dtype=torch.complex128)

    S = vacuum_evolutor(oscillation, E, L)
    evolved = vacuum_evolved_state(state, oscillation, E, L)
    expected = S @ state.to(S.dtype)

    assert_close(evolved, expected, name="vacuum_evolved_state matches S @ state")


def test_legacy_precision_flag_is_a_noop():
    oscillation = _oscillation()
    E = torch.tensor(1500.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(900.0, device=DEVICE, dtype=DTYPE)

    S_default = vacuum_evolutor(oscillation, E, L, legacy_precision=False)
    S_legacy = vacuum_evolutor(oscillation, E, L, legacy_precision=True)

    assert_close(S_legacy, S_default, atol=0.0, rtol=0.0, name="legacy_precision does not alter vacuum kinetic phases")


def test_evolution_scale_invariance():
    oscillation = _oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    S_default = vacuum_evolutor(oscillation, E, L, evolution_scale_m=R_E)
    S_double = vacuum_evolutor(oscillation, E, L, evolution_scale_m=2.0 * R_E)
    S_small = vacuum_evolutor(oscillation, E, L, evolution_scale_m=0.1 * R_E)

    assert_close(S_double, S_default, name="physical evolutor is invariant under evolution_scale_m rescaling")
    assert_close(S_small, S_default, name="physical evolutor is invariant under evolution_scale_m rescaling")


def test_non_positive_evolution_scale_raises():
    oscillation = _oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    with pytest.raises(ValueError, match="evolution_scale_m must be positive"):
        vacuum_evolutor(oscillation, E, L, evolution_scale_m=torch.tensor(0.0, device=DEVICE, dtype=DTYPE))


def test_sterile_evolutor_is_exactly_unitary_n4():
    """Regression test: vacuum_evolutor used to hardcode 3 flavours when
    broadcasting the kinetic eigenvalue vector, crashing outright for a
    4-flavour (3+1 sterile) oscillation object (shape mismatch 3 vs 4)."""
    oscillation = _sterile_oscillation()
    E = torch.tensor([250.0, 1000.0, 8000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    S = vacuum_evolutor(oscillation, E, L)

    assert S.shape == (3, 4, 4)
    assert torch.max(_unitarity_error(S)) < 1.0e-12


def test_sterile_evolutor_matches_documented_formula_n4():
    oscillation = _sterile_oscillation()
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    E = torch.tensor([500.0, 2000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(735.0, device=DEVICE, dtype=DTYPE)

    S = vacuum_evolutor(oscillation, E, L, context=ctx)

    ki = kinetic_eigenvalue_vector(oscillation, E, context=ctx, evolution_scale_m=R_E)
    x = L * 1.0e3 / R_E
    U = oscillation.pmns.pmns_matrix(antinu=oscillation.antinu)
    phase = torch.exp(-1j * ki.to(U.dtype) * x.to(U.dtype))
    expected = (U * phase[..., None, :]) @ U.conj().transpose(-2, -1)

    assert S.shape[-2:] == (4, 4)
    assert_close(S, expected, name="N=4 vacuum_evolutor matches U diag(exp(-i k_i x)) U^dagger")


def test_sterile_evolved_state_matches_manual_matrix_multiply_n4():
    oscillation = _sterile_oscillation()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)
    state = torch.tensor([0.6, 0.8, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)

    S = vacuum_evolutor(oscillation, E, L)
    evolved = vacuum_evolved_state(state, oscillation, E, L)
    expected = S @ state.to(S.dtype)

    assert_close(evolved, expected, name="N=4 vacuum_evolved_state matches S @ state")


def test_context_none_infers_dtype_from_inputs():
    oscillation = _oscillation(context=RuntimeContext.resolve(DEVICE, torch.float32))
    E = torch.tensor(1000.0, device=DEVICE, dtype=torch.float32)
    L = torch.tensor(1300.0, device=DEVICE, dtype=torch.float32)

    S = vacuum_evolutor(oscillation, E, L, context=None)

    assert S.dtype == torch.complex64
    assert S.device.type == DEVICE.type
