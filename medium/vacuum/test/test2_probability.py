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
Pytest-compatible tests for tpeanuts.medium.vacuum.probability.

Includes validation against the legacy peanuts implementation via
``tpeanuts.medium.vacuum.validation`` (``compare_vacuum_probability_state_with_legacy``,
``compare_vacuum_evolved_state_with_legacy``), which already handles the
legacy import and object construction. The diagnostic plots from the
historical backup tests live in notebooks; this file keeps only fast
numerical sanity checks that can run automatically.
"""

from __future__ import annotations

import torch

import pytest

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.SM.sm_mass_spectrum import MassSpectrum_SM
from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.medium.vacuum.probability import vacuum_probability_state, vacuum_probability_transition
from tpeanuts.medium.vacuum import validation as legacy_validation
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close

try:
    import peanuts  # noqa: F401
    _LEGACY_AVAILABLE = True
except Exception:
    _LEGACY_AVAILABLE = False


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DM21_EV2 = 7.42e-5
DM3L_EV2 = 2.517e-3


def _oscillation(*, delta: float = 1.20, antinu: bool = False) -> OscillationParameters:
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    pmns = PMNS_SM(PMNSParams(theta12=0.59, theta13=0.15, theta23=0.78, delta=delta, context=ctx))
    mass_spectrum = MassSpectrum_SM(
        DeltamSq21=torch.as_tensor(DM21_EV2, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(DM3L_EV2, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns, mass_spectrum=mass_spectrum, antinu=antinu)


def _sterile_oscillation(*, antinu: bool = False) -> OscillationParameters:
    """4-flavour (3+1 sterile) oscillation for N=4 vacuum probability checks."""
    from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
    from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM

    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
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


def _assert_probability_vector(P: torch.Tensor, atol: float = 1.0e-10) -> None:
    row_sum = torch.sum(P, dim=-1)
    assert torch.all(torch.isfinite(P))
    assert torch.all(P >= -atol)
    assert_close(row_sum, torch.ones_like(row_sum), atol=atol, rtol=atol, name="probabilities sum to one")


def test_vacuum_probability_matrix_is_doubly_stochastic():
    oscillation = _oscillation()
    E = torch.tensor([250.0, 1000.0, 3000.0, 10000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(735.0, device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_transition(oscillation, E, L, context=RuntimeContext.resolve(DEVICE, DTYPE))

    assert P.shape == (4, 3, 3)
    assert torch.all(torch.isfinite(P))
    col_sums = P.sum(dim=-2)
    row_sums = P.sum(dim=-1)
    assert_close(col_sums, torch.ones_like(col_sums), atol=1.0e-10, rtol=1.0e-10, name="each initial flavour column sums to one")
    assert_close(row_sums, torch.ones_like(row_sums), atol=1.0e-10, rtol=1.0e-10, name="each final flavour row sums to one")


def test_pvacuum_coherent_flavourbasis_sums_to_one():
    oscillation = _oscillation()
    E = torch.linspace(500.0, 5000.0, 7, device=DEVICE, dtype=DTYPE)
    initial_mu = torch.tensor([0.0, 1.0, 0.0], device=DEVICE, dtype=torch.complex128)

    P = vacuum_probability_state(initial_mu, oscillation, E, torch.tensor(1300.0, device=DEVICE, dtype=DTYPE), massbasis=False)

    assert P.shape == (7, 3)
    _assert_probability_vector(P)


def test_pvacuum_massbasis_weights_normalized_and_nonnegative():
    oscillation = _oscillation()
    weights = torch.tensor([0.20, 0.30, 0.50], device=DEVICE, dtype=DTYPE)
    baselines = torch.tensor([0.0, 295.0, 1300.0, 12000.0], device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(weights, oscillation, torch.tensor(2000.0, device=DEVICE, dtype=DTYPE), baselines, massbasis=True)

    assert P.shape == (4, 3)
    _assert_probability_vector(P)


def test_pvacuum_zero_baseline_returns_input_flavour():
    oscillation = _oscillation()
    state = torch.tensor([0.6, 0.8, 0.0], device=DEVICE, dtype=torch.complex128)

    P = vacuum_probability_state(state, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), torch.tensor(0.0, device=DEVICE, dtype=DTYPE), massbasis=False)

    assert_close(P, torch.abs(state) ** 2, name="zero-baseline coherent probability equals |input amplitude|^2")


def test_pvacuum_massbasis_zero_baseline_matches_pmns_column():
    oscillation = _oscillation()
    weights = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(weights, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), torch.tensor(0.0, device=DEVICE, dtype=DTYPE), massbasis=True)
    expected = torch.abs(oscillation.pmns.pmns_matrix()[:, 0]) ** 2

    assert_close(P, expected.real, name="zero-baseline mass-basis probability reduces to |U_alpha1|^2")


def test_pvacuum_antineutrino_equals_neutrino_when_delta_zero():
    oscillation_nu = _oscillation(delta=0.0, antinu=False)
    oscillation_anu = _oscillation(delta=0.0, antinu=True)
    initial_e = torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)
    E = torch.tensor([300.0, 800.0, 2500.0, 8000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(1300.0, device=DEVICE, dtype=DTYPE)

    P_nu = vacuum_probability_state(initial_e, oscillation_nu, E, L, massbasis=False)
    P_anu = vacuum_probability_state(initial_e, oscillation_anu, E, L, massbasis=False)

    assert_close(P_nu, P_anu, atol=1.0e-12, rtol=1.0e-12, name="nu and antinu vacuum probabilities match when delta=0")


def test_vacuum_probability_matches_pvacuum_coherent_column():
    oscillation = _oscillation()
    E = torch.tensor(1200.0, device=DEVICE, dtype=DTYPE)
    L = torch.tensor(850.0, device=DEVICE, dtype=DTYPE)

    P_matrix = vacuum_probability_transition(oscillation, E, L, context=RuntimeContext.resolve(DEVICE, DTYPE))

    for initial_index in range(3):
        state = torch.zeros(3, device=DEVICE, dtype=torch.complex128)
        state[initial_index] = 1.0
        P_coherent = vacuum_probability_state(state, oscillation, E, L, massbasis=False)
        assert_close(P_coherent, P_matrix[:, initial_index], name=f"vacuum_probability_state column matches vacuum_probability_transition for initial index {initial_index}")


def test_sterile_vacuum_probability_matrix_is_doubly_stochastic_n4():
    """Regression test: vacuum_probability_transition (via vacuum_evolutor)
    used to crash for a 4-flavour (3+1 sterile) oscillation object."""
    oscillation = _sterile_oscillation()
    E = torch.tensor([250.0, 1000.0, 3000.0, 10000.0], device=DEVICE, dtype=DTYPE)
    L = torch.tensor(735.0, device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_transition(oscillation, E, L, context=RuntimeContext.resolve(DEVICE, DTYPE))

    assert P.shape == (4, 4, 4)
    assert torch.all(torch.isfinite(P))
    col_sums = P.sum(dim=-2)
    row_sums = P.sum(dim=-1)
    assert_close(col_sums, torch.ones_like(col_sums), atol=1.0e-10, rtol=1.0e-10, name="N=4 each initial flavour column sums to one")
    assert_close(row_sums, torch.ones_like(row_sums), atol=1.0e-10, rtol=1.0e-10, name="N=4 each final flavour row sums to one")


def test_sterile_pvacuum_massbasis_weights_normalized_and_nonnegative_n4():
    oscillation = _sterile_oscillation()
    weights = torch.tensor([0.20, 0.30, 0.40, 0.10], device=DEVICE, dtype=DTYPE)
    baselines = torch.tensor([0.0, 295.0, 1300.0, 12000.0], device=DEVICE, dtype=DTYPE)

    P = vacuum_probability_state(weights, oscillation, torch.tensor(2000.0, device=DEVICE, dtype=DTYPE), baselines, massbasis=True)

    assert P.shape == (4, 4)
    _assert_probability_vector(P)


def test_sterile_pvacuum_zero_baseline_returns_input_flavour_n4():
    oscillation = _sterile_oscillation()
    state = torch.tensor([0.6, 0.8, 0.0, 0.0], device=DEVICE, dtype=torch.complex128)

    P = vacuum_probability_state(state, oscillation, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), torch.tensor(0.0, device=DEVICE, dtype=DTYPE), massbasis=False)

    assert_close(P, torch.abs(state) ** 2, name="N=4 zero-baseline coherent probability equals |input amplitude|^2")


@pytest.mark.skipif(not _LEGACY_AVAILABLE, reason="legacy peanuts reference package not available")
def test_pvacuum_matches_legacy():
    oscillation = _oscillation()
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)

    cases = [
        ("flavour nu_e", [1.0, 0.0, 0.0], False, 900.0, 295.0),
        ("flavour nu_mu", [0.0, 1.0, 0.0], False, 2500.0, 1300.0),
        ("mass weights", [0.20, 0.30, 0.50], True, 5000.0, 12000.0),
    ]

    max_errors = []
    for _, state, massbasis, energy, baseline in cases:
        result = legacy_validation.compare_vacuum_probability_state_with_legacy(
            state, oscillation, energy, baseline, massbasis=massbasis, context=ctx,
        )
        max_errors.append(result["max_abs"])

    assert max(max_errors) < 1.0e-10, f"vacuum_probability_state vs legacy Pvacuum exceeds tolerance: {max(max_errors):.3e}"


@pytest.mark.skipif(not _LEGACY_AVAILABLE, reason="legacy peanuts reference package not available")
@pytest.mark.xfail(
    reason=(
        "peanuts.vacuum.vacuum_evolved_state calls Upert(..., 0, l, ...) while "
        "the validated peanuts.vacuum.Pvacuum calls Upert(..., l, 0, ...) for the "
        "same physical baseline -- a genuine argument-order bug in the read-only "
        "legacy reference (not in tpeanuts). The underlying physics is already "
        "validated tightly via test_pvacuum_matches_legacy, which goes through "
        "the correct Pvacuum path."
    ),
    strict=True,
)
def test_vacuum_evolved_state_matches_legacy():
    oscillation = _oscillation()
    ctx = RuntimeContext.resolve(DEVICE, DTYPE)
    state = [0.6, 0.8, 0.0]

    result = legacy_validation.compare_vacuum_evolved_state_with_legacy(
        state, oscillation, 1000.0, 1300.0, context=ctx,
    )

    assert result["max_abs"] < 1.0e-10, f"vacuum_evolved_state vs legacy exceeds tolerance: {result['max_abs']:.3e}"
