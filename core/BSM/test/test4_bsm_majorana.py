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
Pytest-compatible checks specific to the Majorana-neutrino extension:
``tpeanuts.core.BSM.bsm_majorana.PMNS_Majorana``/``MajoranaPhases``/
``effective_majorana_mass``, and its wiring through
``tpeanuts.core.common.oscillation.OscillationParameters.BSM_extension_majorana``
and ``tpeanuts.config.propagation.PropagationConfig
.oscillation_parameters_from_preset(neutrino_nature=...)``.

The central physical claim under test is that Majorana phases are provably
inert for oscillation probabilities (see ``bsm_majorana``'s module
docstring): ``PMNS_Majorana`` must reproduce every ``PMNS_SM`` result
exactly up to the phase factor P, and a vacuum oscillation amplitude built
from a Majorana PMNS matrix must equal the Dirac (``PMNS_SM``) one for
arbitrary Majorana phases -- in contrast to the effective Majorana mass,
which genuinely does depend on them.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.core.BSM.bsm_majorana import (
    MajoranaPhases,
    PMNS_Majorana,
    effective_majorana_mass,
)
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sm_params(
    theta12: float = 0.59,
    theta13: float = 0.15,
    theta23: float = 0.78,
    delta: float = 1.20,
) -> PMNSParams:
    context = RuntimeContext.resolve(DEVICE, DTYPE)
    return PMNSParams(
        theta12=theta12, theta13=theta13, theta23=theta23, delta=delta, context=context,
    )


def make_dirac(**kwargs) -> PMNS_SM:
    return PMNS_SM(make_sm_params(**kwargs))


def make_majorana(alpha21: float, alpha31: float, **kwargs) -> PMNS_Majorana:
    context = RuntimeContext.resolve(DEVICE, DTYPE)
    majorana_params = MajoranaPhases(alpha21=alpha21, alpha31=alpha31, context=context)
    return PMNS_Majorana(make_sm_params(**kwargs), majorana_params)


# ---------------------------------------------------------------------------
# PMNS_Majorana structure, unitarity, and relation to PMNS_SM
# ---------------------------------------------------------------------------

def test_majorana_phase_matrix_entries():
    alpha21, alpha31 = 0.7, 2.3
    pmns = make_majorana(alpha21, alpha31)

    expected = torch.eye(3, device=DEVICE, dtype=torch.complex128)
    expected[1, 1] = torch.exp(torch.tensor(1j * alpha21 / 2, device=DEVICE, dtype=torch.complex128))
    expected[2, 2] = torch.exp(torch.tensor(1j * alpha31 / 2, device=DEVICE, dtype=torch.complex128))

    assert_close(pmns._majorana_phase_matrix(), expected, name="P_majorana entries")


def test_zero_phases_reduce_to_identity_factor():
    pmns = make_majorana(0.0, 0.0)
    identity = torch.eye(3, device=DEVICE, dtype=torch.complex128)
    assert_close(pmns._majorana_phase_matrix(), identity, name="P_majorana == I at alpha=0")


def test_outer_block_identical_to_pmns_sm():
    """The Majorana phase lives entirely on the mass-eigenstate side of
    U = O @ U_red; the outer block O = R23 @ Delta must be untouched."""
    dirac = make_dirac()
    majorana = make_majorana(0.7, 2.3)
    assert_close(majorana.outer_block(), dirac.outer_block(), name="outer_block unaffected by Majorana phases")


def test_majorana_matrices_equal_dirac_times_phase_factor():
    dirac = make_dirac()
    majorana = make_majorana(0.7, 2.3)
    P = majorana._majorana_phase_matrix()

    assert_close(majorana.pmns_matrix(), dirac.pmns_matrix() @ P, name="U == U_Dirac @ P")
    assert_close(majorana.reduced(), dirac.reduced() @ P, name="U_red == U_red_Dirac @ P")


def test_majorana_pmns_matrices_stay_unitary():
    pmns = make_majorana(0.7, 2.3)
    identity = torch.eye(3, device=DEVICE, dtype=torch.complex128)

    U = pmns.pmns_matrix()
    Ured = pmns.reduced()

    assert_close(U.conj().transpose(-2, -1) @ U, identity, name="Majorana U unitary")
    assert_close(Ured.conj().transpose(-2, -1) @ Ured, identity, name="Majorana U_red unitary")


# ---------------------------------------------------------------------------
# The physical claim: oscillation probabilities are invariant
# ---------------------------------------------------------------------------

def _vacuum_probability(pmns, k_vector: torch.Tensor, alpha: int, beta: int) -> torch.Tensor:
    """P(nu_alpha -> nu_beta) = |sum_i U_{beta i} U*_{alpha i} exp(-i k_i)|^2.

    The generic vacuum oscillation formula every propagation method (solar,
    atmospheric, Earth, analytic, or numerical) reduces to. Used here to
    verify Majorana-phase invariance once, at the level all of them share,
    rather than duplicating the check across every pipeline.
    """
    U = pmns.pmns_matrix()
    phase = torch.exp(-1j * k_vector.to(dtype=U.dtype))
    amplitude = (U[..., beta, :] * U[..., alpha, :].conj() * phase).sum(dim=-1)
    return amplitude.abs() ** 2


def test_vacuum_oscillation_probability_is_majorana_invariant():
    dirac = make_dirac()
    majorana = make_majorana(1.1, 4.8)
    k_vector = torch.tensor([0.3, 1.7, 4.2], device=DEVICE, dtype=DTYPE)

    for alpha in range(3):
        for beta in range(3):
            p_dirac = _vacuum_probability(dirac, k_vector, alpha, beta)
            p_majorana = _vacuum_probability(majorana, k_vector, alpha, beta)
            assert_close(
                p_majorana, p_dirac,
                name=f"P(nu_{alpha} -> nu_{beta}) Majorana-phase invariance",
            )


# ---------------------------------------------------------------------------
# OscillationParameters.BSM_extension_majorana and config wiring
# ---------------------------------------------------------------------------

def test_oscillation_parameters_bsm_extension_majorana_flag():
    dirac_osc = PropagationConfig.oscillation_parameters_from_preset("_SM_NUFIT52_NO")
    assert dirac_osc.BSM_extension_majorana is False
    assert dirac_osc.BSM_extension is False
    assert isinstance(dirac_osc.pmns, PMNS_SM)

    majorana_osc = PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO", neutrino_nature="majorana", alpha21_deg=45.0, alpha31_deg=90.0,
    )
    assert majorana_osc.BSM_extension_majorana is True
    assert majorana_osc.BSM_extension is True
    assert isinstance(majorana_osc.pmns, PMNS_Majorana)


def test_majorana_with_sterile_preset_raises():
    with pytest.raises(ValueError, match="not yet supported"):
        PropagationConfig.oscillation_parameters_from_preset(
            "sterile_3p1_null_mixing", neutrino_nature="majorana",
        )


# ---------------------------------------------------------------------------
# effective_majorana_mass
# ---------------------------------------------------------------------------

def test_effective_majorana_mass_matches_direct_formula():
    pmns = make_majorana(1.1, 4.8)
    U = pmns.pmns_matrix()

    m1, m2, m3 = 0.01, 0.012, 0.051  # eV
    mass_sq = torch.tensor([m1**2, m2**2, m3**2], device=DEVICE, dtype=DTYPE)

    expected = (U[0, 0] ** 2 * m1 + U[0, 1] ** 2 * m2 + U[0, 2] ** 2 * m3).abs()

    result = effective_majorana_mass(U, mass_sq)
    assert_close(result, expected, name="m_bb direct formula")


def test_effective_majorana_mass_zero_angles_reduces_to_lightest_mass():
    zero = make_majorana(0.0, 0.0, theta12=0.0, theta13=0.0, theta23=0.0, delta=0.0)
    U = zero.pmns_matrix()  # identity: U_e1 = 1, U_e2 = U_e3 = 0

    m1 = 0.02
    mass_sq = torch.tensor([m1**2, 0.05**2, 0.1**2], device=DEVICE, dtype=DTYPE)

    result = effective_majorana_mass(U, mass_sq)
    assert_close(result, torch.tensor(m1, device=DEVICE, dtype=DTYPE), name="m_bb == m_1 at U = I")


def test_effective_majorana_mass_differs_between_dirac_and_majorana():
    """Sanity check that m_bb genuinely depends on the Majorana phases, in
    contrast to test_vacuum_oscillation_probability_is_majorana_invariant.
    """
    theta12, theta13 = 0.59, 0.15
    dirac = make_dirac(theta12=theta12, theta13=theta13, delta=0.0)
    majorana = make_majorana(math.pi, 0.0, theta12=theta12, theta13=theta13, delta=0.0)

    m1, m2, m3 = 0.01, 0.012, 0.051
    mass_sq = torch.tensor([m1**2, m2**2, m3**2], device=DEVICE, dtype=DTYPE)

    m_bb_dirac = effective_majorana_mass(dirac.pmns_matrix(), mass_sq)
    m_bb_majorana = effective_majorana_mass(majorana.pmns_matrix(), mass_sq)

    assert (m_bb_dirac - m_bb_majorana).abs().item() > 1.0e-6
