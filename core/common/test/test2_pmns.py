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
Pytest-compatible mathematical and physical checks for the SM PMNS matrix.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.core.SM.sm_pmns import PMNS_SM
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATOL = 1.0e-12
RTOL = 1.0e-12


def make_pmns(
    theta12: float | torch.Tensor = 0.59,
    theta13: float | torch.Tensor = 0.15,
    theta23: float | torch.Tensor = 0.78,
    delta: float | torch.Tensor = 1.20,
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device = DEVICE,
) -> PMNS_SM:
    context = RuntimeContext.resolve(device, dtype)
    return PMNS_SM(
        PMNSParams(
            theta12=theta12,
            theta13=theta13,
            theta23=theta23,
            delta=delta,
            context=context,
        )
    )


def eye_like(matrix: torch.Tensor) -> torch.Tensor:
    return torch.eye(
        matrix.shape[-1],
        device=matrix.device,
        dtype=matrix.dtype,
    ).expand(matrix.shape)


def test_pmns_scalar_shapes_and_dtypes():
    pmns = make_pmns()

    assert pmns.R12().shape == (3, 3)
    assert pmns.R13().shape == (3, 3)
    assert pmns.R23().shape == (3, 3)
    assert pmns.Delta().shape == (3, 3)
    assert pmns.pmns_matrix().shape == (3, 3)
    assert pmns.reduced().shape == (3, 3)
    assert pmns.params.theta12.dtype == torch.float64
    assert pmns.pmns_matrix().dtype == torch.complex128
    assert pmns.reduced().dtype == torch.complex128


def test_rotation_matrix_entries_and_delta_definition():
    theta12, theta13, theta23, delta = 0.59, 0.15, 0.78, 1.20
    pmns = make_pmns(theta12, theta13, theta23, delta)
    cdtype = torch.complex128

    def expected_rotation(i: int, j: int, theta: float) -> torch.Tensor:
        out = torch.eye(3, device=DEVICE, dtype=cdtype)
        c = math.cos(theta)
        s = math.sin(theta)
        out[i, i] = c
        out[j, j] = c
        out[i, j] = s
        out[j, i] = -s
        return out

    expected_delta = torch.eye(3, device=DEVICE, dtype=cdtype)
    expected_delta[2, 2] = torch.exp(torch.tensor(1j * delta, device=DEVICE, dtype=cdtype))

    assert_close(pmns.R12(), expected_rotation(0, 1, theta12), name="R12 entries")
    assert_close(pmns.R13(), expected_rotation(0, 2, theta13), name="R13 entries")
    assert_close(pmns.R23(), expected_rotation(1, 2, theta23), name="R23 entries")
    assert_close(pmns.Delta(), expected_delta, name="Delta = diag(1, 1, exp(i delta))")


def test_reduced_and_full_pmns_definitions():
    pmns = make_pmns()

    expected_reduced = pmns.R13() @ pmns.R12()
    expected_full = (
        pmns.R23()
        @ pmns.Delta()
        @ pmns.R13()
        @ pmns.Delta().conj()
        @ pmns.R12()
    )

    assert_close(pmns.reduced(), expected_reduced, name="Ured = R13 @ R12")
    assert_close(pmns.pmns_matrix(), expected_full, name="U = R23 Delta R13 Delta* R12")
    assert_close(pmns.U, pmns.reduced(), name="cached reduced matrix")
    assert_close(pmns.pmns, pmns.pmns_matrix(), name="cached full PMNS matrix")


def _assert_matrix_is_unitary(matrix: torch.Tensor, name: str) -> None:
    identity = eye_like(matrix)
    assert_close(matrix.conj().transpose(-2, -1) @ matrix, identity, name=f"{name} Udag U")
    assert_close(matrix @ matrix.conj().transpose(-2, -1), identity, name=f"{name} U Udag")


def test_pmns_matrix_is_unitary():
    pmns = make_pmns()
    _assert_matrix_is_unitary(pmns.pmns_matrix(), "pmns_matrix")


def test_reduced_matrix_is_unitary():
    pmns = make_pmns()
    _assert_matrix_is_unitary(pmns.reduced(), "reduced")


def test_determinants_are_unit_modulus_and_unity():
    pmns = make_pmns()

    assert_close(torch.linalg.det(pmns.pmns_matrix()), torch.ones((), device=DEVICE, dtype=torch.complex128))
    assert_close(torch.linalg.det(pmns.reduced()), torch.ones((), device=DEVICE, dtype=torch.complex128))


def test_dagger_conjugate_transpose_helpers():
    pmns = make_pmns()
    U = pmns.pmns_matrix()
    Ured = pmns.reduced()

    assert_close(pmns.dagger(), U.conj().transpose(-2, -1), name="dagger")
    assert_close(pmns.conjugate(), U.conj(), name="conjugate")
    assert_close(pmns.transpose(), U.transpose(-2, -1), name="transpose")
    assert_close(pmns.reduced_dagger(), Ured.conj().transpose(-2, -1), name="reduced_dagger")
    assert_close(pmns.reduced_conjugate(), Ured.conj(), name="reduced_conjugate")
    assert_close(pmns.reduced_transpose(), Ured.transpose(-2, -1), name="reduced_transpose")


def test_batch_shapes_and_unitarity():
    pmns = make_pmns(
        theta12=torch.tensor([0.58, 0.59, 0.60], device=DEVICE, dtype=DTYPE),
        theta13=torch.tensor([0.14, 0.15, 0.16], device=DEVICE, dtype=DTYPE),
        theta23=torch.tensor([0.77, 0.78, 0.79], device=DEVICE, dtype=DTYPE),
        delta=torch.tensor([0.0, 1.2, math.pi], device=DEVICE, dtype=DTYPE),
    )

    U = pmns.pmns_matrix()
    Ured = pmns.reduced()

    assert U.shape == (3, 3, 3)
    assert Ured.shape == (3, 3, 3)
    assert_close(U.conj().transpose(-2, -1) @ U, eye_like(U), name="batch PMNS unitarity")
    assert_close(Ured.conj().transpose(-2, -1) @ Ured, eye_like(Ured), name="batch reduced unitarity")


def test_antinu_conjugation_scalar_and_batch_mask():
    pmns = make_pmns(
        theta12=torch.tensor([0.58, 0.59, 0.60], device=DEVICE, dtype=DTYPE),
        theta13=torch.tensor([0.14, 0.15, 0.16], device=DEVICE, dtype=DTYPE),
        theta23=torch.tensor([0.77, 0.78, 0.79], device=DEVICE, dtype=DTYPE),
        delta=torch.tensor([0.3, 1.2, 2.4], device=DEVICE, dtype=DTYPE),
    )

    U = pmns.pmns_matrix()
    Ured = pmns.reduced()
    assert_close(pmns.pmns_matrix(antinu=True), U.conj(), name="PMNS antinu scalar")
    assert_close(pmns.reduced(antinu=True), Ured.conj(), name="reduced antinu scalar")

    mask = torch.tensor([False, True, False], device=DEVICE)
    expected = torch.where(mask[..., None, None], U.conj(), U)
    expected_reduced = torch.where(mask[..., None, None], Ured.conj(), Ured)
    assert_close(pmns.pmns_matrix(antinu=mask), expected, name="PMNS antinu batch")
    assert_close(pmns.reduced(antinu=mask), expected_reduced, name="reduced antinu batch")


def test_zero_angles_identity_and_two_flavour_limit():
    zero = make_pmns(theta12=0.0, theta13=0.0, theta23=0.0, delta=0.0)
    identity = torch.eye(3, device=DEVICE, dtype=torch.complex128)

    assert_close(zero.R12(), identity, name="R12 identity")
    assert_close(zero.R13(), identity, name="R13 identity")
    assert_close(zero.R23(), identity, name="R23 identity")
    assert_close(zero.Delta(), identity, name="Delta identity")
    assert_close(zero.reduced(), identity, name="reduced identity")
    assert_close(zero.pmns_matrix(), identity, name="PMNS identity")

    theta12 = 0.59
    two_flavour = make_pmns(theta12=theta12, theta13=0.0, theta23=0.0, delta=1.7)
    assert_close(two_flavour.pmns_matrix(), two_flavour.R12(), name="two-flavour R12 limit")


def test_cp_conserving_limits_are_real_and_jarlskog_zero():
    for delta in (0.0, math.pi):
        pmns = make_pmns(delta=delta)
        assert_close(pmns.pmns_matrix().imag, torch.zeros((3, 3), device=DEVICE, dtype=DTYPE))
        assert_close(pmns.jarlskog_invariant(), torch.zeros((), device=DEVICE, dtype=DTYPE))

    no_theta13 = make_pmns(theta13=0.0, delta=1.2)
    assert_close(no_theta13.jarlskog_invariant(), torch.zeros((), device=DEVICE, dtype=DTYPE))


def test_jarlskog_matches_analytic_formula_and_changes_sign_with_delta():
    theta12, theta13, theta23, delta = 0.59, 0.15, 0.78, 1.20
    pmns = make_pmns(theta12, theta13, theta23, delta)
    pmns_minus = make_pmns(theta12, theta13, theta23, -delta)

    expected = (
        0.125
        * math.sin(2.0 * theta12)
        * math.sin(2.0 * theta13)
        * math.sin(2.0 * theta23)
        * math.cos(theta13)
        * math.sin(delta)
    )

    assert_close(pmns.jarlskog_invariant(), torch.tensor(expected, device=DEVICE, dtype=DTYPE))
    assert_close(pmns_minus.jarlskog_invariant(), -pmns.jarlskog_invariant())


def test_physical_electron_row_elements_match_angles():
    theta12, theta13 = 0.59, 0.15
    pmns = make_pmns(theta12=theta12, theta13=theta13)
    projector = pmns.vacuum_flavour_projector()

    expected_electron_row = torch.tensor(
        [
            math.cos(theta13) ** 2 * math.cos(theta12) ** 2,
            math.cos(theta13) ** 2 * math.sin(theta12) ** 2,
            math.sin(theta13) ** 2,
        ],
        device=DEVICE,
        dtype=DTYPE,
    )

    assert_close(projector[0], expected_electron_row, name="electron row |U_ei|^2")


def test_vacuum_flavour_projector_is_doubly_stochastic_and_cp_even():
    pmns = make_pmns()
    P = pmns.vacuum_flavour_projector()
    P_anti = pmns.vacuum_flavour_projector(antinu=True)

    assert_close(P, pmns.pmns_matrix().abs() ** 2, name="projector = |U|^2")
    assert_close(P, P_anti, name="vacuum projector is CP-even")
    assert torch.all(P >= -ATOL)
    assert torch.all(P <= 1.0 + ATOL)
    assert_close(P.sum(dim=0), torch.ones(3, device=DEVICE, dtype=DTYPE), name="mass-column sums")
    assert_close(P.sum(dim=1), torch.ones(3, device=DEVICE, dtype=DTYPE), name="flavour-row sums")


def test_outer_block_matches_r23_delta_and_drives_flavour_basis():
    """``outer_block`` (promoted from a private per-module helper, see
    ``tpeanuts.core.common.hamiltonian.hamiltonian_reduced`` and
    ``PMNS_sterile.flavour_basis``) is O = R23 . Delta for the
    3-flavour SM, and is exactly what ``flavour_basis`` uses
    internally (O @ op @ O^dagger).
    """
    pmns = make_pmns()
    operator = torch.diag(
        torch.tensor([0.2, 1.0, 3.0], device=DEVICE, dtype=torch.complex128)
    )

    O = pmns.outer_block()
    expected_O = pmns.R23() @ pmns.Delta()
    assert_close(O, expected_O, name="outer_block == R23 @ Delta")

    expected_flavour = O @ operator @ O.conj().transpose(-2, -1)
    assert_close(
        pmns.flavour_basis(operator), expected_flavour,
        name="flavour_basis via outer_block",
    )

    O_antinu = pmns.outer_block(antinu=True)
    assert_close(
        O_antinu, pmns.R23().conj() @ pmns.Delta().conj(),
        name="outer_block antinu conjugates R23 and Delta",
    )


def test_flavour_basis_and_reduced_basis_transformations():
    pmns = make_pmns()
    identity = torch.eye(3, device=DEVICE, dtype=torch.complex128)
    operator = torch.diag(
        torch.tensor([0.2, 1.0, 3.0], device=DEVICE, dtype=torch.complex128)
    )

    expected_flavour = (
        pmns.R23()
        @ pmns.Delta()
        @ operator
        @ pmns.Delta().conj().transpose(-2, -1)
        @ pmns.R23().transpose(-2, -1)
    )

    assert_close(pmns.flavour_basis(identity), identity, name="identity basis transform")
    assert_close(pmns.flavour_basis(operator), expected_flavour, name="flavour_basis")

    operator_flavour = pmns.flavour_basis(operator)
    assert_close(
        pmns.reduced_basis(operator_flavour),
        operator,
        name="reduced_basis inverse transform",
    )

    with pytest.raises(ValueError, match="O_flavour_basis must have final dimensions"):
        pmns.reduced_basis(torch.ones(2, 2, device=DEVICE, dtype=torch.complex128))
