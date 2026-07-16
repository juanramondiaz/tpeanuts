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
Pytest-compatible checks specific to the 3+1 sterile-neutrino extension:
``tpeanuts.core.BSM.PMNS_sterile.PMNS_sterile`` and its integration with the
BSM Hamiltonian builders and numerical evolutor.

Generic BSM Hamiltonian-builder machinery is covered in
``test1_bsm_hamiltonian.py``; NSI-specific checks live in
``test2_bsm_nsi.py``.

The vacuum-probability SM-limit test below
(``test_vacuum_probability_sm_limit_exact_at_zero_sterile_angles``) is a
regression guard for a discrepancy observed in the exploratory notebook
``notebooks/validation/physics/BSM/sterile1_test.ipynb``: PMNS- and
Hamiltonian-level SM-limit checks there are exact (0.00e+00), but the
notebook's own probability-level SM-limit check printed large residuals
(0.07-0.98) while its concluding text claimed they were <=1e-6. This test
recomputes the same comparison directly against ``evolutor_numerical`` with
tight tolerance so that a real regression in the production code (as opposed
to a bug in the notebook's local helper) would be caught automatically.
"""

from __future__ import annotations

import math

import pytest
import torch

from tpeanuts.core.BSM.hamiltonian import hamiltonian_flavour_bsm, hamiltonian_reduced_bsm
from tpeanuts.core.BSM.PMNS_sterile import PMNSSterileParams, PMNS_sterile
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_flavour as hamiltonian_flavour_sm,
    hamiltonian_reduced as hamiltonian_reduced_sm,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.common.presets import OSCILLATION_PRESETS
from tpeanuts.core.numerical.evolutor import evolutor_numerical
from tpeanuts.core.SM.pmns import PMNS_SM
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128

STERILE_PRESET_NAMES = [
    name for name, preset in OSCILLATION_PRESETS.items() if "theta14_deg" in preset
]


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_sm_params(context: RuntimeContext) -> PMNSParams:
    return PMNSParams(theta12=0.5836, theta13=0.1498, theta23=0.8552, delta=3.438, context=context)


def make_sterile_pmns(
    theta14=0.0, theta24=0.0, theta34=0.0,
    delta14=0.0, delta24=0.0, delta34=0.0,
    *, context: RuntimeContext | None = None,
) -> PMNS_sterile:
    ctx = context or make_context()
    sterile_params = PMNSSterileParams(
        theta14=theta14, theta24=theta24, theta34=theta34,
        delta14=delta14, delta24=delta24, delta34=delta34,
        context=ctx,
    )
    return PMNS_sterile(make_sm_params(ctx), sterile_params)


def make_matching_sm_pmns(context: RuntimeContext) -> PMNS_SM:
    return PMNS_SM(make_sm_params(context))


def embed_3x3(U3: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((4, 4), device=U3.device, dtype=U3.dtype)
    out[:3, :3] = U3
    out[3, 3] = 1.0
    return out


def eye_like(matrix: torch.Tensor) -> torch.Tensor:
    return torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).expand(matrix.shape)


def assert_unitary(U: torch.Tensor, *, name: str, atol=1.0e-10) -> None:
    identity = eye_like(U)
    assert_close(U.conj().transpose(-2, -1) @ U, identity, atol=atol, rtol=atol, name=f"{name} (U^dagger U)")
    assert_close(U @ U.conj().transpose(-2, -1), identity, atol=atol, rtol=atol, name=f"{name} (U U^dagger)")


# ---------------------------------------------------------------------------
# Parameter container and rotation builders
# ---------------------------------------------------------------------------

def test_sterile_params_convert_to_tensors_on_context_device_dtype():
    ctx = make_context()
    params = PMNSSterileParams(
        theta14=0.1, theta24=0.2, theta34=0.0, delta14=0.3, delta24=0.4, delta34=0.0, context=ctx,
    )
    for name in ("theta14", "theta24", "theta34", "delta14", "delta24", "delta34"):
        value = getattr(params, name)
        assert torch.is_tensor(value)
        assert value.dtype == DTYPE
        assert value.device.type == DEVICE.type


def test_R14_R24_R34_entries_match_rotation_formula():
    ctx = make_context()
    theta14, theta24, theta34 = 0.12, 0.08, 0.05
    delta14, delta24 = 0.7, -1.1
    pmns4 = make_sterile_pmns(theta14, theta24, theta34, delta14, delta24, 0.0, context=ctx)

    def expected_rotation(i: int, j: int, theta: float, phase: float) -> torch.Tensor:
        out = torch.eye(4, device=DEVICE, dtype=CDTYPE)
        c, s = math.cos(theta), math.sin(theta)
        out[i, i] = c
        out[j, j] = c
        out[i, j] = s * torch.exp(torch.tensor(-1j * phase, device=DEVICE, dtype=CDTYPE))
        out[j, i] = -s * torch.exp(torch.tensor(1j * phase, device=DEVICE, dtype=CDTYPE))
        return out

    assert_close(pmns4.R14(), expected_rotation(0, 3, theta14, delta14), name="R14 entries")
    assert_close(pmns4.R24(), expected_rotation(1, 3, theta24, delta24), name="R24 entries")
    assert_close(pmns4.R34(), expected_rotation(2, 3, theta34, 0.0), name="R34 entries (real, delta34=0)")


def test_zero_sterile_angles_give_identity_active_sterile_rotations():
    pmns4 = make_sterile_pmns()
    identity = torch.eye(4, device=DEVICE, dtype=CDTYPE)
    assert_close(pmns4.R14(), identity, name="R14 identity")
    assert_close(pmns4.R24(), identity, name="R24 identity")
    assert_close(pmns4.R34(), identity, name="R34 identity")


# ---------------------------------------------------------------------------
# SM limit at the PMNS level (theta14 = theta24 = theta34 = 0)
# ---------------------------------------------------------------------------

def test_zero_sterile_angles_reduced_matrix_embeds_sm_reduced_matrix():
    ctx = make_context()
    pmns4 = make_sterile_pmns(context=ctx)
    pmns3 = make_matching_sm_pmns(ctx)

    assert_close(pmns4.reduced(), embed_3x3(pmns3.reduced()), atol=1.0e-14, rtol=1.0e-14, name="Ured_4 -> embed(Ured_SM)")


def test_zero_sterile_angles_full_pmns_matrix_embeds_sm_pmns_matrix():
    ctx = make_context()
    pmns4 = make_sterile_pmns(context=ctx)
    pmns3 = make_matching_sm_pmns(ctx)

    assert_close(pmns4.pmns_matrix(), embed_3x3(pmns3.pmns_matrix()), atol=1.0e-14, rtol=1.0e-14, name="U_4 -> embed(U_SM)")


# ---------------------------------------------------------------------------
# Unitarity and antineutrino conjugation
# ---------------------------------------------------------------------------

def test_pmns_and_reduced_matrices_are_unitary_for_nonzero_mixing():
    ctx = make_context()
    pmns4 = make_sterile_pmns(theta14=0.148, theta24=0.131, theta34=0.0, delta14=0.0, delta24=0.0, context=ctx)

    assert_unitary(pmns4.pmns_matrix(), name="U_4")
    assert_unitary(pmns4.reduced(), name="Ured_4")


def test_pmns_matrices_are_unitary_for_batched_angles():
    ctx = make_context()
    theta14 = torch.tensor([0.05, 0.10, 0.15], device=DEVICE, dtype=DTYPE)
    theta24 = torch.tensor([0.03, 0.06, 0.09], device=DEVICE, dtype=DTYPE)
    pmns4 = make_sterile_pmns(theta14=theta14, theta24=theta24, context=ctx)

    U = pmns4.pmns_matrix()
    assert U.shape == (3, 4, 4)
    assert_unitary(U, name="batched U_4")


def test_antinu_conjugation_scalar_and_batch_mask():
    ctx = make_context()
    theta14 = torch.tensor([0.05, 0.10, 0.15], device=DEVICE, dtype=DTYPE)
    theta24 = torch.tensor([0.03, 0.06, 0.09], device=DEVICE, dtype=DTYPE)
    pmns4 = make_sterile_pmns(theta14=theta14, theta24=theta24, context=ctx)

    U = pmns4.pmns_matrix()
    assert_close(pmns4.pmns_matrix(antinu=True), U.conj(), name="antinu scalar")

    mask = torch.tensor([False, True, False], device=DEVICE)
    expected = torch.where(mask[..., None, None], U.conj(), U)
    assert_close(pmns4.pmns_matrix(antinu=mask), expected, name="antinu batch mask")


# ---------------------------------------------------------------------------
# operator_flavour_basis structural properties
# ---------------------------------------------------------------------------

def test_operator_flavour_basis_matches_outer_block_conjugation_formula():
    ctx = make_context()
    pmns4 = make_sterile_pmns(theta14=0.15, theta24=0.10, theta34=0.05, delta14=0.3, delta24=-0.2, context=ctx)
    op = torch.diag(torch.tensor([0.2, 1.0, 3.0, 0.7], device=DEVICE, dtype=CDTYPE))

    O4 = pmns4.R23() @ pmns4.Delta() @ pmns4.R24() @ pmns4.R34()
    expected = O4 @ op @ O4.conj().transpose(-2, -1)

    assert_close(pmns4.operator_flavour_basis(op), expected, name="operator_flavour_basis formula")


def test_operator_flavour_basis_preserves_hermiticity():
    ctx = make_context()
    pmns4 = make_sterile_pmns(theta14=0.15, theta24=0.10, theta34=0.05, delta14=0.3, delta24=-0.2, context=ctx)

    raw = torch.randn(4, 4, dtype=CDTYPE)
    hermitian_op = raw + raw.conj().transpose(-2, -1)

    out = pmns4.operator_flavour_basis(hermitian_op)
    assert_close(out, out.conj().transpose(-2, -1), atol=1.0e-10, rtol=1.0e-10, name="flavour-basis operator stays Hermitian")


# ---------------------------------------------------------------------------
# SM limit at the Hamiltonian level
# ---------------------------------------------------------------------------

def test_hamiltonian_sm_limit_exact_for_null_mixing_preset():
    """At zero sterile mixing, the active 3x3 block must reproduce the pure
    SM Hamiltonian exactly and decouple exactly from the sterile row/column.

    The (3, 3) corner itself is NOT expected to vanish: it carries the
    sterile kinetic eigenvalue coming from DeltamSq41, which is a real,
    physical (if unmixed) energy -- only the mixing matrix embeds as the
    identity at zero angles (see the PMNS-level SM-limit tests above), not
    the Hamiltonian, since the sterile state keeps its own mass even when
    fully decoupled from the active sector.
    """
    ctx = make_context()
    osc_sm = OscillationParameters.from_preset("_SM_NUFIT52_NO", context=ctx)
    osc_sterile = OscillationParameters.from_preset("sterile_3p1_null_mixing", context=ctx)

    for E, n_e in [(500.0, 0.5), (1000.0, 1.5), (5000.0, 2.5)]:
        E_t = torch.tensor(E, device=DEVICE, dtype=DTYPE)
        n_e_t = torch.tensor(n_e, device=DEVICE, dtype=DTYPE)

        H_reduced_sm = hamiltonian_reduced_sm(osc_sm, E_t, n_e_t, context=ctx)
        H_reduced_bsm = hamiltonian_reduced_bsm(osc_sterile, E_t, n_e_t, context=ctx, epsilon=None)
        assert_close(
            H_reduced_bsm[:3, :3], H_reduced_sm, atol=1.0e-12, rtol=1.0e-12,
            name=f"H_reduced active block SM-limit at E={E}, n_e={n_e}",
        )
        assert_close(
            H_reduced_bsm[:3, 3], torch.zeros(3, device=DEVICE, dtype=CDTYPE),
            atol=1.0e-12, rtol=1.0e-12, name=f"H_reduced active-sterile decoupling at E={E}, n_e={n_e}",
        )

        H_flavour_sm = hamiltonian_flavour_sm(osc_sm, E_t, n_e_t, context=ctx)
        H_flavour_bsm = hamiltonian_flavour_bsm(osc_sterile, E_t, n_e_t, context=ctx, epsilon=None)
        assert_close(
            H_flavour_bsm[:3, :3], H_flavour_sm, atol=1.0e-12, rtol=1.0e-12,
            name=f"H_flavour active block SM-limit at E={E}, n_e={n_e}",
        )
        assert_close(
            H_flavour_bsm[:3, 3], torch.zeros(3, device=DEVICE, dtype=CDTYPE),
            atol=1.0e-12, rtol=1.0e-12, name=f"H_flavour active-sterile decoupling at E={E}, n_e={n_e}",
        )


# ---------------------------------------------------------------------------
# SM limit at the probability level (regression guard, see module docstring)
# ---------------------------------------------------------------------------

def test_vacuum_probability_sm_limit_exact_at_zero_sterile_angles():
    ctx = make_context()
    osc_sm = OscillationParameters.from_preset("_SM_NUFIT52_NO", context=ctx)
    osc_sterile = OscillationParameters.from_preset("sterile_3p1_null_mixing", context=ctx)

    E = torch.tensor(10.0, device=DEVICE, dtype=DTYPE)
    n_e_vacuum = torch.zeros(1, device=DEVICE, dtype=DTYPE)

    for dx_value in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0):
        dx = torch.tensor([dx_value], device=DEVICE, dtype=DTYPE)

        S_sm = evolutor_numerical(osc_sm, E, n_e_vacuum, dx, device=DEVICE, dtype=DTYPE)
        S_sterile = evolutor_numerical(osc_sterile, E, n_e_vacuum, dx, device=DEVICE, dtype=DTYPE)

        P_sm = S_sm.abs() ** 2
        P_sterile_active = S_sterile[:3, :3].abs() ** 2

        assert_close(
            P_sterile_active, P_sm, atol=1.0e-9, rtol=1.0e-9,
            name=f"vacuum probability SM-limit at dx={dx_value}",
        )


def test_probability_conservation_including_sterile_channel():
    ctx = make_context()
    osc = OscillationParameters.from_preset("sterile_3p1_bestfit_giunti2017", context=ctx)
    n_e = torch.tensor([1.0, 1.2, 1.4], device=DEVICE, dtype=DTYPE)
    dx = torch.tensor([0.02, 0.03, 0.04], device=DEVICE, dtype=DTYPE)

    S = evolutor_numerical(osc, 1000.0, n_e, dx, device=DEVICE, dtype=DTYPE)

    assert_unitary(S, name="4-flavour evolutor unitarity")
    P = S.abs() ** 2
    assert_close(
        P.sum(dim=-1), torch.ones(4, device=DEVICE, dtype=DTYPE), atol=1.0e-10, rtol=1.0e-10,
        name="row probability sums to 1 including sterile channel",
    )


# ---------------------------------------------------------------------------
# Sterile decoupling limit
# ---------------------------------------------------------------------------

def test_active_block_deviation_from_sm_shrinks_monotonically_with_theta14():
    ctx = make_context()
    pmns3 = make_matching_sm_pmns(ctx)
    Ured_sm = pmns3.reduced()

    angles = [0.10, 0.03, 0.01, 0.003]
    deviations = []
    for theta14 in angles:
        pmns4 = make_sterile_pmns(theta14=theta14, context=ctx)
        active_block = pmns4.reduced()[:3, :3]
        deviations.append(torch.max(torch.abs(active_block - Ured_sm)).item())

    for earlier, later in zip(deviations, deviations[1:]):
        assert later < earlier, f"deviation did not shrink monotonically: {deviations}"
    assert deviations[-1] < 1.0e-2


# ---------------------------------------------------------------------------
# Registered sterile oscillation presets
# ---------------------------------------------------------------------------

def test_sterile_presets_always_carry_zero_delta34():
    for name in STERILE_PRESET_NAMES:
        assert "delta34_deg" not in OSCILLATION_PRESETS[name], (
            f"preset {name!r} should not carry an explicit delta34_deg "
            "(R34 is always real for Dirac 3+1 neutrinos; from_preset forces delta34=0)"
        )


@pytest.mark.parametrize("name", STERILE_PRESET_NAMES)
def test_all_registered_sterile_presets_build_valid_unitary_hermitian_physics(name):
    ctx = make_context()
    osc = OscillationParameters.from_preset(name, context=ctx)
    assert osc.pmns.n_flavours == 4
    assert osc.DeltamSq41 is not None

    assert_unitary(osc.pmns.pmns_matrix(), name=f"U_4 [{name}]")
    assert_unitary(osc.pmns.reduced(), name=f"Ured_4 [{name}]")

    H = hamiltonian_flavour_bsm(
        osc, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), torch.tensor(1.5, device=DEVICE, dtype=DTYPE),
        context=ctx, epsilon=None,
    )
    assert_close(H, H.conj().transpose(-2, -1), atol=1.0e-10, rtol=1.0e-10, name=f"H Hermitian [{name}]")
    eigvals = torch.linalg.eigvalsh(H)
    assert torch.isfinite(eigvals).all(), f"non-finite eigenvalues for preset {name!r}"
