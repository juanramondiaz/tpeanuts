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
Pytest-compatible checks for the Hamiltonian builders in
``tpeanuts.core.common.hamiltonian``.

There is a single implementation of each builder (kinetic/matter/reduced/
flavour), correct for the 3-flavour Standard Model, NSI, the 3+1 sterile
extension, or any combination, dispatching only on the ``oscillation``
object passed in. This file covers that machinery directly across all of
those scenarios; NSI-specific and sterile-specific physics checks live in
``tpeanuts.core.BSM.test.test2_bsm_nsi`` and
``tpeanuts.core.BSM.test.test3_bsm_sterile`` respectively.
"""

from __future__ import annotations

import pytest
import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.BSM.bsm_nsi import NSIConfig
from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_flavour,
    hamiltonian_kinetic_reduced,
    hamiltonian_matter_reduced,
    hamiltonian_reduced,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.common.potential import kinetic_potential, matter_potential_cc, matter_potential_nc
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close
from tpeanuts.util.type import as_tensor


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_sm_oscillation(
    *, antinu=False, NSI_extension: str | None = None, context: RuntimeContext | None = None,
) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO",
        antinu=antinu,
        NSI_extension=NSI_extension,
        context=context or make_context(),
    )


def make_sterile_oscillation(
    theta14=0.0,
    theta24=0.0,
    theta34=0.0,
    delta14=0.0,
    delta24=0.0,
    DeltamSq41=1.7,
    *,
    antinu=False,
    NSI_extension: str | None = None,
    context: RuntimeContext | None = None,
) -> OscillationParameters:
    ctx = context or make_context()
    sm_params = PMNSParams(
        theta12=0.5836, theta13=0.1498, theta23=0.8552, delta=3.438, context=ctx,
    )
    sterile_params = PMNSSterileParams(
        theta14=theta14, theta24=theta24, theta34=theta34,
        delta14=delta14, delta24=delta24, delta34=0.0,
        context=ctx,
    )
    pmns4 = PMNS_sterile(sm_params, sterile_params)
    nsi_obj = None
    if NSI_extension is not None:
        nsi_obj = NSIConfig.from_preset(NSI_extension, device=ctx.device, real_dtype=ctx.dtype)
    mass_spectrum = MassSpectrum_BSM(
        DeltamSq21=as_tensor(7.41e-5, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=as_tensor(2.511e-3, device=ctx.device, dtype=ctx.dtype),
        DeltamSq41=as_tensor(DeltamSq41, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns4, mass_spectrum=mass_spectrum, antinu=antinu, nsi=nsi_obj)


def assert_hermitian(H: torch.Tensor, name: str) -> None:
    assert_close(H, H.conj().transpose(-2, -1), atol=1.0e-10, rtol=1.0e-10, name=f"{name} is Hermitian")


# ---------------------------------------------------------------------------
# hamiltonian_kinetic_reduced
# ---------------------------------------------------------------------------

def test_hamiltonian_kinetic_reduced_formula_for_3_flavours():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Ured = osc.pmns.reduced()

    Hkin, ki = hamiltonian_kinetic_reduced(osc, E, Ured, return_ki=True)
    mass_sq = osc.mass_spectrum.difference_vector(context=ctx)
    ki_expected = kinetic_potential(mass_sq, E, context=ctx)
    expected = (Ured * ki_expected.to(dtype=Ured.dtype)) @ Ured.conj().transpose(-1, -2)

    assert Hkin.shape == (3, 3)
    assert ki.shape == (3,)
    assert Hkin.dtype == CDTYPE
    assert_close(ki, ki_expected, name="ki formula")
    assert_close(Hkin, expected, name="Hkin = Ured diag(ki) Ured^dagger")


def test_hamiltonian_kinetic_reduced_batched_energy():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)

    Hkin, ki = hamiltonian_kinetic_reduced(osc, energy, osc.pmns.reduced(), return_ki=True)

    assert Hkin.shape == (3, 3, 3)
    assert ki.shape == (3, 3)
    assert torch.isfinite(Hkin.real).all()
    assert torch.isfinite(Hkin.imag).all()


def test_hamiltonian_kinetic_reduced_invalid_shape_raises():
    osc = make_sm_oscillation()
    with pytest.raises(ValueError, match="Ured must have final dimensions"):
        hamiltonian_kinetic_reduced(osc, 1000.0, torch.eye(2, device=DEVICE, dtype=CDTYPE))


def test_hamiltonian_kinetic_reduced_uses_hermitian_conjugate_for_sterile():
    """Regression test for a real convention bug: ``Ured = R13 R12 R14`` is
    genuinely complex whenever the active-sterile CP phase delta14 != 0, and
    only the Hermitian conjugate matches the dressing convention used by
    ``PMNS_sterile.flavour_basis``.
    """
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, delta14=0.3, DeltamSq41=1.7, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Ured4 = osc.pmns.reduced()
    assert not torch.allclose(Ured4.imag, torch.zeros_like(Ured4.imag)), (
        "test setup requires a genuinely complex reduced mixing matrix"
    )

    Hkin = hamiltonian_kinetic_reduced(osc, E, Ured4, evolution_scale_m=constant.R_E)

    mass_sq = osc.mass_spectrum.difference_vector(context=ctx)
    ki_vals = kinetic_potential(mass_sq, E, context=ctx)
    expected = (Ured4 * ki_vals.to(dtype=Ured4.dtype)) @ Ured4.conj().transpose(-1, -2)

    assert_close(Hkin, expected, name="sterile kinetic hamiltonian uses the Hermitian conjugate")


def test_hamiltonian_kinetic_reduced_return_ki_supports_4_flavours():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Ured4 = osc.pmns.reduced()

    Hkin, ki = hamiltonian_kinetic_reduced(osc, E, Ured4, return_ki=True)

    assert Hkin.shape == (4, 4)
    assert ki.shape == (4,)
    trace_Hkin = torch.diagonal(Hkin, dim1=-2, dim2=-1).sum(dim=-1)
    assert_close(trace_Hkin.real, ki.sum(dim=-1), atol=1.0e-10, rtol=1.0e-10, name="trace(Hkin) == sum(ki), N=4")


# ---------------------------------------------------------------------------
# hamiltonian_matter_reduced
# ---------------------------------------------------------------------------

def test_hamiltonian_matter_reduced_formula_no_nsi():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    V = matter_potential_cc(n_e, antinu=osc.antinu, context=ctx)

    Hmat = hamiltonian_matter_reduced(osc, n_e, context=ctx)
    expected = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    expected[0, 0] = V.to(dtype=CDTYPE)

    assert Hmat.shape == (3, 3)
    assert_close(Hmat, expected, name="Hmat = diag(V, 0, 0)")


def test_hamiltonian_matter_reduced_batched():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    n_e = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)
    V = matter_potential_cc(n_e, antinu=osc.antinu, context=ctx)

    Hmat = hamiltonian_matter_reduced(osc, n_e, context=ctx)

    assert Hmat.shape == (3, 3, 3)
    assert_close(Hmat[:, 0, 0].real, V, name="batched matter diagonal")
    assert_close(Hmat[:, 1:, :], torch.zeros((3, 2, 3), device=DEVICE, dtype=CDTYPE))


def test_hamiltonian_matter_reduced_formula_with_offdiagonal_nsi_entries():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx, NSI_extension="nsi_globalfit_esteban2018")
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    V = matter_potential_cc(n_e, antinu=osc.antinu, context=ctx)
    eps = osc.nsi.epsilon

    Hmat = hamiltonian_matter_reduced(osc, n_e, context=ctx)
    Hmat_flavour = V.to(dtype=CDTYPE) * (
        torch.diag(torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)) + eps
    )
    O = osc.pmns.outer_block()
    expected = O.conj().transpose(-2, -1) @ Hmat_flavour @ O

    assert_close(Hmat, expected, name="O^dagger V (diag(1,0,0) + epsilon) O")


def test_hamiltonian_matter_reduced_sterile_no_nsi_is_diag_v():
    ctx = make_context()
    osc4 = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    V = matter_potential_cc(n_e, antinu=osc4.antinu, context=ctx)

    Hmat4 = hamiltonian_matter_reduced(osc4, n_e, context=ctx)
    expected = torch.zeros((4, 4), device=DEVICE, dtype=CDTYPE)
    expected[0, 0] = V.to(dtype=CDTYPE)

    assert Hmat4.shape == (4, 4)
    assert_close(Hmat4, expected, name="no-NSI sterile matter term is diag(V,0,0,0)")


def test_hamiltonian_matter_reduced_sterile_nsi_preserves_electron_entry():
    """O = R23.Delta.R24.R34 commutes with diag(V,0,...,0), i.e. it is block
    diagonal preserving the electron index exactly -- so the (0,0) entry of
    the reduced-basis matter Hamiltonian equals V*(1+eps_ee) regardless of
    the O_sub rotation mixing mu/tau/sterile (unlike the flavour-basis
    embedding, the reduced-basis sterile row/column are not zero in general
    once NSI is active, since O_sub genuinely mixes them with mu/tau).
    """
    ctx = make_context()
    cfg = NSIConfig.from_preset("nsi_globalfit_esteban2018", device=DEVICE, real_dtype=DTYPE)
    osc4 = make_sterile_oscillation(
        theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx, NSI_extension="nsi_globalfit_esteban2018",
    )
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    V = matter_potential_cc(n_e, antinu=osc4.antinu, context=ctx)

    Hmat4 = hamiltonian_matter_reduced(osc4, n_e, context=ctx)
    expected_ee = V.to(dtype=CDTYPE) * (1.0 + cfg.eps_ee)

    assert Hmat4.shape == (4, 4)
    assert_close(Hmat4[0, 0], expected_ee, name="electron entry invariant under O rotation")


# ---------------------------------------------------------------------------
# hamiltonian_reduced / hamiltonian_flavour
# ---------------------------------------------------------------------------

def test_hamiltonian_reduced_equals_kinetic_plus_matter():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    Hkin = hamiltonian_kinetic_reduced(osc, energy, osc.pmns.reduced(antinu=osc.antinu))
    Hmat = hamiltonian_matter_reduced(osc, n_e, context=ctx)

    assert H.shape == (3, 3)
    assert_close(H, Hkin + Hmat, name="H_reduced = Hkin + Hmat")


def test_hamiltonian_reduced_batched_energy_and_density():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    energy = torch.tensor([500.0, 1000.0, 5000.0], device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor([0.5, 1.0, 1.5], device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx)

    assert H.shape == (3, 3, 3)
    assert H.dtype == CDTYPE
    assert torch.isfinite(H.real).all()
    assert torch.isfinite(H.imag).all()


def test_hamiltonian_reduced_context_none_uses_pmns_device_dtype():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)

    H = hamiltonian_reduced(osc, 1000.0, 1.0)

    assert H.shape == (3, 3)
    assert H.device.type == ctx.device.type
    assert H.dtype == CDTYPE


def test_hamiltonian_flavour_matches_pmns_basis_transform():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    Hred = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    Hflav = hamiltonian_flavour(osc, energy, n_e, context=ctx)
    expected = osc.pmns.flavour_basis(Hred, antinu=osc.antinu, device=Hred.device, dtype=Hred.dtype)

    assert Hflav.shape == (3, 3)
    assert_close(Hflav, expected, name="flavour-basis Hamiltonian transform")


def test_hamiltonian_antinu_matches_manual_sign_and_conjugation():
    ctx = make_context()
    osc = make_sm_oscillation(antinu=True, context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    Hkin = hamiltonian_kinetic_reduced(osc, energy, osc.pmns.reduced(antinu=True))
    Hmat = hamiltonian_matter_reduced(osc, n_e, context=ctx)

    assert_close(H, Hkin + Hmat, name="antinu reduced Hamiltonian")


def test_hamiltonian_reduced_legacy_precision_changes_only_matter_term():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_full = hamiltonian_reduced(osc, energy, n_e, context=ctx, legacy_precision=False)
    H_legacy = hamiltonian_reduced(osc, energy, n_e, context=ctx, legacy_precision=True)
    Hmat_full = hamiltonian_matter_reduced(osc, n_e, context=ctx, legacy_precision=False)
    Hmat_legacy = hamiltonian_matter_reduced(osc, n_e, context=ctx, legacy_precision=True)
    expected_delta = Hmat_legacy - Hmat_full

    assert_close(H_legacy - H_full, expected_delta, name="legacy precision matter-only delta")


def test_reduced_with_nsi_matches_manual_kinetic_plus_matter_construction():
    """H_mat in the reduced basis is the flavour-basis NSI matter term
    ``V * (diag(1,0,0) + epsilon)`` rotated by ``O^dagger (.) O``, where
    ``O = R23 . Delta`` -- NOT the flavour-basis term applied verbatim.
    """
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx, NSI_extension="nsi_globalfit_esteban2018")
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    eps = osc.nsi.epsilon

    H = hamiltonian_reduced(osc, E, n_e, context=ctx)

    Hkin = hamiltonian_kinetic_reduced(osc, E, osc.pmns.reduced())
    V = matter_potential_cc(n_e, antinu=osc.antinu, context=ctx)
    Hmat_flavour = V.to(dtype=CDTYPE) * (
        torch.diag(torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)) + eps
    )
    O = osc.pmns.R23() @ osc.pmns.Delta()
    Hmat = O.conj().transpose(-2, -1) @ Hmat_flavour @ O

    assert_close(H, Hkin + Hmat, name="H_reduced(NSI) = Hkin + O^dagger Hmat_nsi O")


def test_reduced_with_sterile_matches_manual_kinetic_plus_matter_construction():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced(osc, E, n_e, context=ctx)

    Ured = osc.pmns.reduced()
    mass_sq = osc.mass_spectrum.difference_vector(context=ctx)
    ki_vals = kinetic_potential(mass_sq, E, context=ctx)
    Hkin = (Ured * ki_vals.to(dtype=Ured.dtype)) @ Ured.conj().transpose(-1, -2)
    V = matter_potential_cc(n_e, antinu=osc.antinu, context=ctx)
    Hmat = torch.zeros((4, 4), device=ctx.device, dtype=Hkin.dtype)
    Hmat[0, 0] = V.to(dtype=Hkin.dtype)

    assert_close(H, Hkin + Hmat, name="H_reduced(sterile) = Hkin + diag(V, 0, 0, 0)")


@pytest.mark.parametrize(
    "scenario",
    ["sm_only", "nsi_only", "sterile_only", "combined"],
)
def test_flavour_equals_reduced_transformed_to_flavour_basis(scenario):
    ctx = make_context()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    if scenario == "sm_only":
        osc = make_sm_oscillation(context=ctx)
    elif scenario == "nsi_only":
        osc = make_sm_oscillation(context=ctx, NSI_extension="nsi_dune_etau")
    elif scenario == "sterile_only":
        osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    else:
        osc = make_sterile_oscillation(
            theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx, NSI_extension="nsi_dune_etau",
        )

    H_reduced = hamiltonian_reduced(osc, E, n_e, context=ctx)
    H_flavour = hamiltonian_flavour(osc, E, n_e, context=ctx)
    expected = osc.pmns.flavour_basis(H_reduced, antinu=osc.antinu, device=H_reduced.device, dtype=H_reduced.dtype)

    assert_close(H_flavour, expected, atol=1.0e-10, rtol=1.0e-10, name=f"flavour = flavour_basis(reduced) [{scenario}]")


def test_flavour_equals_reduced_transformed_to_flavour_basis_nonzero_delta14():
    """Regression test for a real convention bug: ``Ured = R13 R12 R14`` is
    genuinely complex whenever the active-sterile CP phase delta14 != 0
    (unlike the real pure-SM reduced matrix, and unlike ``make_sterile_
    oscillation``'s default delta14=0 used by the ``sterile_only``/
    ``combined`` cases above -- which is why those did not catch this).
    ``hamiltonian_reduced`` must build the kinetic term with the Hermitian
    conjugate for this invariant to hold when Ured is complex; using the
    plain transpose (a naive N=3-style generalization) silently breaks it.
    Found via an Earth-level analytical-vs-numerical differential check
    (``medium/earth/test/test4_probabilities.py``).
    """
    ctx = make_context()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, delta14=0.3, DeltamSq41=1.7, context=ctx)
    Ured4 = osc.pmns.reduced()
    assert not torch.allclose(Ured4.imag, torch.zeros_like(Ured4.imag)), (
        "test setup requires a genuinely complex reduced mixing matrix"
    )

    H_reduced = hamiltonian_reduced(osc, E, n_e, context=ctx)
    H_flavour = hamiltonian_flavour(osc, E, n_e, context=ctx)
    expected = osc.pmns.flavour_basis(H_reduced, antinu=osc.antinu, device=H_reduced.device, dtype=H_reduced.dtype)

    assert_close(H_flavour, expected, atol=1.0e-10, rtol=1.0e-10, name="flavour = flavour_basis(reduced) [nonzero delta14]")


@pytest.mark.parametrize(
    "scenario",
    ["nsi_only", "sterile_only", "combined"],
)
def test_hamiltonians_are_hermitian(scenario):
    ctx = make_context()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    if scenario == "nsi_only":
        osc = make_sm_oscillation(context=ctx, NSI_extension="nsi_globalfit_esteban2018")
    elif scenario == "sterile_only":
        osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    else:
        osc = make_sterile_oscillation(
            theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx,
            NSI_extension="nsi_globalfit_esteban2018",
        )

    H_reduced = hamiltonian_reduced(osc, E, n_e, context=ctx)
    H_flavour = hamiltonian_flavour(osc, E, n_e, context=ctx)

    assert_hermitian(H_reduced, f"H_reduced [{scenario}]")
    assert_hermitian(H_flavour, f"H_flavour [{scenario}]")


def test_reduced_and_flavour_scale_linearly_with_evolution_scale():
    ctx = make_context()
    osc = make_sterile_oscillation(
        theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx, NSI_extension="nsi_ee_only",
    )
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    scale = torch.tensor(1234.0, device=DEVICE, dtype=DTYPE)

    H1 = hamiltonian_reduced(osc, E, n_e, context=ctx, evolution_scale_m=scale)
    H2 = hamiltonian_reduced(osc, E, n_e, context=ctx, evolution_scale_m=2.0 * scale)
    assert_close(H2, 2.0 * H1, name="H_reduced scales linearly with evolution scale")

    F1 = hamiltonian_flavour(osc, E, n_e, context=ctx, evolution_scale_m=scale)
    F2 = hamiltonian_flavour(osc, E, n_e, context=ctx, evolution_scale_m=2.0 * scale)
    assert_close(F2, 2.0 * F1, name="H_flavour scales linearly with evolution scale")


# ---------------------------------------------------------------------------
# hamiltonian_matter_reduced -- sterile neutral-current term (n_n_mol_cm3)
# ---------------------------------------------------------------------------


def _manual_diag4(v_ee: torch.Tensor, v_ss: torch.Tensor) -> torch.Tensor:
    """Build diag(v_ee, 0, 0, v_ss) as a complex 4x4 matrix, matching the batch shape of v_ee/v_ss."""
    batch_shape = torch.broadcast_shapes(v_ee.shape, v_ss.shape)
    H = torch.zeros((*batch_shape, 4, 4), device=v_ee.device, dtype=CDTYPE)
    H[..., 0, 0] = v_ee.to(dtype=CDTYPE)
    H[..., 3, 3] = v_ss.to(dtype=CDTYPE)
    return H


def test_hamiltonian_matter_reduced_omitting_n_n_recovers_cc_only_sterile_term():
    """Regression pin: the default (n_n_mol_cm3=None) must stay byte-identical
    to the pre-NC behaviour, i.e. plain diag(V_CC, 0, 0, 0), no rotation.
    """
    ctx = make_context()
    osc4 = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    V_cc = matter_potential_cc(n_e, antinu=osc4.antinu, context=ctx)

    Hmat4 = hamiltonian_matter_reduced(osc4, n_e, context=ctx)
    expected = _manual_diag4(V_cc, torch.zeros_like(V_cc))

    assert_close(Hmat4, expected, name="omitting n_n_mol_cm3 recovers diag(V_CC,0,0,0)")


def test_hamiltonian_matter_reduced_sterile_nc_matches_manual_construction():
    ctx = make_context()
    osc4 = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)
    V_cc = matter_potential_cc(n_e, antinu=osc4.antinu, context=ctx)
    V_nc = matter_potential_nc(n_n, antinu=osc4.antinu, context=ctx)

    Hmat4 = hamiltonian_matter_reduced(osc4, n_e, n_n_mol_cm3=n_n, context=ctx)

    Hmat_flavour = _manual_diag4(V_cc, -V_nc)
    O = osc4.pmns.outer_block(osc4.antinu)
    expected = O.conj().transpose(-2, -1) @ Hmat_flavour @ O

    assert_close(Hmat4, expected, name="diag(V_CC,0,0,-V_NC) rotated by O^dagger (.) O")


def test_hamiltonian_matter_reduced_sterile_nc_breaks_outer_block_invariance():
    """Regression test for the derivation in the module docstring: unlike the
    CC-only (or pure-NSI) matter term, diag(V_CC,0,0,-V_NC) is NOT invariant
    under conjugation by O, since R24/R34 (part of O) genuinely mix mu/tau
    with the now-distinguished sterile index. If a future refactor
    reintroduces the "skip the rotation" shortcut for this case, this test
    catches it.
    """
    ctx = make_context()
    osc4 = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)
    V_cc = matter_potential_cc(n_e, antinu=osc4.antinu, context=ctx)
    V_nc = matter_potential_nc(n_n, antinu=osc4.antinu, context=ctx)
    unrotated = _manual_diag4(V_cc, -V_nc)

    Hmat4 = hamiltonian_matter_reduced(osc4, n_e, n_n_mol_cm3=n_n, context=ctx)

    assert torch.max(torch.abs(Hmat4 - unrotated)) > 1.0e-6, (
        "diag(V_CC,0,0,-V_NC) must not be invariant under O once the sterile "
        "diagonal entry is genuinely nonzero -- theta24/theta34 mix it into "
        "mu/tau"
    )


def test_hamiltonian_matter_reduced_three_flavour_ignores_n_n_mol_cm3():
    """n_n_mol_cm3 is a pure common phase for a 3-flavour pmns (no sterile
    diagonal entry to place -V_NC on) and must be silently inert, not an
    error and not a silently different result.
    """
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(5.0, device=DEVICE, dtype=DTYPE)

    Hmat_with_nc = hamiltonian_matter_reduced(osc, n_e, n_n_mol_cm3=n_n, context=ctx)
    Hmat_without_nc = hamiltonian_matter_reduced(osc, n_e, context=ctx)

    assert_close(Hmat_with_nc, Hmat_without_nc, name="n_n_mol_cm3 has no effect for a 3-flavour pmns")


def test_hamiltonian_matter_reduced_nc_and_nsi_combine_additively():
    ctx = make_context()
    cfg = NSIConfig.from_preset("nsi_globalfit_esteban2018", device=DEVICE, real_dtype=DTYPE)
    osc4 = make_sterile_oscillation(
        theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx,
        NSI_extension="nsi_globalfit_esteban2018",
    )
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)
    V_cc = matter_potential_cc(n_e, antinu=osc4.antinu, context=ctx)
    V_nc = matter_potential_nc(n_n, antinu=osc4.antinu, context=ctx)

    Hmat4 = hamiltonian_matter_reduced(osc4, n_e, n_n_mol_cm3=n_n, context=ctx)

    eps_active = osc4.nsi.epsilon_tensor(n_flavours=4, context=ctx)
    eps_active = osc4.pmns.select_antinu(eps_active, antinu=osc4.antinu)
    Hmat_flavour = V_cc.to(dtype=CDTYPE) * eps_active
    Hmat_flavour[..., 0, 0] += V_cc.to(dtype=CDTYPE)
    Hmat_flavour[..., 3, 3] -= V_nc.to(dtype=CDTYPE)
    O = osc4.pmns.outer_block(osc4.antinu)
    expected = O.conj().transpose(-2, -1) @ Hmat_flavour @ O

    assert_close(Hmat4, expected, name="NSI epsilon and sterile NC term compose additively before rotation")
    assert_close(Hmat4[..., 0, 0].real, (V_cc * (1.0 + cfg.eps_ee)).real, atol=1.0e-9, rtol=1.0e-9, name="electron entry still V_CC*(1+eps_ee)")


def test_hamiltonian_matter_reduced_nc_antinu_flips_sign():
    ctx = make_context()
    osc4_nu = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, antinu=False, context=ctx)
    osc4_anu = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, antinu=True, context=ctx)
    n_e = torch.tensor(2.2, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(2.0, device=DEVICE, dtype=DTYPE)

    Hmat_nu = hamiltonian_matter_reduced(osc4_nu, n_e, n_n_mol_cm3=n_n, context=ctx)
    Hmat_anu = hamiltonian_matter_reduced(osc4_anu, n_e, n_n_mol_cm3=n_n, context=ctx)

    # Both V_CC and V_NC flip sign for antineutrinos, so the *unrotated*
    # flavour-basis matter term for antinu is exactly minus that for nu;
    # O(antinu) itself also differs (select_antinu), so compare the
    # rotation-invariant electron and sterile diagonal entries directly
    # rather than the full reduced-basis matrices.
    assert torch.isfinite(Hmat_nu).all() and torch.isfinite(Hmat_anu).all()
    assert torch.max(torch.abs(Hmat_nu - Hmat_anu)) > 1.0e-6, "antinu must change the NC-extended matter Hamiltonian"


def test_hamiltonian_reduced_and_flavour_thread_n_n_mol_cm3():
    ctx = make_context()
    osc4 = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(1.3, device=DEVICE, dtype=DTYPE)

    H_reduced_nc = hamiltonian_reduced(osc4, E, n_e, n_n_mol_cm3=n_n, context=ctx)
    H_reduced_no_nc = hamiltonian_reduced(osc4, E, n_e, context=ctx)
    Hkin = hamiltonian_kinetic_reduced(osc4, E, osc4.pmns.reduced(antinu=osc4.antinu))
    Hmat_nc = hamiltonian_matter_reduced(osc4, n_e, n_n_mol_cm3=n_n, context=ctx)

    assert_close(H_reduced_nc, Hkin + Hmat_nc, name="hamiltonian_reduced threads n_n_mol_cm3 through to hamiltonian_matter_reduced")
    assert torch.max(torch.abs(H_reduced_nc - H_reduced_no_nc)) > 1.0e-6, "including the NC term must change H_reduced"

    H_flavour_nc = hamiltonian_flavour(osc4, E, n_e, n_n_mol_cm3=n_n, context=ctx)
    expected_flavour = osc4.pmns.flavour_basis(H_reduced_nc, antinu=osc4.antinu, device=H_reduced_nc.device, dtype=H_reduced_nc.dtype)

    assert_close(H_flavour_nc, expected_flavour, name="hamiltonian_flavour threads n_n_mol_cm3 through hamiltonian_reduced")


def test_hamiltonian_with_nc_term_is_hermitian_and_flavour_consistent():
    ctx = make_context()
    osc4 = make_sterile_oscillation(theta14=0.15, theta24=0.10, delta14=0.3, DeltamSq41=1.7, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    n_n = torch.tensor(1.3, device=DEVICE, dtype=DTYPE)

    H_reduced = hamiltonian_reduced(osc4, E, n_e, n_n_mol_cm3=n_n, context=ctx)
    H_flavour = hamiltonian_flavour(osc4, E, n_e, n_n_mol_cm3=n_n, context=ctx)
    expected = osc4.pmns.flavour_basis(H_reduced, antinu=osc4.antinu, device=H_reduced.device, dtype=H_reduced.dtype)

    assert_hermitian(H_reduced, "H_reduced [sterile + NC]")
    assert_hermitian(H_flavour, "H_flavour [sterile + NC]")
    assert_close(H_flavour, expected, atol=1.0e-10, rtol=1.0e-10, name="flavour = flavour_basis(reduced) [sterile + NC]")
