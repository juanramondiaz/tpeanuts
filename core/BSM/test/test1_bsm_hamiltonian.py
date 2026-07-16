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
Pytest-compatible checks for the BSM Hamiltonian builders in
``tpeanuts.core.BSM.hamiltonian``.

These tests cover the generic Hamiltonian machinery shared by every BSM
extension (kinetic builder for arbitrary flavour count, NSI/sterile matter
terms, reduced/flavour-basis assembly and their SM fall-through), independent
of whether the concrete scenario is NSI, sterile, or a combination of both.
NSI-specific and sterile-specific physics checks live in
``test2_bsm_nsi.py`` and ``test3_bsm_sterile.py`` respectively.
"""

from __future__ import annotations

import pytest
import torch

from tpeanuts.core.BSM.hamiltonian import (
    _active_epsilon_matrix,
    _mass_vector_from_pmns_config,
    hamiltonian_flavour_bsm,
    hamiltonian_matter_nsi,
    hamiltonian_matter_sterile,
    hamiltonian_reduced_bsm,
    kinetic_hamiltonian_from_mass_vector,
)
from tpeanuts.core.BSM.NSIConfig import NSIConfig
from tpeanuts.core.BSM.PMNS_sterile import PMNSSterileParams, PMNS_sterile
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_flavour as hamiltonian_flavour_sm,
    hamiltonian_kinetic_reduced,
    hamiltonian_matter_reduced,
    hamiltonian_reduced as hamiltonian_reduced_sm,
    kinetic_mass_squared_vector,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.pmns import PMNSParams
from tpeanuts.core.common.potential import matter_potential
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.test_utils import assert_close


DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDTYPE = torch.complex128


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_sm_oscillation(*, antinu=False, context: RuntimeContext | None = None) -> OscillationParameters:
    return OscillationParameters.from_preset(
        "_SM_NUFIT52_NO",
        antinu=antinu,
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
    return OscillationParameters.build(
        pmns=pmns4,
        DeltamSq21=7.41e-5,
        DeltamSq3l=2.511e-3,
        DeltamSq41=DeltamSq41,
        antinu=antinu,
        context=ctx,
    )


def eye_like(matrix: torch.Tensor) -> torch.Tensor:
    return torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype).expand(matrix.shape)


def assert_hermitian(H: torch.Tensor, name: str) -> None:
    assert_close(H, H.conj().transpose(-2, -1), atol=1.0e-10, rtol=1.0e-10, name=f"{name} is Hermitian")


# ---------------------------------------------------------------------------
# kinetic_hamiltonian_from_mass_vector
# ---------------------------------------------------------------------------

def test_kinetic_hamiltonian_matches_sm_reduced_builder_for_3_flavours():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Ured = osc.pmns.reduced()

    mass_sq = kinetic_mass_squared_vector(osc.DeltamSq21, osc.DeltamSq3l, context=ctx)
    Hkin_generic = kinetic_hamiltonian_from_mass_vector(
        mass_sq, E, Ured, conjugate_right=False,
    )
    Hkin_sm = hamiltonian_kinetic_reduced(osc.DeltamSq21, osc.DeltamSq3l, E, Ured)

    assert_close(Hkin_generic, Hkin_sm, atol=1.0e-12, rtol=1.0e-12, name="generic vs SM kinetic builder")


def test_kinetic_hamiltonian_conjugate_right_true_uses_hermitian_conjugate():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    U = osc.pmns.pmns_matrix()

    mass_sq = kinetic_mass_squared_vector(osc.DeltamSq21, osc.DeltamSq3l, context=ctx)
    Hkin = kinetic_hamiltonian_from_mass_vector(mass_sq, E, U, conjugate_right=True)

    ki = mass_sq  # placeholder shape check only; formula validated below
    expected = None
    # Rebuild the expected matrix directly from the definition U diag(k) U^dagger.
    from tpeanuts.core.common.potential import kinetic_potential
    ki_vals = kinetic_potential(mass_sq, E, context=ctx)
    expected = (U * ki_vals.to(dtype=U.dtype)) @ U.conj().transpose(-1, -2)

    assert_close(Hkin, expected, atol=1.0e-12, rtol=1.0e-12, name="U diag(k) U^dagger")


def test_kinetic_hamiltonian_supports_4_flavour_mixing_matrices():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Ured4 = osc.pmns.reduced()

    mass_sq = _mass_vector_from_pmns_config(
        osc.DeltamSq21, osc.DeltamSq3l, osc.DeltamSq41, osc.pmns, context=ctx,
    )
    Hkin = kinetic_hamiltonian_from_mass_vector(mass_sq, E, Ured4, conjugate_right=False)

    assert Hkin.shape == (4, 4)
    assert torch.isfinite(Hkin.real).all() and torch.isfinite(Hkin.imag).all()


def test_kinetic_hamiltonian_invalid_mixing_matrix_shape_raises():
    with pytest.raises(ValueError, match="mixing_matrix must have final dimensions"):
        kinetic_hamiltonian_from_mass_vector(
            torch.zeros(3, device=DEVICE, dtype=DTYPE),
            1000.0,
            torch.eye(2, device=DEVICE, dtype=CDTYPE),
        )


# ---------------------------------------------------------------------------
# hamiltonian_matter_sterile
# ---------------------------------------------------------------------------

def test_hamiltonian_matter_sterile_formula():
    V = torch.tensor(1.2345, device=DEVICE, dtype=DTYPE)
    Hmat = hamiltonian_matter_sterile(V, context=make_context())
    expected = torch.zeros((4, 4), device=DEVICE, dtype=CDTYPE)
    expected[0, 0] = V.to(dtype=CDTYPE)

    assert Hmat.shape == (4, 4)
    assert_close(Hmat, expected, name="Hmat_sterile = diag(V, 0, 0, 0)")


def test_hamiltonian_matter_sterile_batched():
    V = torch.tensor([0.1, 0.2, 0.3], device=DEVICE, dtype=DTYPE)
    Hmat = hamiltonian_matter_sterile(V, context=make_context())

    assert Hmat.shape == (3, 4, 4)
    assert_close(Hmat[:, 0, 0].real, V, name="batched sterile matter diagonal")
    assert_close(Hmat[:, 1:, :], torch.zeros((3, 3, 4), device=DEVICE, dtype=CDTYPE))


# ---------------------------------------------------------------------------
# hamiltonian_matter_nsi
# ---------------------------------------------------------------------------

def test_hamiltonian_matter_nsi_zero_epsilon_matches_sm_matter_term():
    V = torch.tensor(0.85, device=DEVICE, dtype=DTYPE)
    ctx = make_context()
    eps = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)

    Hmat_nsi = hamiltonian_matter_nsi(V, eps, n_flavours=3, context=ctx)
    Hmat_sm = hamiltonian_matter_reduced(V, context=ctx)

    assert_close(Hmat_nsi, Hmat_sm, name="NSI matter term with eps=0 equals SM matter term")


def test_hamiltonian_matter_nsi_formula_with_offdiagonal_entries():
    ctx = make_context()
    cfg = NSIConfig(eps_ee=0.30, eps_tautau=0.15, eps_mutau_re=0.01, eps_mutau_im=-0.02)
    eps = cfg.epsilon_tensor(device=DEVICE, real_dtype=DTYPE)
    V = torch.tensor(0.5, device=DEVICE, dtype=DTYPE)

    Hmat = hamiltonian_matter_nsi(V, eps, n_flavours=3, context=ctx)
    expected = V.to(dtype=CDTYPE) * (
        torch.diag(torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE)) + eps
    )

    assert_close(Hmat, expected, name="V * (diag(1,0,0) + epsilon)")


def test_hamiltonian_matter_nsi_embeds_3x3_block_into_larger_sterile_matrix():
    ctx = make_context()
    cfg = NSIConfig(eps_ee=0.2, eps_etau_re=0.1)
    eps3 = cfg.epsilon_tensor(device=DEVICE, real_dtype=DTYPE)
    V = torch.tensor(0.7, device=DEVICE, dtype=DTYPE)

    Hmat3 = hamiltonian_matter_nsi(V, eps3, n_flavours=3, context=ctx)
    Hmat4 = hamiltonian_matter_nsi(V, eps3, n_flavours=4, context=ctx)

    assert Hmat4.shape == (4, 4)
    assert_close(Hmat4[:3, :3], Hmat3, name="active 3x3 block embedded unchanged")
    assert_close(Hmat4[3, :], torch.zeros(4, device=DEVICE, dtype=CDTYPE), name="sterile row is zero")
    assert_close(Hmat4[:, 3], torch.zeros(4, device=DEVICE, dtype=CDTYPE), name="sterile column is zero")


def test_hamiltonian_matter_nsi_invalid_epsilon_shape_raises():
    with pytest.raises(ValueError, match="epsilon must have final dimensions"):
        hamiltonian_matter_nsi(
            1.0,
            torch.zeros((2, 2), device=DEVICE, dtype=CDTYPE),
            n_flavours=3,
            context=make_context(),
        )


# ---------------------------------------------------------------------------
# _active_epsilon_matrix / _mass_vector_from_pmns_config (internal helpers)
# ---------------------------------------------------------------------------

def test_active_epsilon_matrix_passthrough_when_shape_matches():
    ctx = make_context()
    eps4 = torch.zeros((4, 4), device=DEVICE, dtype=CDTYPE)
    eps4[0, 1] = 0.1 + 0.05j
    out = _active_epsilon_matrix(eps4, n_flavours=4, context=ctx)
    assert_close(out, eps4, name="4x4 epsilon passthrough")


def test_active_epsilon_matrix_embeds_3x3_for_larger_n_flavours():
    ctx = make_context()
    eps3 = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    eps3[0, 0] = 0.3
    out = _active_epsilon_matrix(eps3, n_flavours=4, context=ctx)

    assert out.shape == (4, 4)
    assert_close(out[:3, :3], eps3, name="embedded active block")
    assert_close(out[3, :], torch.zeros(4, device=DEVICE, dtype=CDTYPE))


def test_active_epsilon_matrix_incompatible_shape_raises():
    with pytest.raises(ValueError, match="epsilon must have final dimensions"):
        _active_epsilon_matrix(
            torch.zeros((2, 2), device=DEVICE, dtype=CDTYPE), n_flavours=4, context=make_context(),
        )


def test_mass_vector_from_pmns_config_returns_none_for_3_flavour_pmns():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    out = _mass_vector_from_pmns_config(
        osc.DeltamSq21, osc.DeltamSq3l, None, osc.pmns, context=ctx,
    )
    assert out is None


def test_mass_vector_from_pmns_config_appends_deltamsq41_for_sterile_pmns():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.1, DeltamSq41=1.7, context=ctx)
    out = _mass_vector_from_pmns_config(
        osc.DeltamSq21, osc.DeltamSq3l, osc.DeltamSq41, osc.pmns, context=ctx,
    )
    expected = torch.cat(
        [kinetic_mass_squared_vector(osc.DeltamSq21, osc.DeltamSq3l, context=ctx),
         osc.DeltamSq41.unsqueeze(-1)],
        dim=-1,
    )
    assert out.shape == (4,)
    assert_close(out, expected, name="4-component mass-squared vector")


def test_mass_vector_from_pmns_config_missing_deltamsq41_raises():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.1, context=ctx)
    with pytest.raises(ValueError, match="requires a supported mass extension"):
        _mass_vector_from_pmns_config(
            osc.DeltamSq21, osc.DeltamSq3l, None, osc.pmns, context=ctx,
        )


# ---------------------------------------------------------------------------
# hamiltonian_reduced_bsm / hamiltonian_flavour_bsm
# ---------------------------------------------------------------------------

def test_reduced_bsm_delegates_exactly_to_sm_builder_when_pure_3_flavour():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_bsm = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=None)
    H_sm = hamiltonian_reduced_sm(osc, E, n_e, context=ctx)

    assert torch.equal(H_bsm, H_sm)


def test_flavour_bsm_delegates_exactly_to_sm_builder_when_pure_3_flavour():
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H_bsm = hamiltonian_flavour_bsm(osc, E, n_e, context=ctx, epsilon=None)
    H_sm = hamiltonian_flavour_sm(osc, E, n_e, context=ctx)

    assert torch.equal(H_bsm, H_sm)


def test_reduced_bsm_with_nsi_matches_manual_kinetic_plus_matter_construction():
    """H_mat in the reduced basis is the flavour-basis NSI matter term
    ``V * (diag(1,0,0) + epsilon)`` rotated by ``O^dagger (.) O``, where
    ``O = R23 . Delta`` -- NOT the flavour-basis term applied verbatim.
    Epsilon is defined in the flavour basis (see NSIConfig); reusing it
    unrotated in the reduced basis is only exact when epsilon is confined to
    the (e,e) entry (see the ``nsi_only``/``combined`` cases of
    ``test_flavour_bsm_equals_reduced_bsm_transformed_to_flavour_basis``).
    """
    ctx = make_context()
    osc = make_sm_oscillation(context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    eps = NSIConfig.from_preset("nsi_globalfit_esteban2018").epsilon_tensor(device=DEVICE, real_dtype=DTYPE)

    H = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=eps)

    Hkin = hamiltonian_kinetic_reduced(osc.DeltamSq21, osc.DeltamSq3l, E, osc.pmns.reduced())
    V = matter_potential(n_e, antinu=osc.antinu, context=ctx)
    Hmat_flavour = hamiltonian_matter_nsi(V, eps, n_flavours=3, context=ctx)
    O = osc.pmns.R23() @ osc.pmns.Delta()
    Hmat = O.conj().transpose(-2, -1) @ Hmat_flavour @ O

    assert_close(H, Hkin + Hmat, name="H_reduced_bsm(NSI) = Hkin + O^dagger Hmat_nsi O")


def test_reduced_bsm_with_sterile_matches_manual_kinetic_plus_matter_construction():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    H = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=None)

    mass_sq = _mass_vector_from_pmns_config(osc.DeltamSq21, osc.DeltamSq3l, osc.DeltamSq41, osc.pmns, context=ctx)
    Hkin = kinetic_hamiltonian_from_mass_vector(mass_sq, E, osc.pmns.reduced(), conjugate_right=False)
    V = matter_potential(n_e, antinu=osc.antinu, context=ctx)
    Hmat = hamiltonian_matter_sterile(V, context=ctx)

    assert_close(H, Hkin + Hmat, name="H_reduced_bsm(sterile) = Hkin + Hmat_sterile")


@pytest.mark.parametrize(
    "scenario",
    ["nsi_only", "sterile_only", "combined"],
)
def test_flavour_bsm_equals_reduced_bsm_transformed_to_flavour_basis(scenario):
    ctx = make_context()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    if scenario == "nsi_only":
        osc = make_sm_oscillation(context=ctx)
        eps = NSIConfig.from_preset("nsi_dune_etau").epsilon_tensor(device=DEVICE, real_dtype=DTYPE)
    elif scenario == "sterile_only":
        osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
        eps = None
    else:
        osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
        eps = NSIConfig.from_preset("nsi_dune_etau").epsilon_tensor(device=DEVICE, real_dtype=DTYPE)

    H_reduced = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=eps)
    H_flavour = hamiltonian_flavour_bsm(osc, E, n_e, context=ctx, epsilon=eps)
    expected = osc.pmns.H_flavour_basis(H_reduced, antinu=osc.antinu, device=H_reduced.device, dtype=H_reduced.dtype)

    assert_close(H_flavour, expected, atol=1.0e-10, rtol=1.0e-10, name=f"flavour = H_flavour_basis(reduced) [{scenario}]")


@pytest.mark.parametrize(
    "scenario",
    ["nsi_only", "sterile_only", "combined"],
)
def test_bsm_hamiltonians_are_hermitian(scenario):
    ctx = make_context()
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)

    if scenario == "nsi_only":
        osc = make_sm_oscillation(context=ctx)
        eps = NSIConfig.from_preset("nsi_globalfit_esteban2018").epsilon_tensor(device=DEVICE, real_dtype=DTYPE)
    elif scenario == "sterile_only":
        osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
        eps = None
    else:
        osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
        eps = NSIConfig.from_preset("nsi_globalfit_esteban2018").epsilon_tensor(device=DEVICE, real_dtype=DTYPE)

    H_reduced = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=eps)
    H_flavour = hamiltonian_flavour_bsm(osc, E, n_e, context=ctx, epsilon=eps)

    assert_hermitian(H_reduced, f"H_reduced_bsm [{scenario}]")
    assert_hermitian(H_flavour, f"H_flavour_bsm [{scenario}]")


def test_reduced_and_flavour_bsm_scale_linearly_with_evolution_scale():
    ctx = make_context()
    osc = make_sterile_oscillation(theta14=0.15, theta24=0.10, DeltamSq41=1.7, context=ctx)
    E = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    eps = NSIConfig.from_preset("nsi_ee_only").epsilon_tensor(device=DEVICE, real_dtype=DTYPE)
    scale = torch.tensor(1234.0, device=DEVICE, dtype=DTYPE)

    H1 = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=eps, evolution_scale_m=scale)
    H2 = hamiltonian_reduced_bsm(osc, E, n_e, context=ctx, epsilon=eps, evolution_scale_m=2.0 * scale)
    assert_close(H2, 2.0 * H1, name="H_reduced_bsm scales linearly with evolution scale")

    F1 = hamiltonian_flavour_bsm(osc, E, n_e, context=ctx, epsilon=eps, evolution_scale_m=scale)
    F2 = hamiltonian_flavour_bsm(osc, E, n_e, context=ctx, epsilon=eps, evolution_scale_m=2.0 * scale)
    assert_close(F2, 2.0 * F1, name="H_flavour_bsm scales linearly with evolution scale")
