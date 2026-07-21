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

import dataclasses

import pytest
import torch

from tpeanuts.core.BSM.bsm_mass_spectrum import MassSpectrum_BSM
from tpeanuts.core.BSM.bsm_nsi import NSIConfig
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_kinetic_reduced,
    hamiltonian_reduced,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.config.propagation import PropagationConfig
from tpeanuts.core.common.potential import matter_potential_cc, matter_potential_nc
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
        n_flavours: int = 3,
        potential_n: torch.Tensor | None = None,
        residual_scale_n: torch.Tensor | float = 0.0,
        perturbation_n: torch.Tensor | bool = False,
    ) -> None:
        self.average = average
        self.potential = potential
        self.length = length
        self.zero_mask = zero_mask
        self.residual_scale = torch.as_tensor(residual_scale, device=average.device, dtype=average.dtype)
        self.perturbation = torch.as_tensor(perturbation, device=average.device, dtype=torch.bool)
        self.n_flavours = n_flavours
        self.potential_n = potential_n
        self.residual_scale_n = torch.as_tensor(residual_scale_n, device=average.device, dtype=average.dtype)
        self.perturbation_n = torch.as_tensor(perturbation_n, device=average.device, dtype=torch.bool)

    def residual_integral(self, la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        del la, lb
        shape = torch.broadcast_shapes(self.length.shape, self.residual_scale.shape)
        scale = torch.broadcast_to(self.residual_scale, shape)
        n = self.n_flavours
        return scale[..., None, None] * torch.ones((*shape, n, n), device=self.average.device, dtype=self.average.dtype)

    def has_perturbation(self) -> torch.Tensor:
        shape = torch.broadcast_shapes(self.length.shape, self.perturbation.shape)
        return torch.broadcast_to(self.perturbation, shape)

    def residual_integral_neutron(self, la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        del la, lb
        shape = torch.broadcast_shapes(self.length.shape, self.residual_scale_n.shape)
        scale = torch.broadcast_to(self.residual_scale_n, shape)
        n = self.n_flavours
        return scale[..., None, None] * torch.ones((*shape, n, n), device=self.average.device, dtype=self.average.dtype)

    def has_perturbation_neutron(self) -> torch.Tensor:
        shape = torch.broadcast_shapes(self.length.shape, self.perturbation_n.shape)
        return torch.broadcast_to(self.perturbation_n, shape)


def make_context() -> RuntimeContext:
    return RuntimeContext.resolve(DEVICE, DTYPE)


def make_oscillation(*, antinu=False, NSI_extension: str | None = None) -> OscillationParameters:
    return PropagationConfig.oscillation_parameters_from_preset(
        "_SM_NUFIT52_NO", antinu=antinu, NSI_extension=NSI_extension, context=make_context(),
    )


def make_sterile_oscillation(*, antinu=False, NSI_extension: str | None = None) -> OscillationParameters:
    """Realistic 4-flavour (3+1 sterile) preset (Giunti et al. 2017 best fit).

    Note: this preset's sterile CP phases are exactly zero (no sensitivity in
    the underlying reactor/gallium data it is fit to), so its reduced mixing
    matrix happens to be real -- see ``make_sterile_oscillation_with_cp_phase``
    for a case that exercises a genuinely complex reduced matrix.
    """
    return PropagationConfig.oscillation_parameters_from_preset(
        "sterile_3p1_bestfit_giunti2017", antinu=antinu, NSI_extension=NSI_extension, context=make_context(),
    )


def make_sterile_oscillation_with_cp_phase(
    *, antinu=False, NSI_extension: str | None = None,
) -> OscillationParameters:
    """4-flavour oscillation with a non-zero active-sterile CP phase delta14,
    so ``pmns.reduced()`` is genuinely complex (see
    ``test_evolutor_perturbative_segment_sterile_trace_uses_honest_diagonal_not_sum_ki``).
    """
    from tpeanuts.core.BSM.bsm_sterile import PMNSSterileParams, PMNS_sterile
    from tpeanuts.core.common.pmns import PMNSParams

    ctx = make_context()
    sm_params = PMNSParams(theta12=0.5836, theta13=0.1498, theta23=0.8552, delta=3.438, context=ctx)
    sterile_params = PMNSSterileParams(
        theta14=0.15, theta24=0.10, theta34=0.05,
        delta14=0.3, delta24=-0.2, delta34=0.0,
        context=ctx,
    )
    pmns4 = PMNS_sterile(sm_params, sterile_params)
    nsi_obj = None
    if NSI_extension is not None:
        nsi_obj = NSIConfig.from_preset(NSI_extension, device=ctx.device, real_dtype=ctx.dtype)
    mass_spectrum = MassSpectrum_BSM(
        DeltamSq21=torch.as_tensor(7.41e-5, device=ctx.device, dtype=ctx.dtype),
        DeltamSq3l=torch.as_tensor(2.511e-3, device=ctx.device, dtype=ctx.dtype),
        DeltamSq41=torch.as_tensor(1.7, device=ctx.device, dtype=ctx.dtype),
    )
    return OscillationParameters(pmns=pmns4, mass_spectrum=mass_spectrum, antinu=antinu, nsi=nsi_obj)


def with_epsilon(cfg: NSIConfig) -> NSIConfig:
    """Populate a directly-constructed NSIConfig's epsilon field (only
    from_preset does this automatically)."""
    return dataclasses.replace(cfg, epsilon=cfg.epsilon_tensor_base(device=DEVICE, real_dtype=DTYPE))


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
    potential = matter_potential_cc(average, antinu=False, context=ctx)
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
        osc,
        energy,
        osc.pmns.reduced(),
        return_ki=True,
    )
    Hmat = torch.zeros((3, 3), device=DEVICE, dtype=CDTYPE)
    Hmat[0, 0] = potential.to(dtype=CDTYPE)
    H = Hkin + Hmat
    trace_H = (ki[..., 0] + ki[..., 1] + ki[..., 2] + potential).to(dtype=CDTYPE)
    U0 = evolutor_zero_order(H, length, trace_H=trace_H)

    assert_close(U, U0, name="segment constant profile equals U0", atol=1e-10, rtol=1e-10)


def test_evolutor_perturbative_segment_zero_length_identity():
    ctx = make_context()
    osc = make_oscillation()
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
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


# ---------------------------------------------------------------------------
# Fase 4: N-agnostic evolutor (3+1 sterile, NSI)
# ---------------------------------------------------------------------------

def make_hermitian4() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0 + 0.0j, 0.2 + 0.1j, 0.05 - 0.03j, 0.02 + 0.01j],
            [0.2 - 0.1j, 2.0 + 0.0j, 0.3 + 0.2j, -0.04 + 0.02j],
            [0.05 + 0.03j, 0.3 - 0.2j, 3.0 + 0.0j, 0.06 - 0.01j],
            [0.02 - 0.01j, -0.04 - 0.02j, 0.06 + 0.01j, -1.5 + 0.0j],
        ],
        device=DEVICE,
        dtype=CDTYPE,
    )


def identity4(batch_shape=()) -> torch.Tensor:
    return torch.eye(4, device=DEVICE, dtype=CDTYPE).expand(*batch_shape, 4, 4)


def test_evolutor_zero_order_n4_matches_matrix_exp():
    H = make_hermitian4()
    length = torch.tensor(0.29, device=DEVICE, dtype=DTYPE)

    U0 = evolutor_zero_order(H, length)
    expected = torch.matrix_exp(-1j * H * length)

    assert U0.shape == (4, 4)
    assert_close(U0, expected, name="N=4 zero-order evolutor equals matrix_exp", atol=1e-10, rtol=1e-10)


def test_evolutor_first_order_p_none_matches_explicit_rank1_p():
    """``P=None`` must be bit-identical to the general sandwich formula given
    the same ``P=diag(1,0,...,0)`` it is a fast-path shortcut for (the "no
    shortcuts" check called for by the Fase 4 plan).
    """
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

    U1_fast = evolutor_first_order(spectral["M"], spectral["lam"], spectral["trace"], profile)
    P = torch.diag(torch.tensor([1.0, 0.0, 0.0], device=DEVICE, dtype=CDTYPE))
    U1_general = evolutor_first_order(spectral["M"], spectral["lam"], spectral["trace"], profile, P=P)

    assert torch.equal(U1_fast, U1_general)


def test_evolutor_first_order_p_nc_adds_independent_sandwich_term():
    """``P_nc`` must add a genuinely independent term on top of the CC
    correction, using the profile's ``residual_integral_neutron`` and
    ``matter_potential_nc`` (not ``matter_potential_cc``)."""
    H = make_hermitian4()
    spectral = hamiltonian_spectral_data(H)
    profile = DummyProfileModel(
        average=torch.tensor(1.0, device=DEVICE, dtype=DTYPE),
        potential=torch.tensor(0.2, device=DEVICE, dtype=DTYPE),
        length=torch.tensor(0.37, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
        n_flavours=4,
        residual_scale_n=2.0e-4,
        perturbation_n=True,
    )
    P = torch.eye(4, device=DEVICE, dtype=CDTYPE)
    P_nc = torch.diag(torch.tensor([0.0, 0.0, 0.0, -1.0], device=DEVICE, dtype=CDTYPE))

    U1_cc_only = evolutor_first_order(spectral["M"], spectral["lam"], spectral["trace"], profile, P=P)
    U1_with_nc = evolutor_first_order(spectral["M"], spectral["lam"], spectral["trace"], profile, P=P, P_nc=P_nc)

    from tpeanuts.core.common.potential import matter_potential_nc

    integral_nc = profile.residual_integral_neutron(la=None, lb=None)
    potential_correction_nc = matter_potential_nc(integral_nc, antinu=False, evolution_scale_m=6371000.0).to(dtype=spectral["M"].dtype)
    expected_extra = (-1j) * torch.einsum(
        "...ab,...aik,...kl,...blj->...ij", potential_correction_nc, spectral["M"], P_nc, spectral["M"],
    )

    assert_close(U1_with_nc - U1_cc_only, expected_extra, name="P_nc contributes an independent sandwich term", atol=1.0e-10, rtol=1.0e-10)


def test_evolutor_perturbative_segment_epsilon_zero_matches_epsilon_none_n3():
    """``epsilon=0`` (explicit, not None) forces the general BSM/NSI branch
    (no shortcuts) and must numerically match the fast ``epsilon=None`` path.
    """
    ctx = make_context()
    osc_sm = make_oscillation()
    osc_nsi = make_oscillation(NSI_extension="sm_no_nsi")
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
    )

    U_none = evolutor_perturbative_segment(osc_sm, energy, profile)
    U_zero = evolutor_perturbative_segment(osc_nsi, energy, profile)

    assert_close(U_zero, U_none, atol=1.0e-12, rtol=1.0e-12, name="epsilon=0 matches epsilon=None")


def test_evolutor_perturbative_segment_sterile_constant_profile_matches_manual_u0():
    ctx = make_context()
    osc = make_sterile_oscillation()
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=0.0,
        perturbation=False,
        n_flavours=4,
    )

    U = evolutor_perturbative_segment(osc, energy, profile)

    Ured4 = osc.pmns.reduced()
    Hkin = hamiltonian_kinetic_reduced(osc, energy, Ured4)
    Hmat = torch.zeros((4, 4), device=DEVICE, dtype=Hkin.dtype)
    Hmat[0, 0] = potential.to(dtype=Hkin.dtype)
    H = Hkin + Hmat
    trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)
    U0 = evolutor_zero_order(H, length, trace_H=trace_H)

    assert U.shape == (4, 4)
    assert_close(U, U0, name="N=4 segment constant profile equals U0", atol=1.0e-10, rtol=1.0e-10)


def test_evolutor_perturbative_segment_include_matter_nc_matches_hamiltonian_reduced():
    """A constant (no first-order perturbation) N=4 segment with
    ``include_matter_nc=True`` must build the exact same Hamiltonian as
    ``core.common.hamiltonian.hamiltonian_reduced(..., n_n_mol_cm3=...)`` --
    the already-tested single source of truth for the NC matter term.
    """
    ctx = make_context()
    osc = make_sterile_oscillation()
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    average_n = torch.tensor(1.2, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    potential_n = matter_potential_nc(average_n, antinu=False, context=ctx)
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=0.0,
        perturbation=False,
        n_flavours=4,
        potential_n=potential_n,
    )

    U = evolutor_perturbative_segment(osc, energy, profile, include_matter_nc=True)

    H = hamiltonian_reduced(osc, energy, average, n_n_mol_cm3=average_n, context=ctx)
    trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)
    U0 = evolutor_zero_order(H, length, trace_H=trace_H)

    assert_close(U, U0, name="N=4 segment with NC constant profile equals hamiltonian_reduced-based U0", atol=1.0e-10, rtol=1.0e-10)


def test_evolutor_perturbative_segment_include_matter_nc_raises_without_potential_n():
    ctx = make_context()
    osc = make_sterile_oscillation()
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=torch.tensor(0.37, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(False, device=DEVICE),
        n_flavours=4,
    )

    with pytest.raises(ValueError, match="neutron-density coefficients"):
        evolutor_perturbative_segment(osc, energy, profile, include_matter_nc=True)


def test_evolutor_perturbative_segment_include_matter_nc_is_noop_for_three_flavour():
    ctx = make_context()
    osc = make_oscillation()
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=torch.tensor(0.37, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(False, device=DEVICE),
        n_flavours=3,
    )

    U_cc = evolutor_perturbative_segment(osc, energy, profile, include_matter_nc=False)
    U_nc = evolutor_perturbative_segment(osc, energy, profile, include_matter_nc=True)

    assert_close(U_nc, U_cc, atol=1.0e-13, rtol=1.0e-13, name="3-flavour segment ignores include_matter_nc")


def test_evolutor_perturbative_segment_sterile_zero_length_identity():
    ctx = make_context()
    osc = make_sterile_oscillation()
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=torch.tensor(0.0, device=DEVICE, dtype=DTYPE),
        zero_mask=torch.tensor(True, device=DEVICE),
        residual_scale=1.0e-4,
        perturbation=True,
        n_flavours=4,
    )

    U = evolutor_perturbative_segment(osc, torch.tensor(1000.0, device=DEVICE, dtype=DTYPE), profile)

    assert_close(U, identity4(), name="N=4 segment zero length identity")


def test_evolutor_perturbative_segment_sterile_kinetic_trace_matches_sum_ki_for_complex_ured():
    """``Ured4 = R13 R12 R14`` is genuinely complex whenever the active-sterile
    CP phase delta14 != 0 (unlike the real pure-SM reduced matrix). Since
    ``Ured4`` is still unitary, ``trace(Ured4 diag(ki) Ured4^dagger) ==
    sum(ki)`` regardless (a similarity transform by a unitary matrix always
    preserves trace) -- this only holds with the Hermitian-conjugate
    convention (``conjugate_right=True``); using the plain transpose instead
    (as a naive N=3-style generalization would) silently breaks it for
    complex Ured4. Regression test for that convention bug, found via an
    Earth-level analytical-vs-numerical differential check.
    """
    ctx = make_context()
    osc = make_sterile_oscillation_with_cp_phase()
    Ured4 = osc.pmns.reduced()
    assert not torch.allclose(Ured4.imag, torch.zeros_like(Ured4.imag)), (
        "test setup requires a genuinely complex reduced mixing matrix"
    )

    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    Hkin, ki = hamiltonian_kinetic_reduced(osc, energy, Ured4, return_ki=True)
    trace_Hkin = torch.diagonal(Hkin, dim1=-2, dim2=-1).sum(dim=-1)

    assert_close(
        trace_Hkin, ki.sum(dim=-1).to(dtype=CDTYPE),
        atol=1.0e-10, rtol=1.0e-10,
        name="trace(Hkin) == sum(ki) for complex Ured4 under conjugate_right=True",
    )


def test_evolutor_perturbative_segment_sterile_nsi_trace_uses_honest_diagonal_not_sum_ki():
    """Regression test for the ``evolutor_perturbative_segment`` BSM branch's
    honest ``trace_H = torch.diagonal(H, ...).sum(-1)``: for a combined
    sterile+NSI segment the NSI coupling contributes non-electron diagonal
    entries (see ``nsi_globalfit_esteban2018``'s nonzero mu-mu/tau-tau
    epsilon), so ``trace(Hmat) != V`` and the naive ``trace_H = sum(ki) + V``
    shortcut (valid for the plain electron-only matter term) silently
    undercounts it.
    """
    ctx = make_context()
    osc = make_sterile_oscillation_with_cp_phase(NSI_extension="nsi_globalfit_esteban2018")
    eps = osc.nsi.epsilon

    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    average = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(average, antinu=False, context=ctx)
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=average,
        potential=potential,
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=0.0,
        perturbation=False,
        n_flavours=4,
    )

    U = evolutor_perturbative_segment(osc, energy, profile)

    Ured4 = osc.pmns.reduced()
    Hkin, ki = hamiltonian_kinetic_reduced(osc, energy, Ured4, return_ki=True)
    O = osc.pmns.outer_block().to(dtype=CDTYPE)
    D_flavour = torch.zeros(4, 4, device=DEVICE, dtype=CDTYPE)
    D_flavour[:3, :3] = eps
    D_flavour[0, 0] += 1.0
    Hmat = potential.to(dtype=CDTYPE) * (O.conj().transpose(-2, -1) @ D_flavour @ O)
    H = Hkin + Hmat
    trace_H_honest = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)
    trace_H_naive = (ki.sum(dim=-1) + potential).to(dtype=CDTYPE)

    assert float((trace_H_honest - trace_H_naive).abs()) > 1.0e-6, (
        "test setup should exercise a case where the naive shortcut is wrong"
    )

    U0_honest = evolutor_zero_order(H, length, trace_H=trace_H_honest)
    U0_naive = evolutor_zero_order(H, length, trace_H=trace_H_naive)

    assert_close(U, U0_honest, name="segment uses honest trace_H", atol=1.0e-10, rtol=1.0e-10)
    assert float((U - U0_naive).abs().max()) > 1.0e-6


@pytest.mark.parametrize("n_flavours", [3, 4])
def test_evolutor_perturbative_segment_nsi_matches_hamiltonian_reduced(n_flavours):
    """Cross-validate the Fase 4 BSM branch's Hamiltonian construction
    against the independently-implemented, already-tested
    ``hamiltonian_reduced`` (see ``core/BSM/test/test2_bsm_nsi.py``).
    """
    ctx = make_context()
    base_osc = make_sterile_oscillation() if n_flavours == 4 else make_oscillation()
    osc = dataclasses.replace(base_osc, nsi=with_epsilon(NSIConfig(eps_ee=0.05, eps_emu_re=0.02)))
    energy = torch.tensor(1000.0, device=DEVICE, dtype=DTYPE)
    n_e = torch.tensor(1.5, device=DEVICE, dtype=DTYPE)
    potential = matter_potential_cc(n_e, antinu=False, context=ctx)
    length = torch.tensor(0.37, device=DEVICE, dtype=DTYPE)
    profile = DummyProfileModel(
        average=n_e,
        potential=potential,
        length=length,
        zero_mask=torch.tensor(False, device=DEVICE),
        residual_scale=0.0,
        perturbation=False,
        n_flavours=n_flavours,
    )

    U = evolutor_perturbative_segment(osc, energy, profile)

    H_ref = hamiltonian_reduced(osc, energy, n_e, context=ctx)
    trace_H_ref = torch.diagonal(H_ref, dim1=-2, dim2=-1).sum(dim=-1)
    U0_ref = evolutor_zero_order(H_ref, length, trace_H=trace_H_ref)

    assert U.shape == (n_flavours, n_flavours)
    assert_close(
        U, U0_ref, atol=1.0e-10, rtol=1.0e-10,
        name=f"NSI N={n_flavours} segment matches hamiltonian_reduced",
    )
