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

"""Propagation Hamiltonian builders, common to every flavour scenario.

This module contains the *only* Hamiltonian assembly code in the project.
There is no separate Standard Model (SM) or BSM (NSI / 3+1 sterile)
implementation: every scenario is described entirely by the ``oscillation``
object passed in (its ``pmns`` -- ``tpeanuts.core.SM.sm_pmns.PMNS_SM`` or
``tpeanuts.core.BSM.bsm_sterile.PMNS_sterile`` -- and its optional ``nsi``,
see ``tpeanuts.core.common.oscillation.OscillationParameters``), so the
functions here are correct and complete for the 3-flavour Standard Model,
the 3+1 sterile extension, Non-Standard Interactions (NSI), or any
combination, without branching on which module built the objects involved.

Physics background
-------------------
The propagation Hamiltonian in natural units (hbar = c = 1) splits into a
kinetic piece (vacuum oscillations) and a matter piece (coherent forward
scattering on the medium):

    H = H_kin + H_mat

``H_kin`` is built from the neutrino mass-squared splittings projected onto
the mixing matrix:

    H_kin = U diag(k_1, ..., k_n) U^dagger

where ``k_i`` are the dimensionless kinetic eigenvalues returned by
``kinetic_eigenvalue_vector`` (effectively Delta m^2_i1 * L / (2E) up to the
``evolution_scale_m`` normalisation), and ``U`` is whichever mixing matrix
the target basis needs (the reduced matrix ``oscillation.pmns.reduced()``
for the reduced-basis builders below).

``H_mat`` carries the Standard Model MSW charged-current potential on the
electron-flavour diagonal entry, ``diag(V_CC, 0, ..., 0)`` with
``V_CC = +-sqrt(2) G_F n_e`` (sign set by the neutrino/antineutrino
convention), optionally extended by Non-Standard Interactions (NSI):

    H_mat^NSI = V_CC * (diag(1, 0, ..., 0) + epsilon)

where ``epsilon`` is the (possibly complex, Hermitian) NSI coupling matrix
contributed by ``tpeanuts.core.BSM.bsm_nsi.NSIConfig.epsilon_tensor_base``,
accessed only through the duck-typed ``oscillation.nsi.epsilon_tensor(...)``
-- this module never imports ``core.BSM``.

For 3+1 sterile-neutrino propagation (4-flavour PMNS objects), the
mass-squared vector is extended with a fourth eigenvalue derived from
``DeltamSq41`` (the sterile mass splitting in eV^2, handled by
``kinetic_eigenvalue_vector`` below). The sterile state is a CC/NC singlet,
so it never receives ``V_CC``, but it also never receives the neutral-current
potential ``V_NC = -+(1/sqrt(2)) G_F n_n`` (same sign convention as
``V_CC``) that the three active flavours *do* share equally. In the pure
3-flavour Standard Model that shared ``V_NC`` is a common phase with no
effect on probabilities and is correctly never computed at all. Once a
sterile state is present it is no longer an irrelevant common phase: after
subtracting it as an overall ``V_NC * I_4`` phase, a genuine relative
potential survives on the sterile diagonal entry,

    diag(V_CC + V_NC, V_NC, V_NC, 0) - V_NC * I_4 = diag(V_CC, 0, 0, -V_NC),

optionally supplied via ``n_n_mol_cm3`` on ``hamiltonian_matter_reduced``
(``core.common.potential.matter_potential_nc``); omitting it (the default)
recovers the CC-only sterile matter term used throughout earlier phases of
this project. Because ``-V_NC`` sits on the sterile diagonal rather than the
electron one, it does **not** commute with the ``outer_block`` ``O`` that
the pure-CC (and pure-NSI) matter terms are invariant under (see
``hamiltonian_matter_reduced``), so including it forces the general
flavour-basis-then-rotate construction even when NSI is off.

NSI and the sterile extension (with or without the NC term) can be combined
freely: the 3x3 ``epsilon`` block is embedded in the top-left corner of the
4x4 (or larger) matter matrix, and the ``-V_NC`` sterile-diagonal term (when
supplied) is an independent additive contribution -- both live in the
flavour basis and are rotated into the reduced basis together by the same
``O^dagger @ (.) @ O``.

Which extension is active is controlled entirely by
``tpeanuts.core.common.oscillation.OscillationParameters``:
``oscillation.BSM_extension_sterile`` is auto-derived from
``oscillation.pmns.n_flavours == 4``, and ``oscillation.BSM_extension_NSI``
from whether ``oscillation.nsi`` is set. There is no separate ``epsilon``
argument anywhere below -- there is exactly one place NSI configuration
lives, so there is nothing left to validate for agreement. The
neutral-current density ``n_n_mol_cm3`` remains a plain argument (not part
of ``OscillationParameters``) since it describes the medium, not the
oscillation scenario -- symmetric with ``n_e_mol_cm3``.

Module functions:
    kinetic_eigenvalue_vector
        Converts the mass-squared-difference vector (built via
        ``oscillation.mass_spectrum.difference_vector``) into dimensionless
        kinetic eigenvalues through ``kinetic_potential``. It supports both
        the active three-flavour and sterile 3+1 cases.
    hamiltonian_kinetic_reduced
        Reduced-basis kinetic Hamiltonian: builds ``U diag(k_i) U^dagger``
        for any flavour dimension, always using the Hermitian conjugate --
        the only convention for which ``pmns.flavour_basis(Hkin) ==
        U_full diag(k) U_full^dagger`` holds exactly (``U_full = O @ U``).
        For the pure-SM/NSI-only reduced matrix (real, no active-sterile CP
        phase) this coincides numerically with the plain transpose; for the
        3+1 sterile reduced matrix ``Ured = R13 R12 R14`` -- genuinely
        complex whenever the active-sterile CP phase ``delta14 != 0`` -- the
        Hermitian conjugate is the only correct choice.
    hamiltonian_matter_reduced
        Reduced-basis matter Hamiltonian. Takes the fast path -- plain
        ``diag(V_CC, 0, ..., 0)``, no rotation -- only when neither NSI nor a
        sterile NC term is requested (both are exactly invariant under
        ``O`` there). Otherwise builds the full matter term in the flavour
        basis (``V_CC * (diag(1,0,...) + epsilon)`` plus, for a 4-flavour
        ``pmns`` with ``n_n_mol_cm3`` supplied, ``-V_NC`` on the sterile
        diagonal) and rotates it into the reduced basis via
        ``O^dagger @ (.) @ O`` (see ``PMNS.outer_block``).
    hamiltonian_reduced
        Reduced-basis (mass-eigenstate-adjacent) Hamiltonian, supporting
        NSI, the 3+1 sterile extension (with its optional NC term), or any
        combination:
        ``hamiltonian_kinetic_reduced(...) + hamiltonian_matter_reduced(...)``.
    hamiltonian_flavour
        Flavour-basis counterpart of ``hamiltonian_reduced``: dresses its
        output via ``pmns.flavour_basis``.
"""

from __future__ import annotations

from typing import Optional

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.potential import (
    kinetic_potential,
    matter_potential_cc,
    matter_potential_nc,
)
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real


@torch.no_grad()
def kinetic_eigenvalue_vector(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the kinetic eigenvalues in the mass basis.

    Args:
        oscillation: Oscillation parameters supplying ``mass_spectrum``
            (see ``tpeanuts.core.common.mass_spectrum.MassSpectrum``), whose
            ``difference_vector()`` sizes the mass-squared vector to
            ``oscillation.pmns``'s flavour count.
        E_MeV: Neutrino energy in MeV.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.
        evolution_scale_m: Positive evolution length scale in metres.
        legacy_precision: Accepted for propagation-chain consistency. It is
            forwarded to ``kinetic_potential`` and does not alter the kinetic
            calculation.

    Returns:
        Dimensionless kinetic eigenvalues shaped ``(..., 3)`` or ``(..., 4)``.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(
            E_MeV,
            oscillation.mass_spectrum.DeltamSq21,
            oscillation.mass_spectrum.DeltamSq3l,
            evolution_scale_m,
        )
    resolved_context = RuntimeContext(device=device, dtype=dtype)
    E_MeV = as_tensor(E_MeV, device=device, dtype=dtype)
    mass_squared_differences = oscillation.mass_spectrum.difference_vector(
        context=resolved_context,
    )

    return kinetic_potential(
        mass_squared_differences,
        E_MeV,
        evolution_scale_m=evolution_scale_m,
        context=resolved_context,
        legacy_precision=legacy_precision,
    )


@torch.no_grad()
def hamiltonian_kinetic_reduced(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    Ured: torch.Tensor,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    return_ki: bool = False,
    legacy_precision: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build the reduced-basis kinetic Hamiltonian.

    Diagonalises the vacuum (kinetic) part of the propagation Hamiltonian in
    the mass basis via ``kinetic_eigenvalue_vector`` (which sizes the
    mass-squared vector to ``oscillation.pmns``'s flavour count, appending
    ``DeltamSq41`` for a 3+1 sterile ``pmns``), then rotates it as
    ``Ured diag(k_i) Ured^dagger`` into the reduced basis. Used by both
    ``hamiltonian_reduced`` and (transitively, via it) ``hamiltonian_flavour``.

    Args:
        oscillation: ``OscillationParameters`` supplying ``mass_spectrum``
            and ``pmns`` (via ``kinetic_eigenvalue_vector``).
        E_MeV: Neutrino energy in MeV.
        Ured: Reduced mixing matrix shaped ``(..., n_flavours, n_flavours)``.
        evolution_scale_m: Positive evolution length scale in metres,
            entering the dimensionless kinetic potential normalisation.
        return_ki: If True, also return the kinetic eigenvalues ``ki``,
            avoiding a second, duplicated ``mass_vector -> ki -> Hkin``
            computation in callers (e.g. ``core.perturbative``) that need
            both the Hamiltonian and its trace (``sum(ki)`` for a unitary
            ``Ured``).
        legacy_precision: Accepted for propagation-chain consistency and
            forwarded to ``kinetic_eigenvalue_vector``.

    Returns:
        Complex Hamiltonian shaped ``(..., n_flavours, n_flavours)`` or
        ``(Hkin, ki)``.
    """
    device, dtype = infer_device_dtype(
        E_MeV,
        oscillation.mass_spectrum.DeltamSq21,
        oscillation.mass_spectrum.DeltamSq3l,
        evolution_scale_m,
        device=Ured.device,
        dtype=Ured.real.dtype,
    )
    ki = kinetic_eigenvalue_vector(
        oscillation,
        E_MeV,
        evolution_scale_m=evolution_scale_m,
        context=RuntimeContext(device=device, dtype=dtype),
        legacy_precision=legacy_precision,
    )

    cdtype = Ured.dtype
    U = Ured.to(device=device, dtype=cdtype)
    n_flavours = ki.shape[-1]
    if U.ndim < 2 or U.shape[-2:] != (n_flavours, n_flavours):
        raise ValueError(
            "Ured must have final dimensions "
            f"({n_flavours}, {n_flavours}), got {tuple(U.shape[-2:])}."
        )
    Hkin = (U * ki.to(dtype=cdtype)[..., None, :]) @ U.conj().transpose(-1, -2)

    if return_ki:
        return Hkin, ki
    return Hkin


def _diag_entry(
    value: torch.Tensor,
    index: int,
    n_flavours: int,
    cdtype: torch.dtype,
) -> torch.Tensor:
    """Build a ``(..., n_flavours, n_flavours)`` tensor, ``value`` at ``[index, index]``, zero elsewhere.

    Relies on ordinary broadcasting (``value[..., None, None] * onehot``)
    rather than a pre-allocated zeros tensor with indexed assignment, so the
    leading batch shape never has to be resolved by hand.
    """
    onehot = torch.zeros(n_flavours, n_flavours, device=value.device, dtype=cdtype)
    onehot[index, index] = 1.0
    return value.to(dtype=cdtype)[..., None, None] * onehot


def _matter_direction_reduced(
    oscillation: OscillationParameters,
    *,
    antinu,
    include_nc: bool,
    context: RuntimeContext,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Position-independent reduced-basis matter "direction" matrices.

    Decouples the reduced-basis matter Hamiltonian
    ``Hmat = V_cc * P_cc [+ V_nc * P_nc]`` into its (V-independent)
    direction matrices and their (position-dependent) potential magnitudes.
    Used by ``hamiltonian_matter_reduced`` and by the perturbative
    evolutor's first-order correction
    (``core.perturbative.evolutor.evolutor_first_order``), which needs the
    direction matrices on their own to sandwich the spectral projectors.

    ``P_cc`` is the plain ``diag(1, 0, ..., 0)`` (no rotation -- exactly
    invariant under ``O``) when NSI is off, else the flavour-basis
    ``diag(1,0,...,0) + epsilon`` rotated into the reduced basis. ``P_nc`` is
    None unless ``include_nc``, in which case it is the flavour-basis
    ``-diag_entry(1, n_flavours - 1)`` (see the module docstring for the
    sign) rotated into the reduced basis -- it never commutes with ``O``, so
    it always needs the rotation, independent of which branch built
    ``P_cc``.

    Args:
        oscillation: ``OscillationParameters`` supplying ``pmns`` and the
            optional ``nsi`` (NSIConfig) attribute.
        antinu: Antineutrino selector for ``pmns.outer_block`` and
            ``pmns.select_antinu``. Passed explicitly (not read from
            ``oscillation.antinu``) so callers that override antinu per
            segment/trajectory (e.g. ``evolutor_perturbative_segment``) stay
            consistent with the potential magnitude's sign convention.
        include_nc: If True, also build and return ``P_nc``. Only meaningful
            when ``oscillation.pmns.n_flavours == 4``.
        context: Runtime device/dtype for the returned tensors.

    Returns:
        ``(P_cc, P_nc)``, with ``P_nc`` None unless ``include_nc``.
    """
    pmns = oscillation.pmns
    n_flavours = int(pmns.n_flavours)
    device, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)

    if not oscillation.BSM_extension_NSI and not include_nc:
        return (
            _diag_entry(torch.ones((), device=device, dtype=dtype), 0, n_flavours, cdtype),
            None,
        )

    D_flavour = torch.zeros(n_flavours, n_flavours, device=device, dtype=cdtype)
    D_flavour[0, 0] = 1.0
    if oscillation.BSM_extension_NSI:
        eps_active = oscillation.nsi.epsilon_tensor(n_flavours=n_flavours, context=context)
        eps_active = pmns.select_antinu(eps_active, antinu=antinu)
        D_flavour = D_flavour + eps_active

    O = pmns.outer_block(antinu).to(device=device, dtype=cdtype)
    P_cc = O.conj().transpose(-2, -1) @ D_flavour @ O

    P_nc = None
    if include_nc:
        Dn_flavour = torch.zeros(n_flavours, n_flavours, device=device, dtype=cdtype)
        Dn_flavour[n_flavours - 1, n_flavours - 1] = -1.0
        P_nc = O.conj().transpose(-2, -1) @ Dn_flavour @ O

    return P_cc, P_nc


@torch.no_grad()
def hamiltonian_matter_reduced(
    oscillation: OscillationParameters,
    n_e_mol_cm3: TensorLike,
    *,
    n_n_mol_cm3: Optional[TensorLike] = None,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the reduced-basis matter Hamiltonian, with optional NSI and NC extensions.

    Plain matter term ``diag(V_CC, 0, ..., 0)`` -- exactly invariant under
    the outer mixing block ``O``, so no rotation is needed when neither
    ``oscillation.nsi`` nor a genuine sterile neutral-current term is
    requested. 
    
    Otherwise builds the full matter term in the flavour basis
    and rotates it into the reduced basis via ``O^dagger @ (.) @ O`` (see
    ``PMNS.outer_block``), so that

    ``hamiltonian_flavour(...) == oscillation.pmns.flavour_basis(
    hamiltonian_reduced(...))`` 
    
    holds in general. 
    
    That flavour-basis term has up to two independent, additive pieces:

        1. NSI, when ``oscillation.nsi`` is set:
           ``V_CC * (diag(1,0,...,0) + oscillation.nsi.epsilon)``, with the
           3x3 active-flavour epsilon block embedded via
           ``oscillation.nsi.epsilon_tensor`` for the 3+1 sterile case.
           Setting ``epsilon = 0`` recovers the plain CC matter term
           exactly. See ``tpeanuts.core.BSM.bsm_nsi`` for the physical
           bounds on ``epsilon`` entries and named presets.
           
        2. The sterile NC term, when ``pmns.n_flavours == 4`` and
           ``n_n_mol_cm3`` is supplied: ``-V_NC`` on the sterile diagonal
           entry (``core.common.potential.matter_potential_nc``; see the
           module docstring for the derivation). Omitted by default,
           recovering the CC-only sterile matter term used throughout
           earlier phases of this project. Has no effect for a 3-flavour
           ``pmns`` (there ``V_NC`` is a pure common phase, so
           ``n_n_mol_cm3`` is silently ignored rather than raising, since
           passing it is mathematically inert, not wrong).

    Args:
        oscillation: ``OscillationParameters`` supplying ``pmns``, ``antinu``,
            and the optional ``nsi`` (NSIConfig) attribute.
        n_e_mol_cm3: Electron density in mol/cm^3.
        n_n_mol_cm3: Optional neutron density in mol/cm^3, enabling the
            sterile neutral-current term above. Only meaningful when
            ``pmns.n_flavours == 4``.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the density and evolution scale.
        evolution_scale_m: Positive evolution length scale in metres.
        legacy_precision: If True, use the legacy charged-current
            matter-potential prefactor. Has no NC counterpart (see
            ``matter_potential_nc``), so it never affects the NC term.

    Returns:
        Complex reduced-basis matter Hamiltonian shaped
        (..., n_flavours, n_flavours).
    """
    pmns = oscillation.pmns
    antinu = oscillation.antinu
    n_flavours = int(pmns.n_flavours)

    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(n_e_mol_cm3, evolution_scale_m)
    cdtype = cdtype_from_real(dtype)
    resolved_context = RuntimeContext(device=device, dtype=dtype)
    V_cc = matter_potential_cc(
        n_e_mol_cm3,
        antinu=antinu,
        evolution_scale_m=evolution_scale_m,
        context=resolved_context,
        legacy_precision=legacy_precision,
    )

    include_nc = n_n_mol_cm3 is not None and n_flavours == 4

    P_cc, P_nc = _matter_direction_reduced(
        oscillation, antinu=antinu, include_nc=include_nc, context=resolved_context,
    )
    Hmat = V_cc.to(dtype=cdtype)[..., None, None] * P_cc

    if include_nc:
        V_nc = matter_potential_nc(
            n_n_mol_cm3,
            antinu=antinu,
            evolution_scale_m=evolution_scale_m,
            context=resolved_context,
        )
        Hmat = Hmat + V_nc.to(dtype=cdtype)[..., None, None] * P_nc

    return Hmat


@torch.no_grad()
def hamiltonian_reduced(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    *,
    n_n_mol_cm3: Optional[TensorLike] = None,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Assemble the reduced-basis Hamiltonian from electron (and optional neutron) density.

    Supports the 3-flavour Standard Model, NSI alone (``oscillation.nsi``
    set), the 3+1 sterile extension alone (``oscillation.BSM_extension_sterile``,
    i.e. a 4-flavour ``oscillation.pmns``), and both simultaneously (4-flavour
    ``oscillation.pmns`` with ``oscillation.nsi`` also set, where the 3x3
    active-flavour ``epsilon`` block is embedded in the top-left corner of
    the 4x4 matter matrix, see ``NSIConfig.epsilon_tensor``). Assembles
    ``H = H_kin + H_mat`` via ``hamiltonian_kinetic_reduced`` and
    ``hamiltonian_matter_reduced``.

    ``oscillation.nsi``/``BSM_extension_sterile`` are the single source of
    truth for which extension is active (see ``OscillationParameters``);
    there is no separate ``epsilon`` argument to keep in sync.

    Args:
        oscillation: ``OscillationParameters`` bundling the PMNS object
            (active 3-flavour or 3+1 sterile), the ``mass_spectrum`` (eV^2
            splittings), the antinu selection flag, and the optional ``nsi``
            (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron number density of the propagation medium, in
            mol/cm^3 (i.e. N_A * electrons/cm^3).
        n_n_mol_cm3: Optional neutron number density in mol/cm^3, enabling
            the sterile neutral-current term (see
            ``hamiltonian_matter_reduced``). Only meaningful for a
            4-flavour ``oscillation.pmns``; omitted by default.
        context: Optional runtime device/dtype. When omitted, both are
            taken from ``oscillation.pmns``.
        evolution_scale_m: Positive evolution length scale in metres used
            to non-dimensionalise both the kinetic and matter potentials.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor when building the matter Hamiltonian.

    Returns:
        Dimensionless reduced Hamiltonian shaped
        (..., n_flavours, n_flavours), with n_flavours inferred from
        ``oscillation.pmns``.
    """
    pmns = oscillation.pmns
    antinu = oscillation.antinu

    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = pmns.device, pmns.dtype
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    Ured = pmns.reduced(antinu=antinu)

    Hkin = hamiltonian_kinetic_reduced(
        oscillation,
        E_MeV,
        Ured,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )
    Hmat = hamiltonian_matter_reduced(
        oscillation,
        n_e_mol_cm3,
        n_n_mol_cm3=n_n_mol_cm3,
        context=RuntimeContext(device=Hkin.device, dtype=Hkin.real.dtype),
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    return Hkin + Hmat


@torch.no_grad()
def hamiltonian_flavour(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    *,
    n_n_mol_cm3: Optional[TensorLike] = None,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the full flavour-basis Hamiltonian from electron (and optional neutron) density.

    Flavour-basis counterpart of ``hamiltonian_reduced``: builds the
    reduced-basis Hamiltonian and dresses it with ``pmns.flavour_basis``.
    Supports the 3-flavour Standard Model, NSI alone, the 3+1 sterile
    extension alone (with or without its optional neutral-current term), or
    any combination (see ``hamiltonian_reduced`` for the exact combination
    rules).

    Args:
        oscillation: ``OscillationParameters`` bundling the PMNS object
            (active 3-flavour or 3+1 sterile), the ``mass_spectrum`` (eV^2
            splittings), the antinu selection flag, and the optional ``nsi``
            (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron number density of the propagation medium, in
            mol/cm^3.
        n_n_mol_cm3: Optional neutron number density in mol/cm^3, enabling
            the sterile neutral-current term (see
            ``hamiltonian_matter_reduced``). Only meaningful for a
            4-flavour ``oscillation.pmns``; omitted by default.
        context: Optional runtime device/dtype. When omitted, both are
            taken from ``oscillation.pmns``.
        evolution_scale_m: Positive evolution length scale in metres used
            to non-dimensionalise both the kinetic and matter potentials.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in the reduced Hamiltonian.

    Returns:
        Dimensionless flavour-basis Hamiltonian shaped
        (..., n_flavours, n_flavours).
    """
    H_reduced = hamiltonian_reduced(
        oscillation,
        E_MeV,
        n_e_mol_cm3,
        n_n_mol_cm3=n_n_mol_cm3,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    return oscillation.pmns.flavour_basis(
        H_reduced,
        antinu=oscillation.antinu,
        device=H_reduced.device,
        dtype=H_reduced.dtype,
    )
