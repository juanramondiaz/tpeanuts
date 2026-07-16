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

"""BSM Hamiltonian extensions for NSI and sterile-neutrino scenarios.

This module contains the Hamiltonian pieces that do not belong to the pure
three-flavour Standard Model core:

    - arbitrary-dimension kinetic Hamiltonians,
    - 3+1 sterile mass vectors using ``DeltamSq41``,
    - NSI matter Hamiltonians,
    - BSM reduced and flavour-basis Hamiltonian builders.

Physics background
-------------------
The propagation Hamiltonian in natural units (hbar = c = 1) splits into a
kinetic piece (vacuum oscillations) and a matter piece (coherent forward
scattering on the medium):

    H = H_kin + H_mat

``H_kin`` is built from the neutrino mass-squared splittings projected onto
the flavour basis through the mixing matrix:

    H_kin = U diag(k_1, ..., k_n) U^dagger   (or U^T, see ``conjugate_right``)

where ``k_i`` are the dimensionless kinetic eigenvalues returned by
``kinetic_potential`` (effectively Delta m^2_i1 * L / (2E) up to the
``evolution_scale_m`` normalisation).

``H_mat`` carries the Standard Model MSW charged-current potential on the
electron-flavour diagonal entry, ``diag(V, 0, ..., 0)`` with
``V = +-sqrt(2) G_F n_e`` (sign set by the neutrino/antineutrino
convention), optionally extended by Non-Standard Interactions (NSI):

    H_mat^NSI = V * (diag(1, 0, ..., 0) + epsilon)

where ``epsilon`` is the (possibly complex, Hermitian) NSI coupling matrix
contributed by ``tpeanuts.core.BSM.NSIConfig.NSIConfig.epsilon_tensor``.

For 3+1 sterile-neutrino propagation (4-flavour PMNS objects, e.g.
``tpeanuts.core.BSM.PMNS_sterile.PMNS_sterile``), the mass-squared vector is
extended with a fourth eigenvalue derived from ``DeltamSq41`` (the sterile
mass splitting in eV^2), and the matter potential remains confined to the
electron-flavour entry since the sterile state is a CC/NC singlet. NSI and
the sterile extension can be combined: the 3x3 ``epsilon`` block is then
embedded in the top-left corner of the 4x4 (or larger) matter matrix and
the remaining sterile rows/columns are left at zero.

Module functions:
    kinetic_hamiltonian_from_mass_vector
        Generic ``U diag(k) U^dagger`` (or transpose) builder for any
        flavour dimension.
    hamiltonian_matter_sterile
        3+1 sterile matter Hamiltonian ``diag(V, 0, 0, 0)``.
    hamiltonian_matter_nsi
        NSI-extended matter Hamiltonian ``V * (diag(1,0,...) + epsilon)``.
    hamiltonian_reduced_bsm
        Reduced-basis (mass-eigenstate-adjacent) Hamiltonian with optional
        NSI and/or sterile extension; falls back to the pure SM builder
        when neither extension is requested.
    hamiltonian_flavour_bsm
        Flavour-basis counterpart of ``hamiltonian_reduced_bsm``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.hamiltonian import (
    hamiltonian_flavour as hamiltonian_flavour_sm,
    hamiltonian_kinetic_reduced,
    hamiltonian_matter_reduced,
    hamiltonian_reduced as hamiltonian_reduced_sm,
    kinetic_mass_squared_vector,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.potential import kinetic_potential, matter_potential
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real


def _validate_square_matrix(
    matrix: torch.Tensor,
    *,
    name: str,
    n_flavours: int,
) -> None:
    """Raise ``ValueError`` if *matrix* does not end in ``(n_flavours, n_flavours)``.

    Args:
        matrix: Tensor to validate.
        name: Human-readable identifier used in the error message.
        n_flavours: Expected size of the final two dimensions.

    Raises:
        ValueError: If the final two dimensions of *matrix* differ from
            ``(n_flavours, n_flavours)``.
    """
    if matrix.ndim < 2 or matrix.shape[-2:] != (n_flavours, n_flavours):
        raise ValueError(
            f"{name} must have final dimensions "
            f"({n_flavours}, {n_flavours}), got {tuple(matrix.shape[-2:])}."
        )


def _mass_vector_from_pmns_config(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    DeltamSq41: Optional[TensorLike],
    pmns: object,
    *,
    context: RuntimeContext,
) -> Optional[torch.Tensor]:
    """Build the mass-squared difference vector for a BSM PMNS object.

    Returns ``None`` for 3-flavour objects so the caller can fall through to
    the Standard Model path.  For 4-flavour (3+1 sterile) objects it appends
    ``DeltamSq41`` to the standard 3-flavour vector
    ``[0, DeltamSq21, DeltamSq3l]``, returning a tensor of shape
    ``(..., 4)``.

    Args:
        DeltamSq21: Mass-squared splitting m²₂ − m²₁ in eV².
        DeltamSq3l: Mass-squared splitting m²₃ − m²₁ (NH) or
            m²₃ − m²₂ (IH) in eV².
        DeltamSq41: Mass-squared splitting m²₄ − m²₁ in eV².  Required
            when ``pmns.n_flavours == 4``; ignored otherwise.
        pmns: PMNS-like object exposing ``n_flavours`` as an integer.
        context: Runtime device/dtype for tensor creation.

    Returns:
        ``None`` if ``pmns.n_flavours == 3``; otherwise a tensor shaped
        ``(..., n_flavours)`` with the full mass-squared vector.

    Raises:
        ValueError: If ``pmns.n_flavours`` is unsupported (not 3 or 4), or
            if ``pmns.n_flavours == 4`` but ``DeltamSq41`` is ``None``.
    """
    device, dtype = context.device, context.dtype
    n_flavours = int(pmns.n_flavours)
    if n_flavours == 3:
        return None

    if n_flavours == 4 and DeltamSq41 is not None:
        base = kinetic_mass_squared_vector(
            DeltamSq21,
            DeltamSq3l,
            context=context,
        )
        DeltamSq41_t = as_tensor(DeltamSq41, device=device, dtype=dtype)
        return torch.cat(
            [base, DeltamSq41_t.expand(base.shape[:-1]).unsqueeze(-1)],
            dim=-1,
        )

    raise ValueError(
        "BSM Hamiltonian construction for PMNS objects with "
        f"{n_flavours} flavours requires a supported mass extension such as "
        "DeltamSq41."
    )


def _active_epsilon_matrix(
    epsilon: torch.Tensor,
    *,
    n_flavours: int,
    context: RuntimeContext,
) -> torch.Tensor:
    """Embed or validate *epsilon* for a Hamiltonian with *n_flavours* flavours.

    If *epsilon* is already shaped ``(..., n_flavours, n_flavours)`` it is
    returned as-is (after a device/dtype cast).  If *n_flavours* > 3 and
    *epsilon* is shaped ``(..., 3, 3)``, it is embedded in the top-left
    corner of an ``(n_flavours, n_flavours)`` zero matrix — the sterile-sector
    rows and columns carry no NSI coupling.

    Args:
        epsilon: NSI coupling matrix.  Accepted last two dimensions are
            ``(n_flavours, n_flavours)`` or ``(3, 3)`` when ``n_flavours``
            > 3.
        n_flavours: Total number of flavours of the target Hamiltonian.
        context: Runtime device/dtype.

    Returns:
        Complex tensor shaped ``(..., n_flavours, n_flavours)``.

    Raises:
        ValueError: If *epsilon* has an incompatible shape.
    """
    device, dtype = context.device, context.dtype
    cdtype = cdtype_from_real(dtype)
    eps = epsilon.to(device=device, dtype=cdtype)

    if eps.shape[-2:] == (n_flavours, n_flavours):
        return eps

    if n_flavours > 3 and eps.shape[-2:] == (3, 3):
        out = torch.zeros(
            (*eps.shape[:-2], n_flavours, n_flavours),
            device=device,
            dtype=cdtype,
        )
        out[..., :3, :3] = eps
        return out

    raise ValueError(
        "epsilon must have final dimensions "
        f"({n_flavours}, {n_flavours}) or active block (3, 3); "
        f"got {tuple(eps.shape[-2:])}."
    )


def _outer_block(
    pmns: object,
    antinu: Union[bool, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the ``outer`` PMNS block ``O`` that commutes with ``diag(V, 0, ..., 0)``.

    ``O = R23 . Delta`` for the 3-flavour SM, ``O = R23 . Delta . R24 . R34``
    for the 3+1 sterile extension -- exactly the block used internally by
    ``pmns.operator_flavour_basis`` (``O @ op_reduced @ O^dagger``). Since
    ``O`` is unitary, the inverse conjugation ``O^dagger @ (.) @ O`` maps a
    flavour-basis operator into the reduced basis; this is what
    ``hamiltonian_reduced_bsm`` needs to rotate an NSI matter term (defined
    in the flavour basis, see ``tpeanuts.core.BSM.NSIConfig``) into the
    reduced basis it assembles.

    Args:
        pmns: PMNS-compatible object exposing ``R23``/``Delta`` (and, for
            4-flavour sterile objects, ``R24``/``R34``), ``select_antinu``,
            and ``n_flavours``.
        antinu: Bool or boolean tensor selecting the antineutrino convention.
        device: Target torch device.
        dtype: Target complex dtype.

    Returns:
        Complex unitary tensor shaped (..., n_flavours, n_flavours).
    """
    r23 = pmns.select_antinu(pmns.R23(), antinu).to(device=device, dtype=dtype)
    delta = pmns.select_antinu(pmns.Delta(), antinu).to(device=device, dtype=dtype)
    O = r23 @ delta

    if int(pmns.n_flavours) == 4:
        r24 = pmns.select_antinu(pmns.R24(), antinu).to(device=device, dtype=dtype)
        r34 = pmns.select_antinu(pmns.R34(), antinu).to(device=device, dtype=dtype)
        O = O @ r24 @ r34

    return O


def kinetic_hamiltonian_from_mass_vector(
    mass_squared_vector: TensorLike,
    E_MeV: TensorLike,
    mixing_matrix: torch.Tensor,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    conjugate_right: bool = True,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the kinetic Hamiltonian ``U diag(k_i) U^dagger`` (or ``U^T``).

    Diagonalises the vacuum (kinetic) part of the propagation Hamiltonian in
    the mass basis, then rotates it into whatever basis ``mixing_matrix``
    maps to. The eigenvalues ``k_i`` are the dimensionless kinetic
    potentials computed from the mass-squared splittings and neutrino
    energy via ``kinetic_potential`` — proportional to
    ``Delta m^2_i1 * evolution_scale_m / (2 E)`` in natural units. This
    builder works for any flavour dimension (3 for the SM, 4+ for sterile
    extensions), unlike ``hamiltonian_kinetic_reduced`` which is
    3-flavour-only.

    Args:
        mass_squared_vector: Mass-squared splittings relative to the
            lightest state, shape (..., n_flavours), in eV^2.
        E_MeV: Neutrino energy in MeV.
        mixing_matrix: Complex mixing matrix shaped (..., n_flavours,
            n_flavours) used to rotate the diagonal kinetic matrix into the
            target basis (e.g. the reduced or full PMNS matrix).
        evolution_scale_m: Positive evolution length scale in metres,
            entering the dimensionless kinetic potential normalisation.
        conjugate_right: If True, the Hamiltonian is built as
            ``U diag(k) U^dagger`` (Hermitian conjugate on the right) — the
            standard convention for the flavour basis. If False, it uses
            ``U diag(k) U^T`` (plain transpose), which is the convention
            used internally for the reduced basis where ``U`` is the
            "reduced" mixing matrix rather than the full PMNS matrix.

    Returns:
        Complex kinetic Hamiltonian shaped (..., n_flavours, n_flavours).
    """
    device, dtype = infer_device_dtype(
        mass_squared_vector,
        E_MeV,
        evolution_scale_m,
        device=mixing_matrix.device,
        dtype=mixing_matrix.real.dtype,
    )
    mass_sq = as_tensor(mass_squared_vector, device=device, dtype=dtype)
    E = as_tensor(E_MeV, device=device, dtype=dtype)
    ki = kinetic_potential(
        mass_sq,
        E,
        evolution_scale_m=evolution_scale_m,
        context=RuntimeContext(device=device, dtype=dtype),
        legacy_precision=legacy_precision,
    )

    cdtype = mixing_matrix.dtype
    U = mixing_matrix.to(device=device, dtype=cdtype)
    _validate_square_matrix(U, name="mixing_matrix", n_flavours=ki.shape[-1])
    right = U.conj().transpose(-1, -2) if conjugate_right else U.transpose(-1, -2)
    return (U * ki.to(dtype=cdtype)[..., None, :]) @ right


@torch.no_grad()
def hamiltonian_matter_sterile(
    V: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Build the 3+1 sterile matter Hamiltonian ``diag(V, 0, 0, 0)``.

    The sterile neutrino carries no CC or NC matter potential; only the
    electron-flavour entry is non-zero. Direct 4x4 generalization of
    ``tpeanuts.core.common.hamiltonian.hamiltonian_matter_reduced``.

    Args:
        V: Dimensionless charged-current matter potential, batch shape
            (...). Must already carry the neutrino/antineutrino sign.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from ``V``.

    Returns:
        Complex matter Hamiltonian shaped (..., 4, 4).
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(V)
    V_t = as_tensor(V, device=device, dtype=dtype)
    cdtype = cdtype_from_real(dtype)

    Hmat = torch.zeros((*V_t.shape, 4, 4), device=device, dtype=cdtype)
    Hmat[..., 0, 0] = V_t.to(dtype=cdtype)
    return Hmat


@torch.no_grad()
def hamiltonian_matter_nsi(
    V: TensorLike,
    epsilon: torch.Tensor,
    *,
    n_flavours: int = 3,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Build the NSI matter Hamiltonian ``V * (diag(1,0,0,...) + epsilon)``.

    Implements the Non-Standard Interaction extension of the MSW matter
    potential: the standard charged-current term ``V * diag(1,0,...,0)``
    (electron-flavour only) is augmented by ``V * epsilon``, where
    ``epsilon`` is a dimensionless Hermitian coupling matrix representing
    new-physics four-fermion operators that contribute to coherent forward
    scattering. Setting ``epsilon = 0`` recovers the Standard Model matter
    Hamiltonian exactly. See ``tpeanuts.core.BSM.NSIConfig`` for the
    physical bounds on ``epsilon`` entries and named presets.

    Args:
        V: Dimensionless charged-current matter potential
            (``+-sqrt(2) G_F n_e * evolution_scale_m``, sign already
            carrying the neutrino/antineutrino convention), batch shape
            (...).
        epsilon: Complex NSI coupling matrix. Accepted shapes are
            (..., n_flavours, n_flavours), or (..., 3, 3) when
            ``n_flavours`` > 3 — in the latter case the 3x3 active-flavour
            block is embedded in the top-left corner and the
            sterile-sector entries are zero (see ``_active_epsilon_matrix``).
        n_flavours: Total number of flavours of the output Hamiltonian
            (3 for pure active, 4 for 3+1 sterile, ...).
        context: Optional runtime device/dtype. When omitted, both are
            inferred from ``V``.

    Returns:
        Complex matter Hamiltonian shaped (..., n_flavours, n_flavours).
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(V)
    V_t = as_tensor(V, device=device, dtype=dtype)
    cdtype = cdtype_from_real(dtype)
    eps = _active_epsilon_matrix(
        epsilon,
        n_flavours=n_flavours,
        context=RuntimeContext(device=device, dtype=dtype),
    )
    Hmat = eps.clone()
    Hmat[..., 0, 0] += 1.0
    return V_t.to(dtype=cdtype)[..., None, None] * Hmat


@torch.no_grad()
def hamiltonian_reduced_bsm(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build a reduced-basis Hamiltonian with optional NSI or sterile extension.

    Generalises ``tpeanuts.core.common.hamiltonian.hamiltonian_reduced`` to
    BSM scenarios: when ``oscillation.pmns`` is a plain 3-flavour PMNS
    object and no NSI ``epsilon`` is given, this delegates directly to the
    Standard Model builder (identical numerics). Otherwise it assembles
    ``H = H_kin + H_mat`` itself, where:

    - ``H_kin`` uses the reduced mixing matrix ``pmns.reduced(antinu)`` and
      either the 3-component SM mass-squared vector or, for 4-flavour
      (3+1 sterile) PMNS objects, a 4-component vector with the extra
      sterile splitting ``DeltamSq41`` appended (see
      ``_mass_vector_from_pmns_config``);
    - ``H_mat`` is the plain SM matter term ``diag(V,0,...,0)`` when
      ``epsilon`` is None, or, when an NSI ``epsilon`` is given, the
      flavour-basis NSI matter term ``V * (diag(1,0,...,0) + epsilon)``
      (via ``hamiltonian_matter_nsi``) rotated into the reduced basis via
      ``O^dagger @ (.) @ O`` (see ``_outer_block``), so that
      ``hamiltonian_flavour_bsm(...) == oscillation.pmns.H_flavour_basis(
      hamiltonian_reduced_bsm(...))`` holds for arbitrary (not just
      diagonal-uniform) ``epsilon``; the sterile state never receives a
      matter potential.

    Args:
        oscillation: ``OscillationParameters`` bundling the PMNS object
            (active 3-flavour or 3+1 sterile), the mass splittings
            ``DeltamSq21``/``DeltamSq3l`` (eV^2), and the antinu selection
            flag.
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron number density of the propagation medium, in
            mol/cm^3 (i.e. N_A * electrons/cm^3).
        context: Optional runtime device/dtype. When omitted, both are
            taken from ``oscillation.pmns``.
        evolution_scale_m: Positive evolution length scale in metres used
            to non-dimensionalise both the kinetic and matter potentials.
        epsilon: Optional NSI coupling matrix (see
            ``hamiltonian_matter_nsi``). When None, no NSI term is added
            and only the SM matter potential is used.

    Returns:
        Dimensionless reduced Hamiltonian shaped
        (..., n_flavours, n_flavours), with n_flavours inferred from
        ``oscillation.pmns``.
    """
    pmns = oscillation.pmns
    antinu = oscillation.antinu
    n_flavours = int(pmns.n_flavours)
    if n_flavours == 3 and epsilon is None:
        return hamiltonian_reduced_sm(
            oscillation,
            E_MeV,
            n_e_mol_cm3,
            context=context,
            evolution_scale_m=evolution_scale_m,
            legacy_precision=legacy_precision,
        )

    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = pmns.device, pmns.dtype
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    V = matter_potential(
        n_e_mol_cm3,
        antinu=antinu,
        evolution_scale_m=evolution_scale_m,
        context=resolved_context,
        legacy_precision=legacy_precision,
    )
    Ured = pmns.reduced(antinu=antinu)
    mass_squared_vector = _mass_vector_from_pmns_config(
        oscillation.DeltamSq21,
        oscillation.DeltamSq3l,
        oscillation.DeltamSq41,
        pmns,
        context=RuntimeContext(device=V.device, dtype=V.dtype),
    )

    if mass_squared_vector is None:
        Hkin = hamiltonian_kinetic_reduced(
            DeltamSq21=oscillation.DeltamSq21,
            DeltamSq3l=oscillation.DeltamSq3l,
            E_MeV=E_MeV,
            Ured=Ured,
            evolution_scale_m=evolution_scale_m,
            legacy_precision=legacy_precision,
        )
    else:
        Hkin = kinetic_hamiltonian_from_mass_vector(
            mass_squared_vector,
            E_MeV,
            Ured,
            evolution_scale_m=evolution_scale_m,
            conjugate_right=False,
            legacy_precision=legacy_precision,
        )

    if epsilon is None:
        Hmat = torch.zeros(
            (*V.shape, n_flavours, n_flavours),
            device=Hkin.device,
            dtype=Hkin.dtype,
        )
        Hmat[..., 0, 0] = V.to(dtype=Hkin.dtype)
    else:
        eps = pmns.select_antinu(
            epsilon.to(device=Hkin.device, dtype=Hkin.dtype),
            antinu=antinu,
        )
        Hmat_flavour = hamiltonian_matter_nsi(
            V,
            eps,
            n_flavours=n_flavours,
            context=RuntimeContext(device=Hkin.device, dtype=Hkin.real.dtype),
        )
        O = _outer_block(pmns, antinu, device=Hkin.device, dtype=Hkin.dtype)
        Hmat = O.conj().transpose(-2, -1) @ Hmat_flavour @ O

    return Hkin + Hmat


@torch.no_grad()
def hamiltonian_flavour_bsm(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build a flavour-basis Hamiltonian with optional NSI or sterile extension.

    Flavour-basis counterpart of ``hamiltonian_reduced_bsm``: falls back to
    the pure Standard Model builder
    (``tpeanuts.core.common.hamiltonian.hamiltonian_flavour``) when the
    PMNS object is 3-flavour and no NSI ``epsilon`` is supplied; otherwise
    assembles ``H = H_kin + H_mat`` directly in the flavour basis using the
    full PMNS mixing matrix ``pmns.pmns_matrix(antinu)`` (rather than the
    reduced matrix), so that ``H_kin = U diag(k) U^dagger``. The matter
    term is identical in construction to ``hamiltonian_reduced_bsm``: plain
    SM ``diag(V,0,...,0)`` when ``epsilon`` is None, otherwise the
    NSI-extended ``V * (diag(1,0,...,0) + epsilon)``.

    Args:
        oscillation: ``OscillationParameters`` bundling the PMNS object
            (active 3-flavour or 3+1 sterile), the mass splittings
            ``DeltamSq21``/``DeltamSq3l`` (eV^2), and the antinu selection
            flag.
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron number density of the propagation medium, in
            mol/cm^3.
        context: Optional runtime device/dtype. When omitted, both are
            taken from ``oscillation.pmns``.
        evolution_scale_m: Positive evolution length scale in metres used
            to non-dimensionalise both the kinetic and matter potentials.
        epsilon: Optional NSI coupling matrix (see
            ``hamiltonian_matter_nsi``). When None, no NSI term is added.

    Returns:
        Dimensionless flavour-basis Hamiltonian shaped
        (..., n_flavours, n_flavours).
    """
    pmns = oscillation.pmns
    antinu = oscillation.antinu
    n_flavours = int(pmns.n_flavours)
    if n_flavours == 3 and epsilon is None:
        return hamiltonian_flavour_sm(
            oscillation,
            E_MeV,
            n_e_mol_cm3,
            context=context,
            evolution_scale_m=evolution_scale_m,
            legacy_precision=legacy_precision,
        )

    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = pmns.device, pmns.dtype
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    V = matter_potential(
        n_e_mol_cm3,
        antinu=antinu,
        evolution_scale_m=evolution_scale_m,
        context=resolved_context,
        legacy_precision=legacy_precision,
    )
    mass_squared_vector = _mass_vector_from_pmns_config(
        oscillation.DeltamSq21,
        oscillation.DeltamSq3l,
        oscillation.DeltamSq41,
        pmns,
        context=RuntimeContext(device=V.device, dtype=V.dtype),
    )
    if mass_squared_vector is None:
        mass_squared_vector = kinetic_mass_squared_vector(
            oscillation.DeltamSq21,
            oscillation.DeltamSq3l,
            context=RuntimeContext(device=V.device, dtype=V.dtype),
        )

    U = pmns.pmns_matrix(antinu=antinu).to(
        device=V.device,
        dtype=cdtype_from_real(V.dtype),
    )
    Hkin = kinetic_hamiltonian_from_mass_vector(
        mass_squared_vector,
        E_MeV,
        U,
        evolution_scale_m=evolution_scale_m,
        conjugate_right=True,
        legacy_precision=legacy_precision,
    )

    if epsilon is None:
        Hmat = torch.zeros(
            (*V.shape, n_flavours, n_flavours),
            device=Hkin.device,
            dtype=Hkin.dtype,
        )
        Hmat[..., 0, 0] = V.to(dtype=Hkin.dtype)
    else:
        eps = pmns.select_antinu(
            epsilon.to(device=V.device, dtype=Hkin.dtype),
            antinu=antinu,
        )
        Hmat = hamiltonian_matter_nsi(
            V,
            eps,
            n_flavours=n_flavours,
            context=RuntimeContext(device=V.device, dtype=V.dtype),
        )

    return Hkin + Hmat
