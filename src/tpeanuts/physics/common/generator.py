#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""The two base generator terms every scenario needs unconditionally.

This module builds the two elementary pieces of the propagation Hamiltonian
``H`` that generates unitary propagation (in natural units, hbar = c = 1),
from a ``MixingMatrix`` and a ``MassSpectrum``, generalized to any 3+N
sterile extension: nothing here is hardcoded to 3 or to 4 flavours, or
forked into a separate "SM" and "sterile" code path.

    - The kinetic term ``H_kin = U_red diag(k_i) U_red^dagger``,
     where  ``k_i`` are the dimensionless kinetic eigenvalues from
      ``tpeanuts.physics.common.potential.kinetic_potential``.
    - The charged-current matter term ``H_mat = V_cc * diag(1, 0, ..., 0)``
      in the reduced basis -- exactly invariant under the mixing matrix's
      outer block, so this fast path never needs to rotate into the flavour
      basis and back.

Neither this module nor any function here assembles them into a full
generator: ``tpeanuts.physics.model.PropagationModel`` is the composition
root that sums these two (wrapped in private adapters) with whatever
``tpeanuts.physics.extensions`` terms are active and, when
``space.n_helicity == 2``, the companion helicity block -- a combination
these two elementary functions alone cannot express. A previous version of
this module also exposed bare ``generator_reduced``/``generator_flavour``
convenience functions (kinetic + matter only, no extensions); they were
removed once it became clear nothing in the simulation/inference pipeline
actually called them (``PropagationModel`` reimplements its own, more
general assembly instead) -- they only ever served as test oracles, which
now build the same reference value directly from ``kinetic_term_reduced``/
``matter_term_reduced``.

Module contents:
    kinetic_term_reduced(...)
        Reduced-basis kinetic term ``U_red diag(k_i) U_red^dagger``.
    matter_term_reduced(...)
        Reduced-basis charged-current matter term ``V_cc diag(1,0,...,0)``.
    rotate_to_reduced(...)
        ``O^dagger @ flavour_matrix @ O``. Shared by every optional
        extension term (``tpeanuts.physics.extensions.sterile``, ``.nsi``,
        ...): unlike the charged-current fast path above, a term whose
        "direction" matrix does not commute with the outer block must be
        built in the (physically native) flavour basis and rotated down
        into the reduced basis this way before it can be summed alongside
        ``kinetic_term_reduced``/``matter_term_reduced`` and dressed by a
        single final flavour-basis rotation, instead of each term rotating
        itself into the flavour basis separately.
"""

from __future__ import annotations

from typing import Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.potential import kinetic_potential, matter_potential_cc
from tpeanuts.util.tensor import TensorLike, cdtype_from_real


def validate_consistency(mixing: MixingMatrix, mass_spectrum: MassSpectrum) -> None:
    """Raise if ``mixing`` and ``mass_spectrum`` disagree on space or context.

    Public (not module-private) because ``tpeanuts.physics.model`` also
    needs it, to validate the same pair at ``PropagationModel``
    construction time rather than only when a generator is actually built.

    Args:
        mixing: MixingMatrix to check.
        mass_spectrum: MassSpectrum to check.

    Raises:
        ValueError: If ``mixing.space != mass_spectrum.space`` or
            ``mixing.context != mass_spectrum.context``.
    """
    if mixing.space != mass_spectrum.space:
        raise ValueError(
            "mixing and mass_spectrum must share the same FlavourSpace, got "
            f"{mixing.space} and {mass_spectrum.space}."
        )
    if mixing.context != mass_spectrum.context:
        raise ValueError(
            "mixing and mass_spectrum must share the same RuntimeContext, got "
            f"{mixing.context} and {mass_spectrum.context}."
        )


def electron_projector(n: int, *, device: torch.device, cdtype: torch.dtype) -> torch.Tensor:
    """Build ``diag(1, 0, ..., 0)``, the electron-flavour projector.

    Public (not module-private) because ``tpeanuts.physics.extensions.nsi``
    also needs it, to build ``diag(1,0,...,0) + epsilon`` -- shared by more
    than one module, so it lives here once rather than being duplicated.

    Args:
        n: Matrix dimension (``mixing.space.n_flavour``).
        device: Device for the returned tensor.
        cdtype: Complex dtype for the returned tensor.

    Returns:
        Complex tensor shaped ``(n, n)``, 1 at ``(0, 0)``, 0 elsewhere.
    """
    P = torch.zeros(n, n, device=device, dtype=cdtype)
    P[0, 0] = 1.0
    return P


def kinetic_term_reduced(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    evolution_scale_m: TensorLike = constant.R_E,
) -> torch.Tensor:
    """Build the reduced-basis kinetic term ``U_red diag(k_i) U_red^dagger``.

    ``U_red = mixing.reduced(antinu)`` is unitary on its own (a product of
    unitary Givens rotations and phase factors, see
    ``tpeanuts.physics.common.mixing``), so this reproduces the same
    eigenvalues ``k_i`` regardless of ``n_sterile`` or ``nature`` -- neither
    changes anything about how this term is built, only what ``mixing``
    itself already encodes.

    Args:
        mixing: Built MixingMatrix.
        mass_spectrum: Built MassSpectrum, sharing ``mixing``'s
            FlavourSpace and RuntimeContext.
        E_MeV: Neutrino energy in MeV.
        antinu: Bool or boolean tensor selecting the antineutrino
            (complex-conjugated ``U_red``) convention.
        evolution_scale_m: Positive length scale in metres used to
            non-dimensionalize the kinetic term. Defaults to the Earth
            radius.

    Returns:
        Complex Hermitian tensor shaped
        ``(..., mixing.space.n_flavour, mixing.space.n_flavour)``.

    Raises:
        ValueError: If ``mixing``/``mass_spectrum`` disagree on space or
            context (see ``validate_consistency``).
    """
    validate_consistency(mixing, mass_spectrum)
    context = mixing.context

    Ured = mixing.reduced(antinu=antinu)
    ki = kinetic_potential(
        mass_spectrum.difference_vector(),
        E_MeV,
        evolution_scale_m=evolution_scale_m,
        context=context,
    )

    cdtype = Ured.dtype
    ki_c = ki.to(dtype=cdtype)
    return (Ured * ki_c[..., None, :]) @ Ured.conj().transpose(-2, -1)


def matter_term_reduced(
    mixing: MixingMatrix,
    n_e_mol_cm3: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    evolution_scale_m: TensorLike = constant.R_E,
) -> torch.Tensor:
    """Build the reduced-basis charged-current matter term ``V_cc diag(1,0,...,0)``.

    Args:
        mixing: Built MixingMatrix (only ``mixing.space``/``mixing.context``
            are read; the matter term itself does not depend on the mixing
            angles, since ``diag(1,0,...,0)`` is invariant under the outer
            block).
        n_e_mol_cm3: Electron density of the propagation medium, in
            mol/cm^3 (i.e. N_A * electrons/cm^3).
        antinu: Bool or boolean tensor selecting the antineutrino
            (sign-flipped) convention.
        evolution_scale_m: Positive length scale in metres, matching
            whatever was used to build the kinetic term. Defaults to the
            Earth radius.

    Returns:
        Complex Hermitian (in fact real-diagonal) tensor shaped
        ``(..., mixing.space.n_flavour, mixing.space.n_flavour)``.
    """
    context = mixing.context
    V_cc = matter_potential_cc(
        n_e_mol_cm3, antinu, evolution_scale_m=evolution_scale_m, context=context
    )

    n = mixing.space.n_flavour
    cdtype = cdtype_from_real(context.dtype)
    P_cc = electron_projector(n, device=context.device, cdtype=cdtype)

    return V_cc.to(dtype=cdtype)[..., None, None] * P_cc


def rotate_to_reduced(flavour_matrix: torch.Tensor, O: torch.Tensor) -> torch.Tensor:
    """Rotate a flavour-basis matrix into the reduced basis: ``O^dagger @ flavour_matrix @ O``.

    Args:
        flavour_matrix: Complex tensor shaped ``(..., n, n)``, built directly
            in the flavour basis (e.g. an NSI or sterile neutral-current
            "direction" matrix -- see the module docstring).
        O: Outer mixing block, e.g. ``mixing.outer_block(antinu=...)``,
            shaped ``(..., n, n)``.

    Returns:
        Complex tensor shaped ``(..., n, n)``, the reduced-basis
        counterpart of ``flavour_matrix``.
    """
    return O.conj().transpose(-2, -1) @ flavour_matrix @ O
