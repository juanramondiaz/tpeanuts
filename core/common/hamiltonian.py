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

"""Standard-Model three-flavour Hamiltonian construction utilities.

This module contains only the 3-flavour Standard Model Hamiltonian used by
the common, numerical, and perturbative propagation layers.

Module functions:
    kinetic_mass_squared_vector(...)
        Build the reduced three-component mass-squared vector.
    kinetic_mass_vector(...)
        Convert the mass-squared vector into dimensionless kinetic phases.
    hamiltonian_kinetic_reduced(...)
        Build ``H_kin = U_red diag(k_i) U_red^T``.
    hamiltonian_matter_reduced(...)
        Build the SM matter Hamiltonian ``diag(V, 0, 0)``.
    hamiltonian_reduced(...)
        Build the reduced 3-flavour SM Hamiltonian from electron density.
    hamiltonian_flavour(...)
        Transform the reduced SM Hamiltonian to the flavour basis.
"""

from __future__ import annotations

from typing import Optional

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.potential import kinetic_potential, matter_potential
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import infer_device_dtype
from tpeanuts.util.type import TensorLike, as_tensor, cdtype_from_real


def kinetic_mass_squared_vector(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Build the reduced three-component mass-squared vector.

    Args:
        DeltamSq21: Solar mass splitting in eV^2.
        DeltamSq3l: Atmospheric mass splitting in eV^2. Its sign selects the
            normal or inverted ordering convention.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.

    Returns:
        Tensor shaped ``(..., 3)`` with the common phase removed.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(DeltamSq21, DeltamSq3l)
    dm21 = as_tensor(DeltamSq21, device=device, dtype=dtype)
    dm3l = as_tensor(DeltamSq3l, device=device, dtype=dtype)
    zeros = torch.zeros_like(dm21)

    if dm3l.ndim == 0 and dm3l.device.type == "cpu":
        if dm3l.item() > 0:
            return torch.stack([zeros, dm21, dm3l], dim=-1)
        return torch.stack([-dm21, zeros, dm3l], dim=-1)

    normal = torch.stack([zeros, dm21, dm3l], dim=-1)
    inverted = torch.stack([-dm21, zeros, dm3l], dim=-1)
    return torch.where((dm3l > 0)[..., None], normal, inverted)


@torch.no_grad()
def kinetic_mass_vector(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the three kinetic eigenvalues in the mass basis.

    Args:
        DeltamSq21: Solar mass splitting in eV^2.
        DeltamSq3l: Atmospheric mass splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from the tensor inputs.
        evolution_scale_m: Positive evolution length scale in metres.
        legacy_precision: Accepted for propagation-chain consistency. It is
            forwarded to ``kinetic_potential`` and does not alter the kinetic
            calculation.

    Returns:
        Dimensionless kinetic eigenvalues shaped ``(..., 3)``.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(E_MeV, DeltamSq21, DeltamSq3l, evolution_scale_m)
    resolved_context = RuntimeContext(device=device, dtype=dtype)
    E_MeV = as_tensor(E_MeV, device=device, dtype=dtype)
    ki_vec = kinetic_mass_squared_vector(
        DeltamSq21,
        DeltamSq3l,
        context=resolved_context,
    )

    return kinetic_potential(
        ki_vec,
        E_MeV,
        evolution_scale_m=evolution_scale_m,
        context=resolved_context,
        legacy_precision=legacy_precision,
    )


@torch.no_grad()
def hamiltonian_kinetic_reduced(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    Ured: torch.Tensor,
    *,
    evolution_scale_m: TensorLike = constant.R_E,
    return_ki: bool = False,
    legacy_precision: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Build the SM kinetic Hamiltonian in the reduced flavour basis.

    Args:
        DeltamSq21: Solar mass splitting in eV^2.
        DeltamSq3l: Atmospheric mass splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        Ured: Reduced PMNS matrix shaped ``(..., 3, 3)``.
        evolution_scale_m: Positive evolution length scale in metres.
        return_ki: If True, also return the kinetic eigenvalues.
        legacy_precision: Accepted for propagation-chain consistency and
            forwarded to ``kinetic_mass_vector``.

    Returns:
        Complex Hamiltonian shaped ``(..., 3, 3)`` or ``(Hkin, ki)``.
    """
    if Ured.shape[-2:] != (3, 3):
        raise ValueError("Ured must have final dimensions (3, 3).")

    device, dtype = infer_device_dtype(
        E_MeV,
        DeltamSq21,
        DeltamSq3l,
        evolution_scale_m,
        device=Ured.device,
        dtype=Ured.real.dtype,
    )
    ki = kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        context=RuntimeContext(device=device, dtype=dtype),
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    cdtype = Ured.dtype
    Ured = Ured.to(device=ki.device, dtype=cdtype)
    Hkin = (Ured * ki.to(dtype=cdtype)[..., None, :]) @ Ured.transpose(-1, -2)

    if return_ki:
        return Hkin, ki
    return Hkin


@torch.no_grad()
def hamiltonian_matter_reduced(
    V: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the SM matter Hamiltonian ``diag(V, 0, 0)``.

    Args:
        V: Dimensionless charged-current matter potential.
        context: Optional runtime device/dtype. When omitted, both are
            inferred from ``V``.
        legacy_precision: Accepted for API consistency. ``V`` must already
            contain the chosen matter-potential prefactor.

    Returns:
        Complex matter Hamiltonian shaped ``(..., 3, 3)``.
    """
    _ = legacy_precision
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = infer_device_dtype(V)
    V = as_tensor(V, device=device, dtype=dtype)
    cdtype = cdtype_from_real(dtype)

    Hmat = torch.zeros((*V.shape, 3, 3), device=device, dtype=cdtype)
    Hmat[..., 0, 0] = V.to(dtype=cdtype)
    return Hmat


@torch.no_grad()
def hamiltonian_reduced(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Assemble the reduced 3-flavour SM Hamiltonian from electron density.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron density in mol/cm^3.
        context: Optional runtime device/dtype. When omitted, both are
            taken from ``oscillation.pmns``.
        evolution_scale_m: Positive evolution length scale in metres.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor when building the matter Hamiltonian.

    Returns:
        Dimensionless reduced Hamiltonian shaped ``(..., 3, 3)``.
    """
    pmns = oscillation.pmns
    antinu = oscillation.antinu
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device, dtype = pmns.device, pmns.dtype

    Ured = pmns.reduced(antinu=antinu)
    if Ured.shape[-2:] != (3, 3):
        raise ValueError("SM hamiltonian_reduced requires a 3-flavour PMNS object.")

    V = matter_potential(
        n_e_mol_cm3,
        antinu=antinu,
        evolution_scale_m=evolution_scale_m,
        context=RuntimeContext(device=device, dtype=dtype),
        legacy_precision=legacy_precision,
    )
    Hkin = hamiltonian_kinetic_reduced(
        DeltamSq21=oscillation.DeltamSq21,
        DeltamSq3l=oscillation.DeltamSq3l,
        E_MeV=E_MeV,
        Ured=Ured,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )
    Hmat = hamiltonian_matter_reduced(
        V,
        context=RuntimeContext(device=Hkin.device, dtype=Hkin.real.dtype),
        legacy_precision=legacy_precision,
    )
    return Hkin + Hmat


@torch.no_grad()
def hamiltonian_flavour(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    n_e_mol_cm3: TensorLike,
    *,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = constant.R_E,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Build the full flavour-basis 3-flavour SM Hamiltonian.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        n_e_mol_cm3: Electron density in mol/cm^3.
        context: Optional runtime device/dtype. When omitted, both are
            taken from ``oscillation.pmns``.
        evolution_scale_m: Positive evolution length scale in metres.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in the reduced Hamiltonian.

    Returns:
        Dimensionless flavour-basis Hamiltonian shaped ``(..., 3, 3)``.
    """
    H_reduced = hamiltonian_reduced(
        oscillation,
        E_MeV,
        n_e_mol_cm3,
        context=context,
        evolution_scale_m=evolution_scale_m,
        legacy_precision=legacy_precision,
    )

    return oscillation.pmns.H_flavour_basis(
        H_reduced,
        antinu=oscillation.antinu,
        device=H_reduced.device,
        dtype=H_reduced.dtype,
    )
