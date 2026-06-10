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
Vacuum neutrino oscillation probabilities in pure PyTorch.

This module implements three-flavour propagation in the vacuum limit. No matter
potential is included: mass eigenstates acquire only kinetic phases and are then
rotated back to the flavour basis with the PMNS matrix. The routines here are
used as reference calculations for Earth, solar, atmospheric, and benchmark
workflows.

Functions
---------
vacuum_evolutor(...)
    Build the flavour-basis evolution operator S(L, E) in vacuum.
vacuum_probability_matrix(...)
    Return transition probabilities |S_ab|^2 for all flavour pairs.
vacuum_evolved_state(...)
    Apply the vacuum evolution operator to an initial flavour state.
pvacuum(...)
    Return final flavour probabilities for either flavour-basis amplitudes or
    mass-basis incoherent weights.
Pvacuum
    Backwards-compatible alias for pvacuum.

Private helpers
---------------
_pmns_matrix(...)
    Resolve the PMNS matrix from supported PMNS container objects and apply the
    antineutrino convention.
_broadcast_last3(...)
    Broadcast state or weight vectors whose final dimension contains the three
    flavours/mass components.
"""



from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.core.hamiltonian import kinetic_mass_vector, _select_antinu_matrix
from tpeanuts.util.constant import R_E
from tpeanuts.util.torch_util import _default_device, _resolve_dtype
from tpeanuts.util.type import _as_tensor, _cdtype_from_real, _state_tensor

TensorLike = Union[float, int, torch.Tensor]


def _pmns_matrix(
    pmns: object,
    *,
    antinu: Union[bool, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Resolve a PMNS mixing matrix on the requested device and dtype.

    Args:
        pmns: Object exposing either a pmns_matrix() method or a pmns tensor
            attribute.
        antinu: Boolean or tensor flag selecting the antineutrino convention.
        device: Target torch device.
        dtype: Real floating dtype used to choose complex64 or complex128.

    Returns:
        Complex PMNS matrix, conjugated as needed for antineutrino propagation.

    Raises:
        AttributeError: If pmns does not provide a supported matrix interface.
    """
    cdtype = _cdtype_from_real(dtype)

    if hasattr(pmns, "pmns_matrix"):
        U = pmns.pmns_matrix()
    elif hasattr(pmns, "pmns"):
        U = pmns.pmns
    else:
        raise AttributeError("pmns must provide pmns.pmns_matrix() or pmns.pmns.")

    U = U.to(device=device, dtype=cdtype)
    return _select_antinu_matrix(U, antinu)


def _broadcast_last3(vector: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
    """
    Broadcast a vector whose last dimension contains three components.

    Args:
        vector: Tensor with final dimension equal to three.
        batch_shape: Desired leading broadcast shape.

    Returns:
        Tensor with shape (*batch_shape, 3).

    Raises:
        ValueError: If vector does not have final dimension equal to three.
    """
    if vector.shape[-1] != 3:
        raise ValueError("Input vector must have last dimension equal to 3.")

    if len(batch_shape) == 0:
        return vector

    if vector.ndim == 1:
        return vector.expand(*batch_shape, 3)

    return torch.broadcast_to(vector, (*batch_shape, 3))


@torch.no_grad()
def vacuum_evolutor(
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build the vacuum flavour-basis evolution operator.

    The operator is constructed as U diag(exp(-i k_i L/R_E)) U^\dagger, where
    k_i are the kinetic mass-basis terms returned by kinetic_mass_vector. E and
    L may be scalar or tensor-valued and are broadcast with the kinetic vector.

    Args:
        pmns: PMNS object exposing pmns_matrix() or pmns.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        antinu: If True, use the antineutrino PMNS convention.
        device: Optional torch device. Defaults to CUDA when available, else CPU.
        dtype: Optional real dtype for the calculation. If omitted, inferred
            from E_MeV or L_km tensors, otherwise float64.

    Returns:
        Complex tensor with shape (..., 3, 3), where leading dimensions follow
        the broadcast shape of energy, baseline, and mass splittings.
    """
    device = _default_device(device)
    dtype = _resolve_dtype(dtype, E_MeV, L_km)
    cdtype = _cdtype_from_real(dtype)

    E = _as_tensor(E_MeV, device=device, dtype=dtype)
    L = _as_tensor(L_km, device=device, dtype=dtype)
    x = L * 1.0e3 / R_E

    ki = kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E,
        device=device,
        dtype=dtype,
    )

    batch_shape = torch.broadcast_shapes(ki.shape[:-1], x.shape)
    ki = torch.broadcast_to(ki, (*batch_shape, 3))
    x = torch.broadcast_to(x, batch_shape)

    phase = torch.exp(-1j * ki.to(dtype=cdtype) * x.unsqueeze(-1).to(dtype=cdtype))
    Dphase = torch.diag_embed(phase)

    U = _pmns_matrix(pmns, antinu=antinu, device=device, dtype=dtype)
    Udag = U.conj().transpose(-2, -1)

    return U @ Dphase @ Udag


@torch.no_grad()
def vacuum_probability_matrix(
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Compute all vacuum flavour-transition probabilities.

    Args:
        pmns: PMNS object exposing pmns_matrix() or pmns.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        antinu: If True, use the antineutrino PMNS convention.
        device: Optional torch device.
        dtype: Optional real dtype for the calculation.

    Returns:
        Real tensor |S_ab|^2 with shape (..., 3, 3). The final two dimensions
        are final flavour and initial flavour.
    """
    S = vacuum_evolutor(
        pmns,
        DeltamSq21,
        DeltamSq3l,
        E_MeV,
        L_km,
        antinu=antinu,
        device=device,
        dtype=dtype,
    )

    return torch.abs(S) ** 2


@torch.no_grad()
def vacuum_evolved_state(
    nustate: TensorLike,
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Evolve one initial flavour-basis state through vacuum.

    Args:
        nustate: Initial flavour amplitudes with final dimension 3. Leading
            dimensions may be broadcast against the evolution operator.
        pmns: PMNS object exposing pmns_matrix() or pmns.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        antinu: If True, use the antineutrino PMNS convention.
        device: Optional torch device.
        dtype: Optional real dtype for the calculation.

    Returns:
        Complex evolved flavour amplitudes with final dimension 3.
    """
    device = _default_device(device)
    dtype = _resolve_dtype(dtype, E_MeV, L_km)
    cdtype = _cdtype_from_real(dtype)

    S = vacuum_evolutor(
        pmns,
        DeltamSq21,
        DeltamSq3l,
        E_MeV,
        L_km,
        antinu=antinu,
        device=device,
        dtype=dtype,
    )

    state = _state_tensor(nustate, device=device, dtype=cdtype)
    state = _broadcast_last3(state, S.shape[:-2])

    return torch.einsum("...ab,...b->...a", S, state)


@torch.no_grad()
def pvacuum(
    nustate: TensorLike,
    pmns: object,
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    L_km: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    massbasis: bool = True,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Compute final vacuum flavour probabilities for an initial state.

    Args:
        nustate: Initial state. When massbasis=False, this is a flavour-basis
            amplitude vector with final dimension 3. When massbasis=True, this
            is interpreted as incoherent mass-basis weights.
        pmns: PMNS object exposing pmns_matrix() or pmns.
        DeltamSq21: Solar mass-squared splitting in eV^2.
        DeltamSq3l: Atmospheric mass-squared splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        antinu: If True, use the antineutrino PMNS convention.
        massbasis: Selects the interpretation of nustate. True returns the
            incoherent mass-basis probability mixture; False evolves flavour
            amplitudes coherently and squares them.
        device: Optional torch device.
        dtype: Optional real dtype for the calculation.

    Returns:
        Real tensor of final flavour probabilities with final dimension 3.
    """
    device = _default_device(device)
    dtype = _resolve_dtype(dtype, E_MeV, L_km)

    S = vacuum_evolutor(
        pmns,
        DeltamSq21,
        DeltamSq3l,
        E_MeV,
        L_km,
        antinu=antinu,
        device=device,
        dtype=dtype,
    )

    if not massbasis:
        psi = vacuum_evolved_state(
            nustate,
            pmns,
            DeltamSq21,
            DeltamSq3l,
            E_MeV,
            L_km,
            antinu=antinu,
            device=device,
            dtype=dtype,
        )
        return torch.abs(psi) ** 2

    weights = _state_tensor(nustate, device=device, dtype=dtype)
    weights = _broadcast_last3(weights, S.shape[:-2])

    U = _pmns_matrix(pmns, antinu=antinu, device=device, dtype=dtype)
    flavour_from_mass = S @ U
    P_flavour_from_mass = torch.abs(flavour_from_mass) ** 2

    return torch.einsum("...ai,...i->...a", P_flavour_from_mass, weights)


Pvacuum = pvacuum
