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
Hamiltonian construction utilities for peanuts-torch.

This module contains the low-level functions required to build the reduced
flavour-basis Hamiltonian used in the peanuts perturbative evolution scheme.

The Hamiltonian has the form

    H = H_kin + H_mat

with

    H_kin = U_red diag(k_i) U_red^T

and

    H_mat = diag(V, 0, 0).

Here U_red is the reduced mixing matrix R13 R12, k_i are the kinetic
eigenvalues in the mass basis, and V is the charged-current matter potential.

The functions are organized as follows:

    kinetic_mass_vector(...)
        Builds the kinetic eigenvalues k_i from the mass splittings 
        (Delta m^2 values) and neutrino energy.

    average_polynomial_density(...)
        Computes the average electron density along a segment and convert 
        them to matter potential assuming
        n_e(x) = a + b x^2 + c x^4.

    matter_potential_from_polynomial_average(...)
        Converts the average electron density into a matter potential.

    reduced_mixing_matrix(...)
        Extracts the reduced mixing matrix from the PMNS-like object.

    kinetic_hamiltonian_reduced(...)
        Builds H_kin = U_red diag(k_i) U_red^T.

    matter_hamiltonian_reduced(...)
        Builds H_mat = diag(V, 0, 0).

    reduced_hamiltonian(...)
        Builds the reduced Hamiltonian from a given matter potential V.
        Assemble H = H_kin + H_mat from matter potential

    reduced_hamiltonian_from_polynomial_density(...)
        Builds the reduced Hamiltonian directly from the polynomial density
        coefficients of a matter segment.

This module should not depend on any specific physical environment such as
the Sun, earth, atmosphere, or detector. It only constructs Hamiltonian objects
that can later be used by the evolution and perturbation modules.
"""



from __future__ import annotations

from typing import Union, Optional
import torch

from tpeanuts.util.type import _as_tensor, _cdtype_from_real
from tpeanuts.util.torch_util import _default_device
from tpeanuts.core.potential import kinetic_potential, matter_potential

TensorLike = Union[float, int, torch.Tensor]


def _infer_device_dtype(
    *values: TensorLike,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.device, torch.dtype]:
    """
    Infer the torch device and dtype to use from optional tensor inputs and explicit overrides.
    
    Args:
        *values: Optional scalar or tensor values used to infer device and dtype.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Tuple (device, dtype) selected for tensor construction.
    """
    for value in values:
        if torch.is_tensor(value):
            return (
                value.device if device is None else torch.device(device),
                value.dtype if dtype is None else dtype,
            )

    return (
        _default_device(device),
        torch.float64 if dtype is None else dtype,
    )


def _select_antinu_matrix(
    matrix: torch.Tensor,
    antinu: Union[bool, torch.Tensor],
) -> torch.Tensor:
    """
    Select the neutrino matrix or its complex conjugate according to an antineutrino mask.
    
    Args:
        matrix: Matrix or matrix batch shaped (..., 3, 3).
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Matrix tensor with neutrino or antineutrino convention applied.
    """
    if isinstance(antinu, bool):
        return torch.conj(matrix) if antinu else matrix

    antinu = antinu.to(device=matrix.device, dtype=torch.bool)
    while antinu.ndim < matrix.ndim - 2:
        antinu = antinu.unsqueeze(-1)

    return torch.where(
        antinu[..., None, None],
        torch.conj(matrix),
        matrix,
    )


def kinetic_mass_vector(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    E_MeV: TensorLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build the three kinetic eigenvalues k_i = m_i^2/(2E) in the mass basis.
    
    Formula: Uses k_i = m_i^2 / (2 E) with unit conversion to km^-1.
    
    Args:
        DeltamSq21: Solar mass splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass splitting Delta m^2_3l in eV^2; sign selects normal or inverted ordering.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Tensor shaped (..., 3) containing k_i in km^-1.
    """
    device, dtype = _infer_device_dtype(
        E_MeV,
        device=device,
        dtype=dtype,
    )
    dm21 = _as_tensor(DeltamSq21, device=device, dtype=dtype)
    dm3l = _as_tensor(DeltamSq3l, device=device, dtype=dtype)
    E_MeV = _as_tensor(E_MeV, device=device, dtype=dtype)

    ki_vec = torch.where(
        dm3l > 0,
        torch.stack(
            [
                torch.zeros_like(dm21),
                dm21,
                dm3l,
            ],
            dim=-1,
        ),
        torch.stack(
            [
                -dm21,
                torch.zeros_like(dm21),
                dm3l,
            ],
            dim=-1,
        ),
    )

    ki = kinetic_potential(ki_vec, E_MeV)

    return ki


def average_polynomial_density(
    x1: TensorLike,
    x2: TensorLike,
    a: TensorLike,
    b: TensorLike,
    c: TensorLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Average a polynomial electron-density profile n_e(x)=a+b x^2+c x^4 over a segment.
    
    Formula: Uses <n_e> = (x2-x1)^-1 int_x1^x2 (a+b x^2+c x^4) dx.
    
    Args:
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Tuple (n_average, L, zero_mask) with average density, segment length, and zero-length mask.
    """
    device, dtype = _infer_device_dtype(
        x1,
        x2,
        a,
        b,
        c,
        device=device,
        dtype=dtype,
    )

    x1 = _as_tensor(x1, device=device, dtype=dtype)
    x2 = _as_tensor(x2, device=device, dtype=dtype)
    a = _as_tensor(a, device=device, dtype=dtype)
    b = _as_tensor(b, device=device, dtype=dtype)
    c = _as_tensor(c, device=device, dtype=dtype)

    L = x2 - x1
    zero_mask = L == 0

    L_safe = torch.where(zero_mask, torch.ones_like(L), L)

    numerator = (
        a * L
        + b * (x2**3 - x1**3) / 3.0
        + c * (x2**5 - x1**5) / 5.0
    )

    naverage = numerator / L_safe
    naverage = torch.where(zero_mask, torch.zeros_like(naverage), naverage)

    return naverage, L, zero_mask


def matter_potential_from_polynomial_average(
    x1: TensorLike,
    x2: TensorLike,
    a: TensorLike,
    b: TensorLike,
    c: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert the segment-averaged polynomial electron density into the charged-current matter potential.
    
    Args:
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Matter potential tensor V in km^-1.
    """
    naverage, L, zero_mask = average_polynomial_density(
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        device=device,
        dtype=dtype,
    )

    V = matter_potential(naverage, antinu=antinu).to(dtype=naverage.dtype)
    V = torch.where(zero_mask, torch.zeros_like(V), V)

    return V, naverage, L, zero_mask


def reduced_mixing_matrix(
    pmns: object,
    *,
    antinu: Union[bool, torch.Tensor] = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Return the reduced PMNS mixing matrix U_red=R13 R12 used in the reduced Hamiltonian.
    
    Args:
        pmns: PMNS object exposing full and reduced mixing matrices plus R23 and Delta builders.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Reduced mixing matrix tensor U_red shaped (..., 3, 3).
    """
    Ured = getattr(pmns, "U", None)

    if Ured is None:
        raise AttributeError("pmns must provide reduced mixing matrix pmns.U = R13 R12.")

    Ured = Ured.to(device=device, dtype=dtype)

    return _select_antinu_matrix(Ured, antinu)


def kinetic_hamiltonian_reduced(
    ki: torch.Tensor,
    Ured: torch.Tensor,
) -> torch.Tensor:
    """
    Build the reduced kinetic Hamiltonian H_kin = U_red diag(k_i) U_red^T.
    
    Formula: Uses H_kin = U_red diag(k_i) U_red^T.
    
    Args:
        ki: Kinetic eigenvalue vector shaped (..., 3) in km^-1.
        Ured: Reduced PMNS matrix U_red shaped (..., 3, 3) or (3, 3).
    
    Returns:
        Complex kinetic Hamiltonian tensor shaped (..., 3, 3).
    """
    cdtype = Ured.dtype
    Ured = Ured.to(device=ki.device, dtype=cdtype)

    Dki = torch.diag_embed(ki.to(dtype=cdtype))

    Hkin = Ured @ Dki @ Ured.transpose(-1, -2)

    return Hkin


def matter_hamiltonian_reduced(
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Build the reduced matter Hamiltonian H_mat = diag(V, 0, 0).
    
    Formula: Uses H_mat = diag(V, 0, 0).
    
    Args:
        V: Charged-current matter potential in km^-1; scalar or tensor broadcastable with the batch shape.
    
    Returns:
        Complex matter Hamiltonian tensor shaped (..., 3, 3).
    """
    cdtype = torch.complex128 if V.dtype == torch.float64 else torch.complex64

    zeros = torch.zeros_like(V)

    Hmat = torch.diag_embed(
        torch.stack([V, zeros, zeros], dim=-1).to(dtype=cdtype)
    )

    return Hmat


def reduced_hamiltonian(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E_MeV: TensorLike,
    V: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """
    Assemble the reduced Hamiltonian H = H_kin + H_mat for a fixed matter potential.
    
    Formula: Uses H = U_red diag(k_i) U_red^T + diag(V, 0, 0).
    
    Args:
        DeltamSq21: Solar mass splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass splitting Delta m^2_3l in eV^2; sign selects normal or inverted ordering.
        pmns: PMNS object exposing full and reduced mixing matrices plus R23 and Delta builders.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
        V: Charged-current matter potential in km^-1; scalar or tensor broadcastable with the batch shape.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Complex reduced Hamiltonian tensor shaped (..., 3, 3).
    """
    device, rdtype = _infer_device_dtype(E_MeV, V)

    cdtype = _cdtype_from_real(rdtype)

    E_MeV = _as_tensor(E_MeV, device=device, dtype=rdtype)
    V = _as_tensor(V, device=device, dtype=rdtype)

    ki = kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        device=device,
        dtype=rdtype,
    )

    Ured = reduced_mixing_matrix(
        pmns,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    Hkin = kinetic_hamiltonian_reduced(ki, Ured)
    Hmat = matter_hamiltonian_reduced(V)

    H = Hkin + Hmat

    return H


def reduced_hamiltonian_from_polynomial_density(
    DeltamSq21: TensorLike,
    DeltamSq3l: TensorLike,
    pmns: object,
    E_MeV: TensorLike,
    x1: TensorLike,
    x2: TensorLike,
    a: TensorLike,
    b: TensorLike,
    c: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble the reduced Hamiltonian using the average density of a polynomial matter segment.
    
    Args:
        DeltamSq21: Solar mass splitting Delta m^2_21 in eV^2.
        DeltamSq3l: Atmospheric mass splitting Delta m^2_3l in eV^2; sign selects normal or inverted ordering.
        pmns: PMNS object exposing full and reduced mixing matrices plus R23 and Delta builders.
        E_MeV: Neutrino energy in MeV; scalar or tensor broadcastable with other inputs.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Tuple (H, V, n_average, L, zero_mask) for the polynomial-density segment.
    """
    device, rdtype = _infer_device_dtype(E_MeV, x1, x2, a, b, c)

    cdtype = _cdtype_from_real(rdtype)

    E_MeV = _as_tensor(E_MeV, device=device, dtype=rdtype)

    ki = kinetic_mass_vector(
        DeltamSq21=DeltamSq21,
        DeltamSq3l=DeltamSq3l,
        E_MeV=E_MeV,
        device=device,
        dtype=rdtype,
    )

    V, naverage, L, zero_mask = matter_potential_from_polynomial_average(
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        antinu=antinu,
        device=device,
        dtype=rdtype,
    )

    Ured = reduced_mixing_matrix(
        pmns,
        antinu=antinu,
        device=device,
        dtype=cdtype,
    )

    Hkin = kinetic_hamiltonian_reduced(ki, Ured)
    Hmat = matter_hamiltonian_reduced(V)

    H = Hkin + Hmat

    return H, ki, V, L, zero_mask
