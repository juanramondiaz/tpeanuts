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
First-order perturbative correction utilities for peanuts-torch.

This module contains the functions required to compute the first-order density
perturbation correction used in the peanuts segment evolutor.

The segment density is assumed to be polynomial:

    n_e(x) = a + b x^2 + c x^4.

The Hamiltonian is first constructed using the average density over the segment.
The remaining position-dependent part of the density is treated perturbatively.

The final segment evolution operator is

    U = u0 + u1,

where:

    u0
        Constant-density evolution operator built from the average density.

    u1
        First-order correction due to the density variation inside the segment.

The functions are organized as follows:

    has_density_perturbation(...)
        Checks whether the density profile has non-constant terms.

    polynomial_density_delta_constant(...)
        Computes the constant part of the perturbation around the average
        density.

    first_order_integral_Iab(...)
        Calls the peanuts analytical integral I_ab.

    first_order_correction_from_projectors(...)
        Builds u1 from the spectral projectors and perturbative integrals.

    first_order_density_correction(...)
        Computes the full first-order correction u1.

    perturbative_segment_evolutor(...)
        High-level function returning U = u0 + u1.

This module does not construct the Hamiltonian. It assumes that H, L, the
polynomial coefficients, and the average density have already been computed.
"""



from __future__ import annotations

from typing import Union
import torch

from tpeanuts.core.potential import matter_potential
from tpeanuts.core.spectral import hamiltonian_spectral_data
from tpeanuts.core.evolution import (
    constant_density_evolutor_from_spectral,
    enforce_identity_for_zero_length,
)
from tpeanuts.core.integration import Iab


TensorLike = Union[float, int, torch.Tensor]


def has_density_perturbation(
    b: torch.Tensor,
    c: torch.Tensor,
) -> bool:
    """
    Return whether polynomial coefficients contain a non-constant density perturbation.
    
    Args:
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
    
    Returns:
        Python bool indicating whether any perturbative density coefficient is non-zero.
    """
    mask = (torch.abs(b) > 0) | (torch.abs(c) > 0)

    return bool(mask.any().item())


def density_perturbation_mask(
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Build a boolean mask selecting segments with non-zero quadratic or quartic density terms.
    
    Args:
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
    
    Returns:
        Boolean tensor with True where b or c is non-zero.
    """
    return (torch.abs(b) > 0) | (torch.abs(c) > 0)


def polynomial_density_delta_constant(
    a: torch.Tensor,
    naverage: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the constant part of delta n_e(x)=n_e(x)-<n_e> for a polynomial segment.
    
    Args:
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        naverage: Segment-averaged electron density <n_e> in mol/cm^3.
    
    Returns:
        Tensor atilde = a - naverage in mol/cm^3.
    """
    return a - naverage


def first_order_integral_Iab(
    la: torch.Tensor,
    lb: torch.Tensor,
    atilde: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    x2: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the first-order integral matrix associated with polynomial density perturbations.
    
    Args:
        la: Initial eigenvalue or eigenvalue batch in km^-1.
        lb: Final eigenvalue or eigenvalue batch in km^-1.
        atilde: Constant perturbation coefficient after subtracting the segment average density.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
    
    Returns:
        Complex matrix tensor of first-order integrals shaped (..., 3, 3).
    """
    return Iab(
        la,
        lb,
        atilde.to(dtype=la.dtype),
        b.to(dtype=la.dtype),
        c.to(dtype=la.dtype),
        x2.to(dtype=la.dtype),
        x1.to(dtype=la.dtype),
    )


def first_order_correction_from_projectors(
    M: torch.Tensor,
    Vcorr: torch.Tensor,
) -> torch.Tensor:
    """
    Contract spectral projectors with perturbation integrals to obtain the first-order correction.
    
    Args:
        M: Spectral projector tensor shaped (..., 3, 3, 3), with the eigenvalue index on the third-from-last axis.
        Vcorr: Matrix of first-order perturbation integrals in the instantaneous eigenbasis.
    
    Returns:
        Complex first-order correction tensor shaped (..., 3, 3).
    """
    Ma_i0 = M[..., :, :, 0]
    Mb_0j = M[..., :, 0, :]

    u1 = (-1j) * torch.einsum(
        "...ab,...ai,...bj->...ij",
        Vcorr,
        Ma_i0,
        Mb_0j,
    )

    return u1


def first_order_density_correction(
    M: torch.Tensor,
    lam: torch.Tensor,
    trace_H: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    naverage: torch.Tensor,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """
    Compute the first-order perturbative correction induced by polynomial density variations.
    
    Args:
        M: Spectral projector tensor shaped (..., 3, 3, 3), with the eigenvalue index on the third-from-last axis.
        lam: Hamiltonian eigenvalues shaped (..., 3).
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        naverage: Segment-averaged electron density <n_e> in mol/cm^3.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Complex first-order correction tensor shaped (..., 3, 3).
    """
    atilde = polynomial_density_delta_constant(
        a,
        naverage,
    )

    eig_H = lam + trace_H[..., None] / 3.0

    la_g = eig_H[..., :, None]
    lb_g = eig_H[..., None, :]

    Iab_integral = first_order_integral_Iab(
        la=la_g,
        lb=lb_g,
        atilde=atilde,
        b=b,
        c=c,
        x2=x2,
        x1=x1,
    )

    Vcorr = matter_potential(
        Iab_integral,
        antinu=antinu,
    ).to(dtype=M.dtype)

    u1 = first_order_correction_from_projectors(
        M,
        Vcorr,
    )

    return u1


def perturbative_segment_evolutor(
    H: torch.Tensor,
    L: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    naverage: torch.Tensor,
    *,
    trace_H: torch.Tensor | None = None,
    zero_mask: torch.Tensor | None = None,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """
    Combine the exact constant-density evolutor with the first-order density correction for one segment.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
        L: Segment length in km; scalar or tensor broadcastable with the Hamiltonian batch shape.
        x1: Initial segment coordinate in Earth-radius units or polynomial coordinate.
        x2: Final segment coordinate in Earth-radius units or polynomial coordinate.
        a: Constant coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        b: Quadratic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        c: Quartic coefficient of n_e(x)=a+b x^2+c x^4 in mol/cm^3.
        naverage: Segment-averaged electron density <n_e> in mol/cm^3.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
        zero_mask: Boolean tensor selecting zero-length segments that must return the identity.
        antinu: Bool or boolean tensor; True selects antineutrino sign or complex-conjugated mixing.
    
    Returns:
        Complex segment evolutor tensor shaped (..., 3, 3).
    """
    spectral = hamiltonian_spectral_data(
        H,
        trace_H=trace_H,
    )

    u0 = constant_density_evolutor_from_spectral(
        lam=spectral["lam"],
        M=spectral["M"],
        trace_H=spectral["trace"],
        L=L,
    )

    u1 = first_order_density_correction(
        M=spectral["M"],
        lam=spectral["lam"],
        trace_H=spectral["trace"],
        x1=x1,
        x2=x2,
        a=a,
        b=b,
        c=c,
        naverage=naverage,
        antinu=antinu,
    )

    perturbation_mask = density_perturbation_mask(b, c)
    while perturbation_mask.ndim < u1.ndim - 2:
        perturbation_mask = perturbation_mask.unsqueeze(-1)

    U = u0 + torch.where(
        perturbation_mask[..., None, None],
        u1,
        torch.zeros_like(u1),
    )

    if zero_mask is not None:
        U = enforce_identity_for_zero_length(
            U,
            L,
            zero_mask,
        )

    return U
