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
Evolution operator utilities for peanuts-torch.

This module contains the functions responsible for constructing neutrino
evolution operators from already-built Hamiltonians and their spectral
decomposition.

The peanuts perturbative formalism expresses the constant-density evolution
operator as

    U0(L) =
        sum_a exp[-i (lambda_a + Tr(H)/3) L] M_a,

where:

    lambda_a
        Eigenvalues of the traceless Hamiltonian T.

    M_a
        Spectral projectors associated with lambda_a.

    Tr(H)
        Trace of the full Hamiltonian.

The functions in this module operate only on already-built Hamiltonians or
already-computed spectral objects. They do not construct the Hamiltonian itself.

The functions are organized as follows:

    identity_evolutor_like(...)
        Builds a broadcasted 3x3 identity identity evolution operator,
        compatible with a baseline tensor, matching a Hamiltonian batch..

    enforce_identity_for_zero_length(...)
        Replaces the evolution operator by the identity matrix when L == 0.

    constant_density_evolutor_from_spectral(...)
        Builds the peanuts constant-density evolution operator exp(-iHL) 
        from spectral projectors and eigenvalues.

    constant_density_evolutor(...)
        High-level interface that computes the full spectral decomposition
        internally and returns the peanuts constant-density evolution operator.

    matrix_exp_evolutor(...)
        Computes the exact evolution operator using torch.matrix_exp.
        This is mainly useful for validation and testing.

This module only handles the zeroth-order constant-density evolution.
First-order perturbative corrections are implemented separately in
perturbation.py.

"""



from __future__ import annotations

import torch

from tpeanuts.core.spectral import hamiltonian_spectral_data


def identity_evolutor_like(
    L: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a batched 3x3 identity evolutor compatible with the shape, device, and dtype of a path length tensor.
    
    Args:
        L: Segment length in km; scalar or tensor broadcastable with the Hamiltonian batch shape.
        device: Optional torch device for newly created tensors.
        dtype: Real or complex torch dtype for newly created tensors.
    
    Returns:
        Identity evolutor tensor shaped L.shape + (3, 3).
    """
    I = torch.eye(3, device=device, dtype=dtype)

    if L.ndim == 0:
        return I

    return I.expand(*L.shape, 3, 3)


def enforce_identity_for_zero_length(
    U: torch.Tensor,
    L: torch.Tensor,
    zero_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replace evolutors on zero-length segments by the identity operator.
    
    Args:
        U: Evolution-operator tensor shaped (..., 3, 3).
        L: Segment length in km; scalar or tensor broadcastable with the Hamiltonian batch shape.
        zero_mask: Boolean tensor selecting zero-length segments that must return the identity.
    
    Returns:
        Evolutor tensor with identity matrices inserted where zero_mask is True.
    """
    I = identity_evolutor_like(
        zero_mask,
        device=U.device,
        dtype=U.dtype,
    )

    return torch.where(
        zero_mask[..., None, None],
        I,
        U,
    )


def constant_density_evolutor_from_spectral(
    lam: torch.Tensor,
    M: torch.Tensor,
    trace_H: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the exact constant-density evolutor from spectral eigenvalues and projectors.
    
    Args:
        lam: Hamiltonian eigenvalues shaped (..., 3).
        M: Spectral projector tensor shaped (..., 3, 3, 3), with the eigenvalue index on the third-from-last axis.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
        L: Segment length in km; scalar or tensor broadcastable with the Hamiltonian batch shape.
    
    Returns:
        Complex evolutor tensor shaped (..., 3, 3).
    """
    Lc = L.to(dtype=lam.dtype)
    trace_H = trace_H.to(dtype=lam.dtype)

    phase = torch.exp(
        -1j * (lam + trace_H[..., None] / 3.0)
        * Lc[..., None]
    )

    u0 = (
        phase[..., :, None, None]
        * M
    ).sum(dim=-3)

    return u0


def constant_density_evolutor(
    H: torch.Tensor,
    L: torch.Tensor,
    *,
    trace_H: torch.Tensor | None = None,
    zero_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the exact constant-density evolution operator from a Hamiltonian and a segment length.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
        L: Segment length in km; scalar or tensor broadcastable with the Hamiltonian batch shape.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
        zero_mask: Boolean tensor selecting zero-length segments that must return the identity.
    
    Returns:
        Complex evolutor tensor shaped (..., 3, 3).
    """
    data = hamiltonian_spectral_data(
        H,
        trace_H=trace_H,
    )

    u0 = constant_density_evolutor_from_spectral(
        lam=data["lam"],
        M=data["M"],
        trace_H=data["trace"],
        L=L,
    )

    if zero_mask is not None:
        u0 = enforce_identity_for_zero_length(
            u0,
            L,
            zero_mask,
        )

    return u0


def matrix_exp_evolutor(
    H: torch.Tensor,
    L: torch.Tensor,
    *,
    zero_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the constant-density evolution operator with a direct matrix exponential.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
        L: Segment length in km; scalar or tensor broadcastable with the Hamiltonian batch shape.
        zero_mask: Boolean tensor selecting zero-length segments that must return the identity.
    
    Returns:
        Complex evolutor tensor shaped (..., 3, 3).
    """
    U = torch.matrix_exp(
        -1j
        * H
        * L[..., None, None].to(dtype=H.dtype)
    )

    if zero_mask is not None:
        U = enforce_identity_for_zero_length(
            U,
            L,
            zero_mask,
        )

    return U
