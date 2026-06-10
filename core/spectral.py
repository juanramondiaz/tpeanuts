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
Spectral decomposition utilities for peanuts-torch Hamiltonians.

This module contains the spectral tools required by the peanuts perturbative
evolution scheme. The Hamiltonian is first decomposed into a trace part and a
traceless part,

    H = T + Tr(H) / 3 I,

where T is a traceless 3x3 matrix.

The peanuts evolution method uses the eigenvalues of T and the associated
spectral projectors M_a. These projectors allow the constant-density evolution
operator to be written as

    U0(L) = sum_a exp[-i (lambda_a + Tr(H)/3) L] M_a.

The functions are organized as follows:

    identity3_like(...)
        Builds a 3x3 identity matrix on the same device and dtype as H.

    hamiltonian_trace_from_ki_and_V(...)
        Computes Tr(H) directly from the kinetic eigenvalues and matter
        potential.

    traceless_hamiltonian(...)
        Splits H into trace and traceless components.

    hermitize(...)
        Enforces numerical Hermiticity by symmetrizing H.

    traceless_invariant_c1(...)
        Computes the c1 invariant of the traceless Hamiltonian.

    traceless_eigenvalues(...)
        Computes the eigenvalues of the traceless Hamiltonian.

    spectral_projectors_traceless(...)
        Builds the spectral projectors M_a from T and its eigenvalues.

    hamiltonian_spectral_data(...)
        Computes and returns all spectral quantities required by the evolution
        module.

This module does not construct the Hamiltonian itself. It only receives an
already-built Hamiltonian and prepares the spectral objects needed for evolution.

"""



from __future__ import annotations

import torch


def identity3_like(H: torch.Tensor) -> torch.Tensor:
    """
    Build a 3x3 identity matrix broadcastable to the batch shape of a Hamiltonian.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
    
    Returns:
        Identity matrix tensor shaped H.shape[:-2] + (3, 3).
    """
    return torch.eye(3, device=H.device, dtype=H.dtype)


def hamiltonian_trace_from_ki_and_V(
    ki: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the trace of the reduced Hamiltonian from kinetic eigenvalues and matter potential.
    
    Args:
        ki: Kinetic eigenvalue vector shaped (..., 3) in km^-1.
        V: Charged-current matter potential in km^-1; scalar or tensor broadcastable with the batch shape.
    
    Returns:
        Trace tensor shaped like the broadcast of ki[..., 0] and V.
    """
    return (ki[..., 0] + ki[..., 1] + ki[..., 2] + V).to(
        dtype=torch.complex128 if ki.dtype == torch.float64 else torch.complex64
    )


def traceless_hamiltonian(
    H: torch.Tensor,
    trace_H: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subtract one third of the trace from a Hamiltonian to obtain its traceless part.
    
    Formula: Uses T = H - tr(H) I / 3.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
    
    Returns:
        Traceless Hamiltonian tensor shaped (..., 3, 3).
    """
    I3 = identity3_like(H)

    if trace_H is None:
        trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    trace_H = trace_H.to(dtype=H.dtype)

    T = H - trace_H[..., None, None] * I3 / 3.0

    return T, trace_H


def hermitize(H: torch.Tensor) -> torch.Tensor:
    """
    Symmetrize a matrix with its Hermitian conjugate to suppress numerical anti-Hermitian noise.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
    
    Returns:
        Hermitian matrix tensor shaped like H.
    """
    return 0.5 * (H + H.conj().transpose(-1, -2))


def traceless_invariant_c1(T: torch.Tensor) -> torch.Tensor:
    """
    Compute the quadratic invariant c1 = tr(T^2)/2 of a traceless 3x3 Hamiltonian.
    
    Formula: Uses c1 = tr(T^2) / 2.
    
    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
    
    Returns:
        Real tensor c1 with the batch shape of T.
    """
    T2 = T @ T
    trT2 = torch.diagonal(T2, dim1=-2, dim2=-1).sum(dim=-1)

    return -trT2 / 2.0


def traceless_eigenvalues(T: torch.Tensor) -> torch.Tensor:
    """
    Compute the eigenvalues of a traceless Hermitian Hamiltonian.
    
    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
    
    Returns:
        Real tensor shaped (..., 3) with eigenvalues of T.
    """
    T = T.contiguous()
    T = hermitize(T)

    if not torch.isfinite(T).all():
        raise FloatingPointError("T contains NaN or Inf before eigvalsh.")

    lam = torch.linalg.eigvalsh(T).to(dtype=T.dtype)

    return lam


def spectral_projectors_traceless(
    T: torch.Tensor,
    lam: torch.Tensor | None = None,
    c1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build spectral projectors for a traceless Hamiltonian from eigenvalues and c1.
    
    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        lam: Hamiltonian eigenvalues shaped (..., 3).
        c1: Quadratic invariant c1 of the traceless Hamiltonian.
    
    Returns:
        Complex projector tensor shaped (..., 3, 3, 3).
    """
    I3 = identity3_like(T)

    if lam is None:
        lam = traceless_eigenvalues(T)

    if c1 is None:
        c1 = traceless_invariant_c1(T)

    T2 = T @ T

    denom = 3.0 * lam**2 + c1[..., None]

    M = (
        (lam**2 + c1[..., None])[..., :, None, None] * I3
        + lam[..., :, None, None] * T[..., None, :, :]
        + T2[..., None, :, :]
    ) / denom[..., :, None, None]

    return M, lam, c1


def hamiltonian_spectral_data(
    H: torch.Tensor,
    trace_H: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Return trace, traceless Hamiltonian, eigenvalues, and spectral projectors for H.
    
    Args:
        H: Hamiltonian tensor shaped (..., 3, 3) in km^-1.
        trace_H: Trace of the Hamiltonian shaped (...) and expressed in km^-1.
    
    Returns:
        Tuple (trace_H, T, lam, M) containing trace, traceless Hamiltonian, eigenvalues, and projectors.
    """
    T, trace_H = traceless_hamiltonian(H, trace_H=trace_H)
    T = hermitize(T)

    lam = traceless_eigenvalues(T)
    c1 = traceless_invariant_c1(T)
    M, lam, c1 = spectral_projectors_traceless(T, lam=lam, c1=c1)

    return {
        "T": T,
        "trace": trace_H,
        "lam": lam,
        "c1": c1,
        "M": M,
    }