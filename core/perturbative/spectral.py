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
Hamiltonian utilities for the Tpeanuts perturbative evolution scheme.

This module contains the spectral tools required by the peanuts perturbative
evolution scheme. The Hamiltonian is first decomposed into a trace part and a
traceless part,

    H = T + Tr(H) / 3 I,

where T is a traceless 3x3 matrix.

The peanuts evolution method uses the eigenvalues of T and the associated
spectral projectors M_a. These projectors allow the constant-density evolution
operator to be written as

    U0(L) = sum_a exp[-i (lambda_a + Tr(H)/3) L] M_a.

The module functions are organized as follows:

    hamiltonian_traceless(...)
        Splits H into trace and traceless components.

    hamiltonian_traceless_c0(...)
        Computes the cubic invariant c0 of the traceless Hamiltonian.

    hamiltonian_traceless_c1(...)
        Computes the c1 invariant of the traceless Hamiltonian.

    hamiltonian_traceless_eigenvalues(...)
        Computes the eigenvalues of the traceless Hamiltonian.

    hamiltonian_spectral_projectors_traceless(...)
        Builds the spectral projectors M_a from T and its eigenvalues.

    hamiltonian_spectral_data(...)
        Computes and returns all spectral quantities required by the evolution
        module.

This module receives an already-built Hamiltonian and prepares the spectral
objects needed for evolution.

"""



from __future__ import annotations

import torch

# Minimum absolute value of the spectral projector denominator (3λ² + c1).
# Guards against division by zero when two eigenvalues are nearly degenerate
# (e.g. vacuum propagation or Δm²₂₁ → 0).
_DENOM_EPS: float = 1.0e-30


def hamiltonian_traceless(
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
        Tuple containing the traceless Hamiltonian shaped (..., 3, 3) and its
        trace with the batch shape of H.
    """
    I3 = torch.eye(3, device=H.device, dtype=H.dtype)

    if trace_H is None:
        trace_H = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)

    trace_H = trace_H.to(dtype=H.dtype)

    T = H - trace_H[..., None, None] * I3 / 3.0

    return T, trace_H

def hamiltonian_traceless_c0(T: torch.Tensor) -> torch.Tensor:
    """
    Compute the cubic invariant c0 of a traceless 3x3 Hamiltonian.

    Formula: Uses c0 = -Tr(T^3) / 3. Together with the quadratic invariant
    c1 (see ``hamiltonian_traceless_c1``), c0 enters the characteristic
    polynomial of T, ``lambda^3 - c1*lambda - c0 = 0`` (no quadratic term
    since T is traceless), whose three real roots are the eigenvalues
    returned by ``hamiltonian_traceless_eigenvalues``.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).

    Returns:
        Real tensor c0 with the batch shape of T.
    """

    T3 = T @ T @ T

    trT3 = torch.diagonal(
        T3,
        dim1=-2,
        dim2=-1
    ).sum(dim=-1)

    return -trT3 / 3.0

def hamiltonian_traceless_c1(T: torch.Tensor, T2: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the quadratic invariant c1 of a traceless 3x3 Hamiltonian.

    Formula: Uses c1 = -tr(T^2) / 2.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        T2: Optional precomputed T @ T shaped (..., 3, 3). When provided the
            matrix multiplication is skipped, avoiding a redundant bmm.

    Returns:
        Real tensor c1 with the batch shape of T.
    """
    if T2 is None:
        T2 = T @ T
    trT2 = torch.diagonal(
        T2,
        dim1=-2,
        dim2=-1
    ).sum(dim=-1)

    return -trT2 / 2.0


def hamiltonian_traceless_eigenvalues(
    T: torch.Tensor,
    *,
    already_symmetric: bool = False,
) -> torch.Tensor:
    """
    Compute the eigenvalues of a traceless Hermitian Hamiltonian.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        already_symmetric: When True, skip the symmetrization step.  Set this
            flag when the caller (e.g. ``hamiltonian_spectral_data``) has
            already enforced Hermitian symmetry to avoid a redundant operation.

    Returns:
        Tensor shaped (..., 3) with the real eigenvalues represented in T.dtype.
    """
    T = T.contiguous()
    if not already_symmetric:
        T = 0.5 * (T + T.conj().transpose(-1, -2))

    if not torch.isfinite(T).all():
        raise FloatingPointError("T contains NaN or Inf before eigvalsh.")

    lam = torch.linalg.eigvalsh(T).to(dtype=T.dtype)

    return lam


def hamiltonian_spectral_projectors_traceless(
    T: torch.Tensor,
    lam: torch.Tensor | None = None,
    c1: torch.Tensor | None = None,
    T2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build spectral projectors for a traceless Hamiltonian from eigenvalues and c1.

    Args:
        T: Traceless Hermitian Hamiltonian tensor shaped (..., 3, 3).
        lam: Hamiltonian eigenvalues shaped (..., 3).
        c1: Quadratic invariant c1 of the traceless Hamiltonian.
        T2: Optional precomputed T @ T shaped (..., 3, 3). When provided the
            matrix multiplication is skipped, avoiding a redundant bmm.

    Returns:
        Tuple containing projectors shaped (..., 3, 3, 3), eigenvalues shaped
        (..., 3), and c1 with the batch shape of T.
    """
    I3 = torch.eye(3, device=T.device, dtype=T.dtype)

    if T2 is None:
        T2 = T @ T

    if lam is None:
        lam = hamiltonian_traceless_eigenvalues(T, already_symmetric=True)

    if c1 is None:
        c1 = hamiltonian_traceless_c1(T, T2=T2)

    denom = 3.0 * lam**2 + c1[..., None]

    # Guard against near-degenerate eigenvalues (e.g. vacuum or Δm²₂₁ → 0).
    safe_denom = torch.where(
        denom.abs() < _DENOM_EPS,
        denom.new_full((), _DENOM_EPS),
        denom,
    )

    M = (
        (lam**2 + c1[..., None])[..., :, None, None] * I3
        + lam[..., :, None, None] * T[..., None, :, :]
        + T2[..., None, :, :]
    ) / safe_denom[..., :, None, None]

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
        Dictionary containing the traceless Hamiltonian, trace, eigenvalues,
        c1 invariant, and spectral projectors.

    Notes:
        T @ T is computed once here and forwarded to both ``hamiltonian_traceless_c1``
        and ``hamiltonian_spectral_projectors_traceless`` to avoid redundant bmm.
        The symmetrization of T is also done once; downstream helpers receive
        ``already_symmetric=True`` so they skip the redundant transpose.
    """
    T, trace_H = hamiltonian_traceless(H, trace_H=trace_H)
    # Enforce Hermitian symmetry once; pass the flag to avoid a second transpose.
    T = 0.5 * (T + T.conj().transpose(-1, -2))

    # Compute T² once and reuse for both c1 and the spectral projectors.
    T2 = T @ T

    lam = hamiltonian_traceless_eigenvalues(T, already_symmetric=True)
    c1 = hamiltonian_traceless_c1(T, T2=T2)
    M, lam, c1 = hamiltonian_spectral_projectors_traceless(T, lam=lam, c1=c1, T2=T2)

    return {
        "T": T,
        "trace": trace_H,
        "lam": lam,
        "c1": c1,
        "M": M,
    }


