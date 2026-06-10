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
Probability utilities for peanuts-torch.

This module converts neutrino evolution operators into oscillation
probabilities.

Given a flavour-basis evolution operator S, the transition probability is

    P(alpha -> beta) = |S[beta, alpha]|^2.

The convention used here is:

    - The initial flavour alpha labels the column index.
    - The final flavour beta labels the row index.
    - Therefore P[beta, alpha] = P(alpha -> beta).

This module does not build Hamiltonians and does not propagate states. It only
receives an already-computed evolution operator, usually produced by:

    segment_evolution.py
    evolution.py
    perturbation.py
    trajectory_evolution.py

The functions are organized as follows:

    flavour_index(...)
        Maps flavour names or integer labels to the canonical index order [e,
        mu, tau].

    probability_matrix_from_evolutor(...)
        Converts an evolution operator S into a full probability matrix.

    transition_probability(...)
        Extracts one transition probability P(alpha -> beta).

    survival_probability(...)
        Computes P(alpha -> alpha).

    apply_probability_matrix_to_flux(...)
        Applies P to an initial flavour flux vector.

    probability_columns_sum(...),
        Provide normalization diagnostics and corrections for probability
        matrices.
        
    check_probability_conservation(...)
        Checks whether probability is conserved column-wise.

    check_probability_matrix(...)
        Performs basic numerical checks on a probability matrix.
"""



from __future__ import annotations

from typing import Union
import torch

TensorLike = Union[float, int, torch.Tensor]


FLAVOUR_TO_INDEX = {
    "e": 0,
    "electron": 0,
    "nue": 0,
    "nu_e": 0,
    "mu": 1,
    "muon": 1,
    "numu": 1,
    "nu_mu": 1,
    "tau": 2,
    "nutau": 2,
    "nu_tau": 2,
}


def flavour_index(flavour: Union[str, int]) -> int:
    """
    Map a flavour label or integer to the canonical order e=0, mu=1, tau=2.
    
    Args:
        flavour: Flavour name or index; accepted names include e, electron, nue, mu, numu, tau, and nutau.
    
    Returns:
        Integer flavour index in the canonical order e=0, mu=1, tau=2.
    """
    if isinstance(flavour, int):
        if flavour not in (0, 1, 2):
            raise ValueError("Flavour index must be 0, 1, or 2.")
        return flavour

    key = flavour.lower()

    if key not in FLAVOUR_TO_INDEX:
        raise ValueError(f"Unknown flavour label: {flavour}")

    return FLAVOUR_TO_INDEX[key]


def probability_matrix_from_evolutor(
    S: torch.Tensor,
    *,
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Convert a flavour-basis evolution operator into transition probabilities P_beta_alpha=|S_beta_alpha|^2.
    
    Formula: Uses P[beta, alpha] = |S[beta, alpha]|^2.
    
    Args:
        S: Flavour-basis evolution operator shaped (..., 3, 3).
        real_dtype: Optional real dtype for the returned probability matrix; inferred from S when None.
    
    Returns:
        Real probability matrix tensor shaped (..., 3, 3).
    """
    if real_dtype is None:
        real_dtype = torch.float64 if S.dtype == torch.complex128 else torch.float32

    P = torch.abs(S) ** 2

    return P.to(dtype=real_dtype)


def transition_probability(
    S_or_P: torch.Tensor,
    alpha: Union[str, int],
    beta: Union[str, int],
    *,
    input_is_probability: bool = False,
) -> torch.Tensor:
    """
    Extract one transition probability P(alpha -> beta) from an evolutor or probability matrix.
    
    Args:
        S_or_P: Evolution operator S or probability matrix P shaped (..., 3, 3).
        alpha: Initial flavour label or index.
        beta: Final flavour label or index.
        input_is_probability: If True, S_or_P is already interpreted as a probability matrix.
    
    Returns:
        Real tensor containing P(alpha -> beta) over any leading batch dimensions.
    """
    alpha_idx = flavour_index(alpha)
    beta_idx = flavour_index(beta)

    if input_is_probability:
        P = S_or_P
    else:
        P = probability_matrix_from_evolutor(S_or_P)

    return P[..., beta_idx, alpha_idx]


def survival_probability(
    S_or_P: torch.Tensor,
    alpha: Union[str, int],
    *,
    input_is_probability: bool = False,
) -> torch.Tensor:
    """
    Extract the survival probability P(alpha -> alpha) from an evolutor or probability matrix.
    
    Args:
        S_or_P: Evolution operator S or probability matrix P shaped (..., 3, 3).
        alpha: Initial flavour label or index.
        input_is_probability: If True, S_or_P is already interpreted as a probability matrix.
    
    Returns:
        Real tensor containing P(alpha -> alpha) over any leading batch dimensions.
    """
    return transition_probability(
        S_or_P,
        alpha=alpha,
        beta=alpha,
        input_is_probability=input_is_probability,
    )


def apply_probability_matrix_to_flux(
    P: torch.Tensor,
    flux_initial: torch.Tensor,
) -> torch.Tensor:
    """
    Propagate an initial flavour-flux vector with a probability matrix.
    
    Args:
        P: Probability matrix shaped (..., 3, 3), with P[..., beta, alpha].
        flux_initial: Initial flavour flux vector shaped (..., 3), ordered as e, mu, tau.
    
    Returns:
        Final flavour flux vector shaped (..., 3).
    """
    if flux_initial.shape[-1] != 3:
        raise ValueError("flux_initial must have last dimension equal to 3.")

    return torch.einsum("...ba,...a->...b", P, flux_initial)


def probability_columns_sum(
    P: torch.Tensor,
) -> torch.Tensor:
    """
    Sum probability columns to diagnose conservation for each initial flavour.
    
    Args:
        P: Probability matrix shaped (..., 3, 3), with P[..., beta, alpha].
    
    Returns:
        Tensor shaped (..., 3) with sums over final flavours beta for each alpha.
    """
    return P.sum(dim=-2)


def check_probability_conservation(
    P: torch.Tensor,
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-6,
    raise_error: bool = False,
) -> bool:
    """
    Check column-wise probability conservation within an absolute tolerance.
    
    Args:
        P: Probability matrix shaped (..., 3, 3), with P[..., beta, alpha].
        atol: Absolute tolerance for comparing each column sum with one.
        rtol: Relative tolerance for comparing each column sum with one.
        raise_error: If True, raise ValueError when probability conservation fails.
    
    Returns:
        Boolean tensor or bool selecting matrices whose columns sum to one within tol.
    """
    column_sums = probability_columns_sum(P)
    target = torch.ones_like(column_sums)

    ok = bool(torch.allclose(column_sums, target, atol=atol, rtol=rtol))

    if raise_error and not ok:
        max_err = torch.max(torch.abs(column_sums - target)).item()
        raise ValueError(
            f"Probability conservation failed. "
            f"max|sum_beta P[beta, alpha] - 1| = {max_err:.4e}"
        )

    return ok


def check_probability_matrix(
    P: torch.Tensor,
    *,
    atol: float = 1.0e-8,
    rtol: float = 1.0e-6,
    raise_error: bool = False,
) -> bool:
    """
    Validate that a probability matrix is finite, non-negative, and column-normalized.
    
    Args:
        P: Probability matrix shaped (..., 3, 3), with P[..., beta, alpha].
        atol: Absolute tolerance for non-negativity and column-normalization checks.
        rtol: Relative tolerance for column-normalization checks.
        raise_error: If True, raise ValueError describing the first failed validation.
    
    Returns:
        Python bool indicating whether all validation checks pass.
    """
    finite_ok = bool(torch.isfinite(P).all())
    nonnegative_ok = bool((P >= -atol).all())
    conservation_ok = check_probability_conservation(
        P,
        atol=atol,
        rtol=rtol,
        raise_error=False,
    )

    ok = finite_ok and nonnegative_ok and conservation_ok

    if raise_error and not ok:
        if not finite_ok:
            raise ValueError("Probability matrix contains NaN or Inf.")

        if not nonnegative_ok:
            min_value = torch.min(P).item()
            raise ValueError(
                f"Probability matrix contains negative values. "
                f"Minimum value = {min_value:.4e}"
            )

        if not conservation_ok:
            check_probability_conservation(
                P,
                atol=atol,
                rtol=rtol,
                raise_error=True,
            )

    return ok


def normalize_probability_columns(
    P: torch.Tensor,
    *,
    eps: float = 1.0e-15,
) -> torch.Tensor:
    """
    Normalize each probability column so that transition probabilities sum to one.
    
    Args:
        P: Probability matrix shaped (..., 3, 3), with P[..., beta, alpha].
        eps: Minimum denominator used to avoid division by zero when a column sum vanishes.
    
    Returns:
        Column-normalized probability matrix with the same shape as P.
    """
    column_sums = P.sum(dim=-2, keepdim=True)

    return P / torch.clamp(column_sums, min=eps)
