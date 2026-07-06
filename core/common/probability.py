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
receives an already-computed evolution operator.

The functions are organized as follows:

    probability_transition(...)
        Converts an evolutor into a full probability matrix or extracts a
        selected transition or survival channel.

    probability_coherent_state(...)
        Converts a flavour-amplitude state into component probabilities.

    probability_coherent(...)
        Applies an evolutor to a state and projects the result to
        probabilities.

    probability_incoherent(...)
        Applies a probability matrix to incoherent weights, or projects
        mass-basis weights through a flavour-basis evolutor and PMNS matrix.

    probability_from_evolutor(...)
        Converts a flavour-basis evolutor and either a coherent flavour state
        or incoherent mass weights into final flavour probabilities.

    normalize_probability_columns(...)
        Normalizes each probability column when explicitly requested.
        
    check_probability_conservation(...)
        Checks whether probability is conserved column-wise.

    check_probability_matrix(...)
        Performs basic numerical checks on a probability matrix.
"""



from __future__ import annotations

from typing import Union
import torch

from tpeanuts.core.common.evolutor import apply_evolutor_to_state
from tpeanuts.core.common.neutrino import flavour_index
from tpeanuts.util.type import real_dtype_from_tensor

def probability_transition(
    S_or_P: torch.Tensor,
    alpha: Union[str, int, None] = None,
    beta: Union[str, int, None] = None,
    *,
    input_is_probability: bool = False,
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a probability matrix or select one flavour transition.

    Uses P[beta, alpha] = |S[beta, alpha]|^2. With no flavour arguments the
    complete matrix is returned. Supplying only alpha selects its survival
    probability; supplying alpha and beta selects P(alpha -> beta).

    Args:
        S_or_P: Evolution operator or probability matrix shaped (..., 3, 3).
        alpha: Optional initial flavour label or index.
        beta: Optional final flavour label or index. Defaults to alpha when
            alpha is provided.
        input_is_probability: Interpret S_or_P directly as probabilities.
        real_dtype: Optional real dtype used when converting an evolutor.

    Returns:
        Full probability matrix or selected transition tensor.

    Raises:
        ValueError: If beta is provided without alpha.
    """
    if input_is_probability:
        P = S_or_P
        if real_dtype is not None:
            P = P.to(dtype=real_dtype)
    else:
        if real_dtype is None:
            real_dtype = real_dtype_from_tensor(S_or_P)
        P = (torch.abs(S_or_P) ** 2).to(dtype=real_dtype)

    if alpha is None:
        if beta is not None:
            raise ValueError("beta cannot be provided without alpha.")
        return P

    beta = alpha if beta is None else beta
    return P[..., flavour_index(beta), flavour_index(alpha)]


def probability_coherent_state(
    state: torch.Tensor,
    *,
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Project flavour amplitudes into component probabilities.

    Args:
        state: Complex or real state amplitudes shaped (..., 3).
        real_dtype: Optional real dtype for the returned probabilities.

    Returns:
        Real tensor with component probabilities shaped (..., 3).
    """
    if state.shape[-1] != 3:
        raise ValueError("state must have last dimension equal to 3.")

    if real_dtype is None:
        real_dtype = real_dtype_from_tensor(state)

    return (torch.abs(state) ** 2).to(dtype=real_dtype)


def probability_coherent(
    evolutor: torch.Tensor,
    state: torch.Tensor,
    *,
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply an evolutor to a state and return final probabilities.

    Args:
        evolutor: Evolution operator shaped (..., 3, 3).
        state: Initial state amplitudes shaped (..., 3).
        real_dtype: Optional real dtype for the returned probabilities.

    Returns:
        Final flavour probabilities shaped (..., 3).
    """
    return probability_coherent_state(
        apply_evolutor_to_state(evolutor, state),
        real_dtype=real_dtype,
    )


def probability_incoherent(
    P_or_evolutor: torch.Tensor,
    weights_initial: torch.Tensor,
    *,
    pmns: object | torch.Tensor | None = None,
    antinu: Union[bool, torch.Tensor] = False,
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply transition probabilities to incoherent initial weights.

    With ``pmns=None``, ``P_or_evolutor`` is interpreted directly as a
    probability matrix ``P`` and the function computes

        out_b = sum_a P_{ba} w_a.

    With ``pmns`` provided, ``P_or_evolutor`` is interpreted as a flavour-basis
    evolutor ``S``. The function constructs mass-to-flavour amplitudes

        A_{alpha i} = (S U)_{alpha i},

    converts them to probabilities, and applies the incoherent mass weights

        P_alpha = sum_i |A_{alpha i}|^2 w_i.

    Args:
        P_or_evolutor: Probability matrix shaped ``(..., 3, 3)`` when
            ``pmns`` is None, otherwise flavour-basis evolution operator
            shaped ``(..., 3, 3)``.
        weights_initial: Initial weights shaped (..., 3). The final dimension
            labels the source basis represented by the columns of P, or mass
            eigenstates when ``pmns`` is provided.
        pmns: Optional PMNS object exposing ``pmns_matrix(antinu=...)`` or an
            explicit PMNS matrix. When provided, enables the mass-to-flavour
            construction.
        antinu: Bool or tensor selecting the antineutrino PMNS convention when
            ``pmns`` is an object.
        real_dtype: Optional real dtype for probability conversion and weights.

    Returns:
        Final incoherent probabilities or fluxes shaped (..., 3).
    """
    if weights_initial.shape[-1] != 3:
        raise ValueError("weights_initial must have last dimension equal to 3.")

    if pmns is None:
        probabilities = P_or_evolutor
        if real_dtype is None:
            real_dtype = probabilities.dtype
        if real_dtype is not None:
            probabilities = probabilities.to(dtype=real_dtype)
    else:
        evolutor = P_or_evolutor
        if real_dtype is None:
            real_dtype = real_dtype_from_tensor(evolutor)

        if torch.is_tensor(pmns):
            U = pmns.to(device=evolutor.device, dtype=evolutor.dtype)
        else:
            U = pmns.pmns_matrix(antinu=antinu).to(
                device=evolutor.device,
                dtype=evolutor.dtype,
            )

        amplitudes = evolutor @ U
        probabilities = probability_transition(amplitudes, real_dtype=real_dtype)

    weights = weights_initial.to(device=probabilities.device, dtype=probabilities.dtype)
    target_ndim = probabilities.ndim - 1
    while weights.ndim < target_ndim:
        weights = weights.unsqueeze(-2)

    return torch.einsum("...ba,...a->...b", probabilities, weights)


def probability_from_evolutor(
    evolutor: torch.Tensor,
    state: torch.Tensor,
    *,
    pmns: object | torch.Tensor | None = None,
    massbasis: bool = False,
    antinu: Union[bool, torch.Tensor] = False,
    real_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert a flavour-basis evolutor and initial state to probabilities.

    Args:
        evolutor: Flavour-basis evolution operator shaped ``(..., 3, 3)``.
        state: Initial state with final dimension 3. When ``massbasis=False``,
            this is interpreted as coherent flavour-basis amplitudes. When
            ``massbasis=True``, this is interpreted as incoherent mass-basis
            weights.
        pmns: PMNS object or explicit PMNS matrix. Required when
            ``massbasis=True`` to project mass weights through ``evolutor @ U``.
        massbasis: Selects the interpretation of ``state``.
        antinu: Bool or tensor selecting antineutrino PMNS convention when
            ``pmns`` is an object.
        real_dtype: Optional real dtype for returned probabilities.

    Returns:
        Final flavour probabilities shaped ``(..., 3)``.

    Raises:
        ValueError: If ``massbasis=True`` and ``pmns`` is omitted.
    """
    if massbasis:
        if pmns is None:
            raise ValueError("pmns is required when massbasis=True.")

        return probability_incoherent(
            evolutor,
            state,
            pmns=pmns,
            antinu=antinu,
            real_dtype=real_dtype,
        )

    return probability_coherent(
        evolutor,
        state,
        real_dtype=real_dtype,
    )



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

###############################################################################
###############################################################################
######      
######      VALIDATE FUNCTIONS
######      
###############################################################################
###############################################################################
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
    column_sums = P.sum(dim=-2)
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
