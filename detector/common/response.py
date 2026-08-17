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
Gaussian energy-response (migration) matrix construction.

``gaussian_response_matrix`` is the generic response builder shared by
detectors whose energy resolution is well described by a (possibly
energy-dependent) Gaussian smearing of the true observable -- Borexino, SNO,
Super-K. A detector with a Monte-Carlo migration matrix instead (e.g.
IceCube's reconstructed-vs-true-energy matrix) supplies its own matrix
directly to ``tpeanuts.detector.common.event_rate.apply_response`` and does
not need this module.

Module contents:
    gaussian_response_matrix(...)
        Build R(T'|T) on a (Tprime_grid, T_grid) pair as a normalized
        Gaussian density on the full real line. A finite reconstructed
        grid may therefore contain less than unit probability.
    scatter_add_linear(...)
        Redistribute point masses onto a fixed grid by linear interpolation,
        conserving total mass. Shared low-level utility for any detector
        that needs to move probability/cross-section mass from an
        off-grid energy value onto its own fixed energy grid -- used by
        ``tpeanuts.detector.dayabay.response`` (LSNL energy-scale warp) and
        ``tpeanuts.detector.interaction.inverse_beta_decay`` (cos(theta) ->
        prompt-energy redistribution of the order-1/M differential cross
        section).
"""

from __future__ import annotations

import torch

from tpeanuts.util.type import TensorLike, as_tensor


@torch.no_grad()
def gaussian_response_matrix(
    T_grid_MeV: torch.Tensor,
    Tprime_grid_MeV: torch.Tensor,
    sigma_MeV: TensorLike,
) -> torch.Tensor:
    """Build a Gaussian energy-response matrix R(T'|T).

    R(T'|T) = Normal(T'; mean=T, std=sigma(T)), including the analytic
    ``1/(sqrt(2*pi)*sigma)`` normalization. The function does not
    renormalize a truncated reconstructed-energy grid: probability outside
    that grid represents events outside the modeled window.

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        sigma_MeV: Energy resolution (standard deviation), positive. Either
            a scalar (constant resolution) or a tensor shaped ``(n_T,)``
            (energy-dependent resolution, one value per ``T_grid_MeV``
            entry).

    Returns:
        Real tensor shaped ``(n_Tp, n_T)``; ``result[:, j]`` is the
        response density for true energy ``T_grid_MeV[j]``, ready for
        ``tpeanuts.detector.common.event_rate.apply_response``.

    Raises:
        ValueError: If ``sigma_MeV`` is not a scalar and does not have
            shape ``(n_T,)``, or if any entry is not positive.
    """
    sigma = as_tensor(sigma_MeV, device=T_grid_MeV.device, dtype=T_grid_MeV.dtype)
    if sigma.ndim > 0 and sigma.shape != T_grid_MeV.shape:
        raise ValueError(
            f"sigma_MeV must be a scalar or shaped {tuple(T_grid_MeV.shape)} "
            f"(matching T_grid_MeV), got shape {tuple(sigma.shape)}."
        )
    if torch.any(sigma <= 0):
        raise ValueError("sigma_MeV must be strictly positive.")

    delta = Tprime_grid_MeV[:, None] - T_grid_MeV[None, :]  # (n_Tp, n_T)
    return torch.exp(-0.5 * (delta / sigma) ** 2) / (
        torch.sqrt(torch.as_tensor(2.0 * torch.pi, device=sigma.device, dtype=sigma.dtype))
        * sigma
    )


@torch.no_grad()
def scatter_add_linear(
    grid_MeV: torch.Tensor,
    positions_MeV: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Redistribute ``values`` located at ``positions_MeV`` onto ``grid_MeV`` by linear interpolation.

    Each ``(position, value)`` pair is split between the two ``grid_MeV``
    points bracketing it, with weights linear in the position -- the
    standard mass-conserving "scatter" used to move an off-grid point mass
    onto a fixed grid without introducing spurious smoothing. Positions
    outside ``grid_MeV``'s range are clamped to the nearest edge point (all
    of their mass deposited there), matching this project's established
    "no renormalization of a truncated grid" convention (see
    ``gaussian_response_matrix``).

    Args:
        grid_MeV: Strictly increasing target grid, shape ``(n_grid,)``.
        positions_MeV: Off-grid positions, any shape ``(...,)``.
        values: Mass at each position, same shape as ``positions_MeV``.

    Returns:
        Real tensor shaped ``(*positions_MeV.shape[:-1], n_grid)`` if
        ``positions_MeV`` is at least 1-D and the redistribution is summed
        over its last axis; more precisely, ``values`` is summed onto
        ``grid_MeV`` along ``positions_MeV``'s last axis, preserving all
        leading (batch) axes.
    """
    n_grid = grid_MeV.shape[0]
    idx_hi = torch.searchsorted(grid_MeV, positions_MeV).clamp(min=1, max=n_grid - 1)
    idx_lo = idx_hi - 1

    x_lo = grid_MeV[idx_lo]
    x_hi = grid_MeV[idx_hi]
    weight_hi = ((positions_MeV - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)
    weight_lo = 1.0 - weight_hi

    out_shape = positions_MeV.shape[:-1] + (n_grid,)
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    out.scatter_add_(-1, idx_lo, values * weight_lo)
    out.scatter_add_(-1, idx_hi, values * weight_hi)
    return out
