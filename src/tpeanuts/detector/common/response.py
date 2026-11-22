#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""Energy-response and grid-redistribution utilities, and the detector response that bundles them.

The functions operate on tensors already placed on a device/dtype (no
``RuntimeContext`` parameter here -- there is always an already-built
sibling tensor, ``T_grid_MeV``/``grid_MeV``, to infer device/dtype from,
the same convention ``tpeanuts.medium.earth.density.
interpolate_prem_density`` already follows).

Module contents:
    gaussian_response_matrix(...)
        Evaluate a Gaussian response density on true and reconstructed
        grids.
    scatter_add_linear(...)
        Deposit off-grid values onto neighboring grid points using linear
        weights.
    DetectorResponse
        Frozen dataclass: response matrix, efficiency and analysis bins of one detector,
        and ``fold`` -- a true-observable spectrum in, expected counts per bin out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tpeanuts.detector.common.efficiency import apply_efficiency
from tpeanuts.detector.common.event_rate import apply_response, bin_counts
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor


def gaussian_response_matrix(
    T_grid_MeV: torch.Tensor,
    Tprime_grid_MeV: torch.Tensor,
    sigma_MeV: TensorLike,
) -> torch.Tensor:
    """Build a Gaussian energy-response matrix R(T'|T).

    ``R(T'|T) = Normal(T'; mean=T, std=sigma(T))``, including the analytic
    ``1/(sqrt(2*pi)*sigma)`` normalization. Does not renormalize a
    truncated reconstructed-energy grid: probability outside that grid
    represents events outside the modeled window.

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        sigma_MeV: Energy resolution (standard deviation), positive.
            Either a scalar (constant resolution) or a tensor shaped
            ``(n_T,)`` (energy-dependent resolution, one value per
            ``T_grid_MeV`` entry).

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
    normalization = torch.sqrt(torch.as_tensor(2.0 * torch.pi, device=sigma.device, dtype=sigma.dtype))
    return torch.exp(-0.5 * (delta / sigma) ** 2) / (normalization * sigma)


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
    outside ``grid_MeV``'s range are clamped to the nearest edge point
    (all of their mass deposited there), matching ``gaussian_response_
    matrix``'s own "no renormalization of a truncated grid" convention.

    Args:
        grid_MeV: Strictly increasing target grid, shape ``(n_grid,)``.
        positions_MeV: Off-grid positions, any shape ``(...,)``.
        values: Mass at each position, same shape as ``positions_MeV``.

    Returns:
        Real tensor shaped ``(*positions_MeV.shape[:-1], n_grid)``:
        ``values`` summed onto ``grid_MeV`` along ``positions_MeV``'s
        last axis, preserving all leading (batch) axes.
    """
    n_grid = grid_MeV.shape[0]
    idx_hi = searchsorted_contiguous(grid_MeV, positions_MeV).clamp(min=1, max=n_grid - 1)
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


@dataclass(frozen=True)
class DetectorResponse:
    """What turns a spectrum of a true observable into the counts of an analysis: response, efficiency, bins.

    Bundles the three detector-specific pieces of the fold ``apply_response`` -> ``apply_efficiency`` ->
    ``bin_counts`` so that a detector holds one object instead of three loose tensors. The true and the reconstructed
    observable share one grid, ``T_grid_MeV`` (the same assumption ``predicted_counts`` makes).

    Attributes:
        T_grid_MeV: Grid of the true observable (e.g. the electron recoil energy), shaped ``(n_T,)``, also the grid of
            the reconstructed observable.
        response_matrix: ``R(T'|T)``, shaped ``(n_T, n_T)`` (see ``gaussian_response_matrix``).
        efficiency: Detection efficiency on the reconstructed grid, in ``[0, 1]``, shaped ``(n_T,)``.
        bin_edges_MeV: Strictly increasing edges of the analysis bins, shaped ``(n_bins + 1,)``, inside the grid.

    Raises:
        ValueError: If the response matrix or the efficiency do not match ``T_grid_MeV``.
    """

    T_grid_MeV: torch.Tensor
    response_matrix: torch.Tensor
    efficiency: torch.Tensor
    bin_edges_MeV: torch.Tensor

    def __post_init__(self) -> None:
        n_T = self.T_grid_MeV.shape[0]
        if self.response_matrix.shape != (n_T, n_T):
            raise ValueError(
                f"response_matrix must be shaped ({n_T}, {n_T}) (reconstructed and true grids are both T_grid_MeV), "
                f"got {tuple(self.response_matrix.shape)}."
            )
        if self.efficiency.shape != (n_T,):
            raise ValueError(f"efficiency must be shaped ({n_T},), got {tuple(self.efficiency.shape)}.")

    def fold(self, true_spectrum: torch.Tensor, exposure: TensorLike) -> torch.Tensor:
        """Expected counts per analysis bin from a true-observable spectrum.

        Args:
            true_spectrum: ``dR/dT`` on ``T_grid_MeV``, shaped ``(..., n_T)``, in events per second per MeV.
            exposure: Live time (seconds), or any factor that converts the rate into counts.

        Returns:
            Counts per bin, shaped ``(..., n_bins)``; no background added.
        """
        reco = apply_response(true_spectrum, self.T_grid_MeV, self.response_matrix)
        detected = apply_efficiency(reco, self.efficiency)
        return bin_counts(detected, self.T_grid_MeV, self.bin_edges_MeV, exposure)
