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
#      August 2026
# =============================================================================

"""
SNO salt-phase (Phase II) energy resolution and response matrix.

sigma_T(Te) = C0 + C1*sqrt(Te) + C2*Te [MeV, Te in MeV] -- Eq. A3 of the
primary source (``tpeanuts.detector.sno_ii.parameters
.ENERGY_RESOLUTION_C0/C1/C2``), verified against the paper directly (see
``tpeanuts.detector.sno_ii``'s package docstring). Distinct from, and not
to be conflated with, ``tpeanuts.detector.sno.response``'s Phase-I
coefficients -- the two salt-phase analyses (this 391-day spectral paper
vs. the earlier flux-only 2003 paper) even use two slightly different
resolution parametrizations of their own; this module implements the one
that matches the 17-bin CC spectrum and covariance tables this package's
data was transcribed from, not the flux-only paper's.

The formula goes non-positive below Te ~= 0.12 MeV (well under the 5.5 MeV
analysis threshold); ``sigma_MeV`` floors it at
``tpeanuts.detector.sno_ii.parameters.ENERGY_RESOLUTION_FLOOR_MEV`` purely
to keep ``gaussian_response_matrix``'s positivity requirement satisfied on
``T_GRID_MEV``'s low-T tail, not as a physical statement about resolution
near threshold -- the same reasoning as
``tpeanuts.detector.sno.response``'s own floor.

Module contents:
    sigma_MeV(...)
        sigma(T) on a given true-energy grid.
    response_matrix(...)
        The salt-phase Gaussian response matrix, built via
        ``tpeanuts.detector.common.response.gaussian_response_matrix`` --
        no new generic response-matrix machinery is needed, only the new
        coefficients above.
"""

from __future__ import annotations

import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.detector.sno_ii.parameters import (
    ENERGY_RESOLUTION_C0,
    ENERGY_RESOLUTION_C1,
    ENERGY_RESOLUTION_C2,
    ENERGY_RESOLUTION_FLOOR_MEV,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)

__all__ = ["sigma_MeV", "response_matrix"]


@torch.no_grad()
def sigma_MeV(
    T_grid_MeV: torch.Tensor,
    *,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    floor_MeV: float = ENERGY_RESOLUTION_FLOOR_MEV,
) -> torch.Tensor:
    """Energy resolution sigma(T) = c0 + c1*sqrt(T) + c2*T, MeV (see module docstring).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        c0, c1, c2: Resolution-formula coefficients (see module docstring).
        floor_MeV: Numerical floor, not a physical parameter (see module
            docstring).

    Returns:
        Real tensor shaped ``(n_T,)``, sigma(T) in MeV.
    """
    T = T_grid_MeV.clamp_min(torch.finfo(T_grid_MeV.dtype).tiny)
    sigma = c0 + c1 * torch.sqrt(T) + c2 * T
    return sigma.clamp_min(floor_MeV)


def response_matrix(
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    *,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    floor_MeV: float = ENERGY_RESOLUTION_FLOOR_MEV,
) -> torch.Tensor:
    """SNO salt-phase Gaussian response matrix R(T'|T).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        c0, c1, c2, floor_MeV: See ``sigma_MeV``.

    Returns:
        Real tensor shaped ``(n_Tp, n_T)``.
    """
    sigma = sigma_MeV(T_grid_MeV, c0=c0, c1=c1, c2=c2, floor_MeV=floor_MeV)
    return gaussian_response_matrix(T_grid_MeV, Tprime_grid_MeV, sigma)
