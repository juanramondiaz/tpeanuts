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

"""SNO salt-phase (Phase II) energy resolution and response matrix.

Ported from v1_0's ``experiments/sno_ii/response.py``:
``sigma_T(Te) = C0 + C1*sqrt(Te) + C2*Te`` [MeV, Te in MeV] -- Eq. A3
of the primary source
(``tpeanuts.experiments.sno_ii.parameters.ENERGY_RESOLUTION_C0/C1/C2``).
Distinct from, and not to be conflated with,
``tpeanuts.experiments.sno.response``'s Phase-I coefficients -- the two
salt-phase analyses (this 391-day spectral paper vs. the earlier
flux-only 2003 paper) even use two slightly different resolution
parametrizations of their own; this module implements the one that
matches the 17-bin CC spectrum and covariance tables this package's
data was transcribed from, not the flux-only paper's.

The formula goes non-positive below Te ~= 0.12 MeV (well under the 5.5
MeV analysis threshold); ``energy_resolution_MeV`` floors it at
``tpeanuts.experiments.sno_ii.parameters.ENERGY_RESOLUTION_FLOOR_MEV``
purely to keep ``gaussian_response_matrix``'s positivity requirement
satisfied on ``T_GRID_MEV``'s low-T tail, not as a physical statement
about resolution near threshold -- the same reasoning as
``tpeanuts.experiments.sno.response``'s own floor.

Module contents:
    energy_resolution_MeV(...)
        sigma(T) on a given true-energy grid.
    response_matrix(...)
        The salt-phase Gaussian response matrix, built via
        ``tpeanuts.detector.common.response.gaussian_response_matrix``
        -- no new generic response-matrix machinery is needed, only
        the new coefficients above.
"""

from __future__ import annotations

import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.experiments.sno_ii.parameters import (
    ENERGY_RESOLUTION_C0,
    ENERGY_RESOLUTION_C1,
    ENERGY_RESOLUTION_C2,
    ENERGY_RESOLUTION_FLOOR_MEV,
)

Tensor = torch.Tensor


def energy_resolution_MeV(
    T_grid_MeV: Tensor,
    *,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    floor_MeV: float = ENERGY_RESOLUTION_FLOOR_MEV,
) -> Tensor:
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
    T_grid_MeV: Tensor,
    Tprime_grid_MeV: Tensor,
    *,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    floor_MeV: float = ENERGY_RESOLUTION_FLOOR_MEV,
) -> Tensor:
    """SNO salt-phase Gaussian response matrix R(T'|T).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        c0, c1, c2, floor_MeV: See ``energy_resolution_MeV``.

    Returns:
        Real tensor shaped ``(n_Tp, n_T)``.
    """
    sigma = energy_resolution_MeV(T_grid_MeV, c0=c0, c1=c1, c2=c2, floor_MeV=floor_MeV)
    return gaussian_response_matrix(T_grid_MeV, Tprime_grid_MeV, sigma)
