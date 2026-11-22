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

"""SNO's real energy resolution and response matrix.

Ported from v1_0's ``experiments/sno/response.py``:
``sigma_T(Te) = c0 + c1*sqrt(Te) + c2*Te`` [MeV, Te in MeV] -- the
widely-quoted SNO Phase-I kinetic-energy resolution parametrization
(``experiments.sno.parameters.ENERGY_RESOLUTION_C0/C1/C2``), used
throughout the solar-neutrino oscillation-fit literature reproducing
SNO's response (e.g. sigma_T(10 MeV) ~= 1.40 MeV). Unlike
``experiments.borexino.response``'s single-parameter illustrative
sigma(T) = A*sqrt(T) form, this is real, not illustrative -- no
primary SNO instrumentation paper was independently consulted here to
re-derive the coefficients, so treat it as a real, widely-used
parametrization rather than one traced to a specific SNO publication.

It is *not* standing in for any unmodeled CC recoil spread -- since
``tpeanuts.detector.deuteron`` uses the real tabulated Nakamura et al.
(2002) electron spectrum, the 3-body (p, p, e-) recoil spread is
itself real input, and this module's sigma(T) only ever represents
genuine detector energy resolution (shared, via
``experiments.sno.event_rate.nc_event_rate``, by the NC channel's
neutron-capture-gamma response).

The formula goes non-positive below Te ~= 0.04 MeV (well under SNO's
actual analysis threshold); ``energy_resolution_MeV`` floors it at
``ENERGY_RESOLUTION_FLOOR_MEV`` purely to keep
``gaussian_response_matrix``'s positivity requirement satisfied on
``T_GRID_MEV``'s low-T tail, not as a physical statement about
resolution near threshold.

Module contents:
    energy_resolution_MeV(...)
        sigma(T) on a given true-energy grid.
    response_matrix(...)
        SNO's Gaussian response matrix, built via
        ``detector.common.response.gaussian_response_matrix``.
"""

from __future__ import annotations

import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.experiments.sno.parameters import (
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
    """SNO's Gaussian response matrix R(T'|T).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        c0, c1, c2, floor_MeV: See ``energy_resolution_MeV``.

    Returns:
        Real tensor shaped ``(n_Tp, n_T)``.
    """
    sigma = energy_resolution_MeV(T_grid_MeV, c0=c0, c1=c1, c2=c2, floor_MeV=floor_MeV)
    return gaussian_response_matrix(T_grid_MeV, Tprime_grid_MeV, sigma)
