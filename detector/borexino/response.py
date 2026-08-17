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
Borexino's energy resolution and response matrix.

sigma(T)/T = ENERGY_RESOLUTION_A / sqrt(T[MeV]) (see
``detector.borexino.parameters.ENERGY_RESOLUTION_A`` for the value and its
"illustrative, unverified against the primary paper" caveat) -- the standard
functional form for a photoelectron-counting liquid scintillator, where
photon-counting (Poisson) statistics make the fractional resolution scale as
1/sqrt(deposited energy).

Module contents:
    sigma_MeV(...)
        sigma(T) on a given true-energy grid.
    response_matrix(...)
        Borexino's Gaussian response matrix, built via
        ``detector.common.response.gaussian_response_matrix``.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.borexino.parameters import ENERGY_RESOLUTION_A, T_GRID_MEV, TPRIME_GRID_MEV
from tpeanuts.detector.common.response import gaussian_response_matrix


@torch.no_grad()
def sigma_MeV(T_grid_MeV: torch.Tensor, *, resolution_a: float = ENERGY_RESOLUTION_A) -> torch.Tensor:
    """Energy resolution sigma(T) = resolution_a * sqrt(T), MeV.

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        resolution_a: Resolution-curve normalization (see module docstring).

    Returns:
        Real tensor shaped ``(n_T,)``, sigma(T) in MeV.
    """
    return resolution_a * torch.sqrt(T_grid_MeV)


def response_matrix(
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    *,
    resolution_a: float = ENERGY_RESOLUTION_A,
) -> torch.Tensor:
    """Borexino's Gaussian response matrix R(T'|T).

    Args:
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``.
        resolution_a: Resolution-curve normalization (see ``sigma_MeV``).

    Returns:
        Real tensor shaped ``(n_Tp, n_T)``.
    """
    return gaussian_response_matrix(T_grid_MeV, Tprime_grid_MeV, sigma_MeV(T_grid_MeV, resolution_a=resolution_a))
