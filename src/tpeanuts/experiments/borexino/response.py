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

"""Borexino's illustrative Gaussian energy-response chain.

Ported from v1_0's ``experiments/borexino/response.py``: a single
sqrt(T)-scaling resolution formula on top of
``tpeanuts.detector.common.response.gaussian_response_matrix`` --
unlike Daya Bay's own 3-term ``energy_resolution_MeV``/IAV+LSNL chain
(``tpeanuts.experiments.dayabay.response``), Borexino has no IAV/LSNL
correction in this project, only the illustrative Gaussian smearing
itself (see ``tpeanuts.experiments.borexino.parameters.ENERGY_RESOLUTION_A``'s
own caveat).

Module contents:
    energy_resolution_MeV(...)
        Illustrative sqrt(T)-scaling energy resolution.
    response_matrix(...)
        Gaussian response matrix on the true/reconstructed grids.
"""

from __future__ import annotations

from typing import cast

import torch

from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.experiments.borexino.parameters import ENERGY_RESOLUTION_A
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor


def energy_resolution_MeV(T_grid_MeV: Tensor, *, resolution_a: TensorLike = ENERGY_RESOLUTION_A) -> Tensor:
    """Illustrative Borexino energy resolution, ``a * sqrt(T)``.

    Args:
        T_grid_MeV: True electron-recoil grid, shape ``(n_T,)``.
        resolution_a: Resolution coefficient; see
            ``tpeanuts.experiments.borexino.parameters.ENERGY_RESOLUTION_A``.

    Returns:
        Standard deviation in MeV, shape ``(n_T,)``.
    """
    return cast(torch.Tensor, resolution_a * torch.sqrt(T_grid_MeV))


def response_matrix(
    T_grid_MeV: Tensor, Tprime_grid_MeV: Tensor, *, resolution_a: TensorLike = ENERGY_RESOLUTION_A,
) -> Tensor:
    """Gaussian response matrix R(T'|T) with Borexino's illustrative resolution.

    Args:
        T_grid_MeV: True electron-recoil grid, shape ``(n_T,)``.
        Tprime_grid_MeV: Reconstructed electron-recoil grid, shape
            ``(n_Tp,)``.
        resolution_a: Resolution coefficient, see ``energy_resolution_MeV``.

    Returns:
        Response density, shape ``(n_Tp, n_T)`` (see
        ``tpeanuts.detector.common.response.gaussian_response_matrix``).
    """
    sigma = energy_resolution_MeV(T_grid_MeV, resolution_a=resolution_a)
    return gaussian_response_matrix(T_grid_MeV, Tprime_grid_MeV, sigma)
