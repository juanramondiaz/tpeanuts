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

"""Tree-level Standard Model neutrino-electron elastic scattering.

nu + e^- -> nu + e^-, differential in the electron recoil kinetic energy T:

    dsigma/dT(E_nu, T) = (2 G_F^2 m_e / pi) * (hbar c)^2
                          * [g_L^2 + g_R^2 (1 - T/E_nu)^2 - g_L g_R m_e T / E_nu^2],

valid for 0 <= T <= T_max(E_nu) = 2 E_nu^2 / (m_e + 2 E_nu) (zero outside),
with couplings

    nu_e         (CC + NC): g_L = 1/2 + sin^2(theta_W), g_R = sin^2(theta_W)
    nu_mu, nu_tau (NC only): g_L = -1/2 + sin^2(theta_W), g_R = sin^2(theta_W)

Radiative corrections are not included.

Module contents:
    NUE_COUPLINGS, NUMUTAU_COUPLINGS
        (g_L, g_R) pairs for the electron-flavour and mu/tau-flavour cases.
    dsigma_dT(...)
        Calculate the differential cross section for specified couplings.
    nue_cross_section_grid(...)
        Evaluate the electron-neutrino cross section on an energy grid.
    numutau_cross_section_grid(...)
        Evaluate the muon/tau-neutrino cross section on an energy grid.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

import tpeanuts.util.constant as constant


class _Couplings(NamedTuple):
    g_L: float
    g_R: float


NUE_COUPLINGS = _Couplings(g_L=0.5 + constant.SIN2_THETA_W, g_R=constant.SIN2_THETA_W)
NUMUTAU_COUPLINGS = _Couplings(g_L=-0.5 + constant.SIN2_THETA_W, g_R=constant.SIN2_THETA_W)

# (hbar*c)^2, MeV^2 cm^2 -- converts G_F^2 * m_e [MeV^-3] into a cross
# section per unit energy [cm^2 / MeV].
_HBARC_MEV_CM2 = (constant.HBARC_MeV_m * 100.0) ** 2


def dsigma_dT(
    E_nu_MeV: torch.Tensor,
    T_MeV: torch.Tensor,
    g_L: float,
    g_R: float,
) -> torch.Tensor:
    """Differential nu-e elastic scattering cross section, dsigma/dT.

    Args:
        E_nu_MeV: Neutrino energy, any shape.
        T_MeV: Electron recoil kinetic energy, broadcastable with
            ``E_nu_MeV``.
        g_L: Left-handed coupling (``NUE_COUPLINGS.g_L`` or
            ``NUMUTAU_COUPLINGS.g_L``).
        g_R: Right-handed coupling (``NUE_COUPLINGS.g_R`` or
            ``NUMUTAU_COUPLINGS.g_R``).

    Returns:
        dsigma/dT in cm^2/MeV, broadcast shape of ``E_nu_MeV``/``T_MeV``;
        exactly 0 outside the kinematically allowed range
        ``0 <= T <= T_max(E_nu)``.
    """
    m_e = constant.M_ELECTRON_MEV
    prefactor = 2.0 * constant.G_F_MEV_M2 ** 2 * m_e / torch.pi * _HBARC_MEV_CM2

    y = T_MeV / E_nu_MeV
    bracket = g_L ** 2 + g_R ** 2 * (1.0 - y) ** 2 - g_L * g_R * m_e * T_MeV / E_nu_MeV ** 2
    result = prefactor * bracket

    T_max = 2.0 * E_nu_MeV ** 2 / (m_e + 2.0 * E_nu_MeV)
    allowed = (T_MeV >= 0.0) & (T_MeV <= T_max)
    return torch.where(allowed, result, torch.zeros_like(result))


@torch.no_grad()
def _cross_section_grid(
    E_nu_grid_MeV: torch.Tensor,
    T_grid_MeV: torch.Tensor,
    couplings: _Couplings,
) -> torch.Tensor:
    """Evaluate a differential cross section on an outer-product grid.

    Args:
        E_nu_grid_MeV: Neutrino energy grid in MeV, shape ``(n_E,)``.
        T_grid_MeV: Electron recoil grid in MeV, shape ``(n_T,)``.
        couplings: Left- and right-handed interaction couplings.

    Returns:
        Differential cross section in cm^2/MeV, shape ``(n_E, n_T)``.
    """
    E = E_nu_grid_MeV[:, None]
    T = T_grid_MeV[None, :]
    return dsigma_dT(E, T, couplings.g_L, couplings.g_R)


def nue_cross_section_grid(
    E_nu_grid_MeV: torch.Tensor,
    T_grid_MeV: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the electron-neutrino elastic cross section on a grid.

    Args:
        E_nu_grid_MeV: Neutrino energy grid in MeV, shape ``(n_E,)``.
        T_grid_MeV: Electron recoil grid in MeV, shape ``(n_T,)``.

    Returns:
        Differential cross section in cm^2/MeV, shape ``(n_E, n_T)``.
    """
    return _cross_section_grid(E_nu_grid_MeV, T_grid_MeV, NUE_COUPLINGS)


def numutau_cross_section_grid(
    E_nu_grid_MeV: torch.Tensor,
    T_grid_MeV: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the muon/tau-neutrino elastic cross section on a grid.

    Args:
        E_nu_grid_MeV: Neutrino energy grid in MeV, shape ``(n_E,)``.
        T_grid_MeV: Electron recoil grid in MeV, shape ``(n_T,)``.

    Returns:
        Differential cross section in cm^2/MeV, shape ``(n_E, n_T)``.
    """
    return _cross_section_grid(E_nu_grid_MeV, T_grid_MeV, NUMUTAU_COUPLINGS)
