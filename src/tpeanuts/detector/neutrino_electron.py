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

"""Tree-level Standard Model neutrino-electron elastic scattering.

``nu + e^- -> nu + e^-``, differential in the electron recoil kinetic
energy T::

    dsigma/dT(E_nu, T) = (2 G_F^2 m_e / pi) * (hbar c)^2
                          * [g_L^2 + g_R^2 (1 - T/E_nu)^2 - g_L g_R m_e T / E_nu^2],

valid for ``0 <= T <= T_max(E_nu) = 2 E_nu^2 / (m_e + 2 E_nu)`` (zero
outside), with couplings::

    nu_e          (CC + NC): g_L = 1/2 + sin^2(theta_W), g_R = sin^2(theta_W)
    nu_mu, nu_tau (NC only): g_L = -1/2 + sin^2(theta_W), g_R = sin^2(theta_W)

Radiative corrections are not included. The result is in units of
``constant.SIGMA_UNIT_CM2`` (1e-42 cm^2/MeV), see ``tpeanuts.detector.ibd_cross_section``.
Reusable physics, independent
of any one detector's own wiring -- the natural cross section for a
liquid-scintillator/water-Cherenkov detector's own elastic-scattering
channel (e.g. Borexino, SNO's ES channel), the same separation
``tpeanuts.detector.ibd_cross_section``'s own module docstring already
keeps for IBD.

Module contents:
    NUE_COUPLINGS, NUMUTAU_COUPLINGS
        (g_L, g_R) pairs for the electron-flavour and mu/tau-flavour
        cases.
    nue_differential_cross_section(...)
        Calculate the differential cross section for specified
        couplings.
    nue_cross_section_grid(...)
        Evaluate the electron-neutrino cross section on an energy grid.
    numutau_cross_section_grid(...)
        Evaluate the muon/tau-neutrino cross section on an energy grid.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from tpeanuts.util import constant

Tensor = torch.Tensor


class _Couplings(NamedTuple):
    g_L: float
    g_R: float


NUE_COUPLINGS = _Couplings(g_L=0.5 + constant.SIN2_THETA_W, g_R=constant.SIN2_THETA_W)
NUMUTAU_COUPLINGS = _Couplings(g_L=-0.5 + constant.SIN2_THETA_W, g_R=constant.SIN2_THETA_W)

# (hbar*c)^2, MeV^2 cm^2 -- converts G_F^2 * m_e [MeV^-3] into a cross
# section per unit energy [cm^2 / MeV]; divided by SIGMA_UNIT_CM2 so the result
# is in 1e-42 cm^2/MeV (the physical ~1e-45 is subnormal in float32).
_HBARC_MEV_CM2 = (constant.HBARC_MeV_m * 100.0) ** 2 / constant.SIGMA_UNIT_CM2


def nue_differential_cross_section(energy_MeV: Tensor, T_MeV: Tensor, g_L: float, g_R: float) -> Tensor:
    """Differential nu-e elastic scattering cross section, dsigma/dT.

    Args:
        energy_MeV: Neutrino energy, any shape.
        T_MeV: Electron recoil kinetic energy, broadcastable with
            ``energy_MeV``.
        g_L: Left-handed coupling (``NUE_COUPLINGS.g_L`` or
            ``NUMUTAU_COUPLINGS.g_L``).
        g_R: Right-handed coupling (``NUE_COUPLINGS.g_R`` or
            ``NUMUTAU_COUPLINGS.g_R``).

    Returns:
        dsigma/dT in units of 1e-42 cm^2/MeV, broadcast shape of ``energy_MeV``/``T_MeV``;
        exactly 0 outside the kinematically allowed range
        ``0 <= T <= T_max(E_nu)``.
    """
    m_e = constant.M_ELECTRON_MEV
    prefactor = 2.0 * constant.G_F_MEV_M2 ** 2 * m_e / torch.pi * _HBARC_MEV_CM2

    y = T_MeV / energy_MeV
    bracket = g_L ** 2 + g_R ** 2 * (1.0 - y) ** 2 - g_L * g_R * m_e * T_MeV / energy_MeV ** 2
    result = prefactor * bracket

    T_max = 2.0 * energy_MeV ** 2 / (m_e + 2.0 * energy_MeV)
    allowed = (T_MeV >= 0.0) & (T_MeV <= T_max)
    return torch.where(allowed, result, torch.zeros_like(result))


def _cross_section_grid(E_nu_grid_MeV: Tensor, T_grid_MeV: Tensor, couplings: _Couplings) -> Tensor:
    """Evaluate a differential cross section on an outer-product grid.

    Args:
        E_nu_grid_MeV: Neutrino energy grid in MeV, shape ``(n_E,)``.
        T_grid_MeV: Electron recoil grid in MeV, shape ``(n_T,)``.
        couplings: Left- and right-handed interaction couplings.

    Returns:
        Differential cross section in units of 1e-42 cm^2/MeV, shape ``(n_E, n_T)``.
    """
    E = E_nu_grid_MeV[:, None]
    T = T_grid_MeV[None, :]
    return nue_differential_cross_section(E, T, couplings.g_L, couplings.g_R)


def nue_cross_section_grid(E_nu_grid_MeV: Tensor, T_grid_MeV: Tensor) -> Tensor:
    """Evaluate the electron-neutrino elastic cross section on a grid.

    Args:
        E_nu_grid_MeV: Neutrino energy grid in MeV, shape ``(n_E,)``.
        T_grid_MeV: Electron recoil grid in MeV, shape ``(n_T,)``.

    Returns:
        Differential cross section in units of 1e-42 cm^2/MeV, shape ``(n_E, n_T)``.
    """
    return _cross_section_grid(E_nu_grid_MeV, T_grid_MeV, NUE_COUPLINGS)


def numutau_cross_section_grid(E_nu_grid_MeV: Tensor, T_grid_MeV: Tensor) -> Tensor:
    """Evaluate the muon/tau-neutrino elastic cross section on a grid.

    Args:
        E_nu_grid_MeV: Neutrino energy grid in MeV, shape ``(n_E,)``.
        T_grid_MeV: Electron recoil grid in MeV, shape ``(n_T,)``.

    Returns:
        Differential cross section in units of 1e-42 cm^2/MeV, shape ``(n_E, n_T)``.
    """
    return _cross_section_grid(E_nu_grid_MeV, T_grid_MeV, NUMUTAU_COUPLINGS)
