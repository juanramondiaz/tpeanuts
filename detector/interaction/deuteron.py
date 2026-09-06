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

"""Neutrino-deuteron breakup cross sections.

    Charged current:  nu_e   + d -> p + p + e-   (Q = Q_CC_DEUTERON_MEV)
    Neutral current:  nu_x   + d -> p + n + nu_x (Q = Q_NC_DEUTERON_MEV,
                                                    the deuteron binding energy)

The tabulated values are from Nakamura et al., Nucl. Phys. A707 (2002) 561.
The charged-current spectrum is converted from dsigma/dp_e to dsigma/dT_e
with dp_e/dT_e = E_e/p_e and interpolated in neutrino energy. Only a total
cross section is provided for the neutral-current process.

Module contents:
    sigma_cc_total(...)
        Interpolate the total charged-current cross section.
    sigma_nc_total(...)
        Interpolate the total neutral-current cross section.
    cc_cross_section_grid(...)
        Evaluate the differential charged-current cross section on a grid.
"""

from __future__ import annotations

import functools
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

import tpeanuts.util.constant as constant
from tpeanuts.util.io import package_dir
from tpeanuts.util.type import as_tensor_like, to_numpy

_NAKAMURA_DIR = package_dir() / "data" / "detector" / "sno" / "nakamura"


@functools.lru_cache(maxsize=1)
def _load_total_table() -> pd.DataFrame:
    """Load the total cross-section table sorted by neutrino energy.

    Args:
        None.

    Returns:
        Table sorted by its ``E_nu_MeV`` column.
    """
    path = _NAKAMURA_DIR / "total_cross_sections.csv"
    return pd.read_csv(path).sort_values("E_nu_MeV").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_cc_electron_curves() -> Tuple[np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """Load, Jacobian-convert, and cache the CC electron spectrum per tabulated E_nu.

    Args:
        None.

    Returns:
        ``(energies, curves)``: ``energies`` is the sorted array of
        tabulated E_nu_MeV values; ``curves[E_nu]`` is
        ``(T_e_MeV, dsigma_dTe_cm2_per_MeV)``, sorted by T_e_MeV.
    """
    path = _NAKAMURA_DIR / "cc_electron_spectrum.csv"
    df = pd.read_csv(path)

    m_e = constant.M_ELECTRON_MEV
    curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for E_nu, group in df.groupby("E_nu_MeV"):
        group = group.sort_values("p_e_MeV")
        E_e = group["E_e_MeV"].to_numpy()
        p_e = group["p_e_MeV"].to_numpy()
        dsigma_dpe = group["dsigma_dpe_cm2_per_MeV"].to_numpy()

        # dsigma/dT_e = dsigma/dp_e * dp_e/dT_e, dp_e/dT_e = dp_e/dE_e = E_e/p_e
        # (from E_e^2 = p_e^2 + m_e^2), T_e = E_e - m_e. At the kinematic
        # threshold row p_e=0, dsigma/dp_e is itself 0, so the product is
        # defined as 0 rather than diverging.
        jacobian = np.divide(E_e, p_e, out=np.zeros_like(E_e), where=p_e > 0)
        dsigma_dTe = dsigma_dpe * jacobian
        T_e = E_e - m_e

        curves[float(E_nu)] = (T_e, dsigma_dTe)

    energies = np.array(sorted(curves))
    return energies, curves


def _interp_total(E_nu_MeV: torch.Tensor, column: str, threshold_MeV: float) -> torch.Tensor:
    """Interpolate one total cross-section column at requested energies.

    Args:
        E_nu_MeV: Neutrino energies in MeV, with any shape.
        column: Name of the cross-section column to interpolate.
        threshold_MeV: Reaction threshold in MeV; lower energies return zero.

    Returns:
        Cross sections in cm^2 with the shape of ``E_nu_MeV``.
    """
    table = _load_total_table()
    E_np = to_numpy(E_nu_MeV)
    x = table["E_nu_MeV"].to_numpy()
    y = table[column].to_numpy()
    sigma = np.interp(E_np, x, y, left=0.0, right=y[-1])
    sigma = np.where(E_np < threshold_MeV, 0.0, sigma)
    return as_tensor_like(sigma, E_nu_MeV)


def sigma_cc_total(E_nu_MeV: torch.Tensor) -> torch.Tensor:
    """Real (Nakamura et al. 2002) total CC cross section sigma(nu_e + d -> p + p + e-), cm^2.

    Linearly interpolated on the tabulated 1.5-170 MeV grid,
    flat-extrapolated above 170 MeV and set to exactly 0 below
    ``constant.Q_CC_DEUTERON_MEV``.

    Args:
        E_nu_MeV: Neutrino energy, any shape.

    Returns:
        Same shape as ``E_nu_MeV``.
    """
    return _interp_total(E_nu_MeV, "sigma_cc_cm2", constant.Q_CC_DEUTERON_MEV)


def sigma_nc_total(E_nu_MeV: torch.Tensor) -> torch.Tensor:
    """Real (Nakamura et al. 2002) total NC cross section sigma(nu_x + d -> p + n + nu_x), cm^2.

    Linearly interpolated on the tabulated 1.5-170 MeV grid,
    flat-extrapolated above 170 MeV and set to exactly 0 below
    ``constant.Q_NC_DEUTERON_MEV``.

    Args:
        E_nu_MeV: Neutrino energy, any shape.

    Returns:
        Same shape as ``E_nu_MeV``.
    """
    return _interp_total(E_nu_MeV, "sigma_nc_cm2", constant.Q_NC_DEUTERON_MEV)


def _dsigma_dTe_at_energy(
    E_nu: float,
    T_grid: np.ndarray,
    energies: np.ndarray,
    curves: Dict[float, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """dsigma/dT_e(T_grid) at one E_nu, linearly interpolated between bracketing tabulated curves.

    Each bracketing curve is itself linearly interpolated on ``T_grid``
    first (zero outside that curve's own kinematically allowed T_e range,
    which grows with E_nu), then the two resulting arrays are linearly
    interpolated in E_nu. Energies outside the tabulated range use the
    nearest edge curve unchanged (flat extrapolation).

    Args:
        E_nu: Neutrino energy in MeV.
        T_grid: Electron kinetic-energy grid in MeV.
        energies: Sorted tabulated neutrino energies in MeV.
        curves: Differential cross-section curve for each tabulated energy.

    Returns:
        Differential cross section in cm^2/MeV on ``T_grid``.
    """
    if E_nu <= energies[0]:
        idx_lo = idx_hi = 0
    elif E_nu >= energies[-1]:
        idx_lo = idx_hi = len(energies) - 1
    else:
        idx_hi = int(np.searchsorted(energies, E_nu))
        idx_lo = idx_hi - 1

    E_lo, E_hi = energies[idx_lo], energies[idx_hi]
    T_lo, S_lo = curves[E_lo]
    s_lo = np.interp(T_grid, T_lo, S_lo, left=0.0, right=0.0)
    if E_hi == E_lo:
        return s_lo

    T_hi, S_hi = curves[E_hi]
    s_hi = np.interp(T_grid, T_hi, S_hi, left=0.0, right=0.0)
    w = (E_nu - E_lo) / (E_hi - E_lo)
    return (1.0 - w) * s_lo + w * s_hi


def cc_cross_section_grid(
    E_nu_grid_MeV: torch.Tensor,
    T_grid_MeV: torch.Tensor,
) -> torch.Tensor:
    """Real (Nakamura et al. 2002) differential CC cross section dsigma_CC/dT on an (E_nu_grid, T_grid) pair.

    Tabulated dsigma/dp_e values are converted to dsigma/dT_e and linearly
    interpolated between neighboring neutrino energies.

    Args:
        E_nu_grid_MeV: True neutrino energy grid, shape ``(n_E,)``.
        T_grid_MeV: Electron kinetic energy grid, shape ``(n_T,)``.

    Returns:
        Real tensor shaped ``(n_E, n_T)``, cm^2/MeV, floored to exactly 0
        for E_nu below ``constant.Q_CC_DEUTERON_MEV``.
    """
    energies, curves = _load_cc_electron_curves()
    E_np = to_numpy(E_nu_grid_MeV)
    T_np = to_numpy(T_grid_MeV)

    rows = [_dsigma_dTe_at_energy(float(E), T_np, energies, curves) for E in E_np]
    grid = np.stack(rows, axis=0)
    grid[E_np < constant.Q_CC_DEUTERON_MEV, :] = 0.0

    return as_tensor_like(grid, T_grid_MeV)
