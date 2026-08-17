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
Neutrino-deuteron breakup cross sections (SNO's CC and NC channels), from
the real published Nakamura et al. calculation.

    Charged current:  nu_e   + d -> p + p + e-   (Q = Q_CC_DEUTERON_MEV)
    Neutral current:  nu_x   + d -> p + n + nu_x (Q = Q_NC_DEUTERON_MEV,
                                                    the deuteron binding energy)

**No longer an illustrative placeholder.** ``sigma_cc_total``/``sigma_nc_total``/
``cc_cross_section_grid`` below interpolate the real tables of Nakamura,
Sato, Ando, Park, Myhrer, Gudkov & Kubodera, Nucl. Phys. A707 (2002) 561
(nucl-th/0201062), fetched from the authors' own online tables and converted
to `.csv` by ``notebooks/external/nakamura/Nakamura1_generator.ipynb`` into
``data/detector/sno/nakamura/``:

    total_cross_sections.csv
        (E_nu_MeV, sigma_cc_cm2, sigma_nc_cm2, sigma_cc_bar_cm2,
        sigma_nc_bar_cm2), 1.5-170 MeV. Only the neutrino (not antineutrino)
        columns are used here -- SNO sees solar nu_e, not reactor/supernova
        nu_e_bar.
    cc_electron_spectrum.csv
        (E_nu_MeV, E_e_MeV, p_e_MeV, dsigma_dpe_cm2_per_MeV), the CC
        reaction's outgoing-electron momentum distribution at each of 63
        tabulated E_nu (1.5-20 MeV, SNO's solar-neutrino range).

``cc_cross_section_grid`` converts the tabulated dsigma/dp_e into dsigma/dT_e
(T_e = E_e - m_e, the kinetic energy convention every other cross section in
this project uses) via the exact kinematic Jacobian dp_e/dT_e = E_e/p_e, and
linearly interpolates between the two tabulated E_nu curves bracketing each
requested grid energy -- this replaces the previous version's illustrative
"outgoing electron is monoenergetic at T_e = E_nu - Q_CC" two-body-like
narrow-kernel approximation with the real tabulated 3-body (p, p, e-)
electron spectrum shape, so the physical recoil spread this project's
detector response then convolves is now itself real, not a numerical
stand-in for an unmodeled spread (contrast the previous module docstring,
and the ``detector.sno.response``/``detector.interaction.inverse_beta_decay``
docstrings this replaced pattern still appears in for other channels).

No NC differential cross section is exposed here: SNO's NC channel is
detected via a neutron-capture gamma cascade with no electron-recoil energy
tied to E_nu the way CC's outgoing electron is, so ``sigma_nc_total`` alone
(the total breakup rate) is what ``detector.sno.event_rate.nc_event_rate``
needs -- see that function's own docstring for how the captured-neutron
visible-energy response is built instead (a fixed capture-gamma energy, not
a function of E_nu).

Module contents:
    sigma_cc_total(...), sigma_nc_total(...)
        Real tabulated total cross sections, linearly interpolated (and
        floored to exactly 0 below each reaction's threshold).
    cc_cross_section_grid(...)
        Real tabulated differential CC cross section on an (E_nu_grid,
        T_grid) pair, ready for ``event_rate.true_observable_spectrum``.
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
    """Load and cache ``total_cross_sections.csv``, sorted by E_nu_MeV."""
    path = _NAKAMURA_DIR / "total_cross_sections.csv"
    return pd.read_csv(path).sort_values("E_nu_MeV").reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_cc_electron_curves() -> Tuple[np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """Load, Jacobian-convert, and cache the CC electron spectrum per tabulated E_nu.

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
    table = _load_total_table()
    E_np = to_numpy(E_nu_MeV)
    x = table["E_nu_MeV"].to_numpy()
    y = table[column].to_numpy()
    sigma = np.interp(E_np, x, y, left=0.0, right=y[-1])
    sigma = np.where(E_np < threshold_MeV, 0.0, sigma)
    return as_tensor_like(sigma, E_nu_MeV)


def sigma_cc_total(E_nu_MeV: torch.Tensor) -> torch.Tensor:
    """Real (Nakamura et al. 2002) total CC cross section sigma(nu_e + d -> p + p + e-), cm^2.

    Linearly interpolated on the tabulated 1.5-170 MeV grid (see module
    docstring); flat-extrapolated above 170 MeV, floored to exactly 0 below
    ``constant.Q_CC_DEUTERON_MEV``.

    Args:
        E_nu_MeV: Neutrino energy, any shape.

    Returns:
        Same shape as ``E_nu_MeV``.
    """
    return _interp_total(E_nu_MeV, "sigma_cc_cm2", constant.Q_CC_DEUTERON_MEV)


def sigma_nc_total(E_nu_MeV: torch.Tensor) -> torch.Tensor:
    """Real (Nakamura et al. 2002) total NC cross section sigma(nu_x + d -> p + n + nu_x), cm^2.

    Linearly interpolated on the tabulated 1.5-170 MeV grid (see module
    docstring); flat-extrapolated above 170 MeV, floored to exactly 0 below
    ``constant.Q_NC_DEUTERON_MEV``. Consumed by
    ``detector.sno.event_rate.nc_event_rate`` (the total NC breakup rate,
    convolved there with a fixed-energy neutron-capture response rather than
    an E_nu-dependent differential cross section -- see that function).

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

    See module docstring for the Jacobian conversion (dsigma/dp_e ->
    dsigma/dT_e) and the bracketing-E_nu interpolation scheme.

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
