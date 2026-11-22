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

"""Neutrino-deuteron breakup cross sections.

::

    Charged current:  nu_e   + d -> p + p + e-   (Q = Q_CC_DEUTERON_MEV)
    Neutral current:  nu_x   + d -> p + n + nu_x (Q = Q_NC_DEUTERON_MEV,
                                                    the deuteron binding energy)

The tabulated values are from Nakamura et al., Nucl. Phys. A707 (2002)
561 -- SNO's own CC/NC/ES channels' cross sections, cached CSVs under
``data/experiments/sno/nakamura/`` (moved here from v1_0's identical data
release, not regenerated). The charged-current spectrum is converted
from dsigma/dp_e to dsigma/dT_e with ``dp_e/dT_e = E_e/p_e`` and
interpolated in neutrino energy. Only a total cross section is provided
for the neutral-current process (no measured/tabulated differential
recoil spectrum for it in this data release).

Units: all cross sections are returned in units of ``constant.SIGMA_UNIT_CM2`` =
1e-42 cm^2 (``dsigma/dT`` in 1e-42 cm^2/MeV): the tabulated values (down to
~1e-49 cm^2/MeV) are zero in float32 otherwise. The CSV files stay in cm^2; the
division by the unit is done in float64 when they are loaded.

Module contents:
    cc_cross_section(...)
        Interpolate the total charged-current cross section.
    nc_cross_section(...)
        Interpolate the total neutral-current cross section.
    cc_cross_section_grid(...)
        Evaluate the differential charged-current cross section on a
        grid.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch

from tpeanuts.data import package_data_dir
from tpeanuts.util import constant
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.math import interp1d_linear, searchsorted_contiguous
from tpeanuts.util.tensor import as_tensor

Tensor = torch.Tensor

_NAKAMURA_DIR = package_data_dir() / "experiments" / "sno" / "nakamura"


def _read_csv_rows(path: Union[str, Path]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV file into its own field names and rows, in file order.

    Raises:
        ValueError: If the file has no header row.
    """
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {path!r}.")
        return list(reader.fieldnames), list(reader)


@dataclass(frozen=True)
class _TotalCrossSectionTable:
    E_nu_MeV: Tensor
    sigma_cc: Tensor  # units of constant.SIGMA_UNIT_CM2
    sigma_nc: Tensor  # units of constant.SIGMA_UNIT_CM2


@lru_cache(maxsize=None)
def _load_total_table(context: RuntimeContext) -> _TotalCrossSectionTable:
    """Load the total cross-section table sorted by neutrino energy, cached per context."""
    _, rows = _read_csv_rows(_NAKAMURA_DIR / "total_cross_sections.csv")
    rows = sorted(rows, key=lambda row: float(row["E_nu_MeV"]))
    return _TotalCrossSectionTable(
        E_nu_MeV=as_tensor([float(r["E_nu_MeV"]) for r in rows], device=context.device, dtype=context.dtype),
        sigma_cc=as_tensor(
            [float(r["sigma_cc_cm2"]) / constant.SIGMA_UNIT_CM2 for r in rows],
            device=context.device, dtype=context.dtype,
        ),
        sigma_nc=as_tensor(
            [float(r["sigma_nc_cm2"]) / constant.SIGMA_UNIT_CM2 for r in rows],
            device=context.device, dtype=context.dtype,
        ),
    )


@lru_cache(maxsize=None)
def _load_cc_electron_curves(context: RuntimeContext) -> Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], ...]]:
    """Load, Jacobian-convert, and cache the CC electron spectrum per tabulated E_nu, cached per context.

    Returns:
        ``(tab_E_nu_MeV, curves)``: ``tab_E_nu_MeV`` is the sorted
        tensor of tabulated energy_MeV values, shape ``(n_tab,)``;
        ``curves[i]`` is ``(T_e_MeV, dsigma_dTe)`` (dsigma/dT_e in 1e-42 cm^2/MeV) for
        ``tab_E_nu_MeV[i]``, sorted by ``T_e_MeV`` (each curve has its
        own, generally different, number of tabulated points).
    """
    _, rows = _read_csv_rows(_NAKAMURA_DIR / "cc_electron_spectrum.csv")
    grouped: Dict[float, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["E_nu_MeV"]), []).append(row)

    m_e = constant.M_ELECTRON_MEV
    tab_energies = sorted(grouped)
    curves: List[Tuple[Tensor, Tensor]] = []
    for E_nu in tab_energies:
        group = sorted(grouped[E_nu], key=lambda row: float(row["p_e_MeV"]))
        E_e = as_tensor([float(row["E_e_MeV"]) for row in group], device=context.device, dtype=context.dtype)
        p_e = as_tensor([float(row["p_e_MeV"]) for row in group], device=context.device, dtype=context.dtype)
        dsigma_dpe = as_tensor(
            [float(row["dsigma_dpe_cm2_per_MeV"]) / constant.SIGMA_UNIT_CM2 for row in group],
            device=context.device, dtype=context.dtype,
        )

        # dsigma/dT_e = dsigma/dp_e * dp_e/dT_e, dp_e/dT_e = dp_e/dE_e = E_e/p_e
        # (from E_e^2 = p_e^2 + m_e^2), T_e = E_e - m_e. At the kinematic
        # threshold row p_e=0, dsigma/dp_e is itself 0, so the product is
        # defined as 0 rather than dividing by zero.
        jacobian = torch.where(p_e > 0, E_e / p_e.clamp_min(torch.finfo(p_e.dtype).tiny), torch.zeros_like(p_e))
        curves.append((E_e - m_e, dsigma_dpe * jacobian))

    tab_E_nu_MeV = as_tensor(tab_energies, device=context.device, dtype=context.dtype)
    return tab_E_nu_MeV, tuple(curves)


def _interp_total(energy_MeV: Tensor, table_column: Tensor, threshold_MeV: float, *, context: RuntimeContext) -> Tensor:
    """Interpolate one total cross-section column at requested energies.

    Linearly interpolated on the tabulated grid, flat-extrapolated above
    it (``right=table_column[-1]``), and set to exactly 0 below both the
    tabulated grid's own start (``left=0``) and ``threshold_MeV``.
    """
    table = _load_total_table(context)
    zero = torch.zeros((), device=context.device, dtype=context.dtype)
    sigma = interp1d_linear(energy_MeV, table.E_nu_MeV, table_column, left=zero, right=table_column[-1])
    return torch.where(energy_MeV < threshold_MeV, zero, sigma)


def cc_cross_section(energy_MeV: Tensor, *, context: RuntimeContext) -> Tensor:
    """Real (Nakamura et al. 2002) total CC cross section sigma(nu_e + d -> p + p + e-), in units of 1e-42 cm^2.

    Linearly interpolated on the tabulated 1.5-170 MeV grid,
    flat-extrapolated above 170 MeV and set to exactly 0 below
    ``tpeanuts.util.constant.Q_CC_DEUTERON_MEV``.

    Args:
        energy_MeV: Neutrino energy, any shape.
        context: RuntimeContext the tabulated data is loaded/cached on.

    Returns:
        Same shape as ``energy_MeV``.
    """
    table = _load_total_table(context)
    return _interp_total(energy_MeV, table.sigma_cc, constant.Q_CC_DEUTERON_MEV, context=context)


def nc_cross_section(energy_MeV: Tensor, *, context: RuntimeContext) -> Tensor:
    """Real (Nakamura et al. 2002) total NC cross section sigma(nu_x + d -> p + n + nu_x), in units of 1e-42 cm^2.

    Linearly interpolated on the tabulated 1.5-170 MeV grid,
    flat-extrapolated above 170 MeV and set to exactly 0 below
    ``tpeanuts.util.constant.Q_NC_DEUTERON_MEV``.

    Args:
        energy_MeV: Neutrino energy, any shape.
        context: RuntimeContext the tabulated data is loaded/cached on.

    Returns:
        Same shape as ``energy_MeV``.
    """
    table = _load_total_table(context)
    return _interp_total(energy_MeV, table.sigma_nc, constant.Q_NC_DEUTERON_MEV, context=context)


def _interp_energy_axis(energy_MeV: Tensor, tab_E_nu_MeV: Tensor, tab_values: Tensor) -> Tensor:
    """Linearly interpolate a ``(n_tab, n_T)`` table along its own leading (tabulated-energy) axis.

    Flat-extrapolated outside ``[tab_E_nu_MeV[0], tab_E_nu_MeV[-1]]``
    (the nearest edge row is used unchanged) -- a local helper, not
    ``tpeanuts.util.math.interp1d_linear``, since that function only
    interpolates a 1-D table; generalizing it to a 2-D ``fp`` for this
    one caller was judged not worth the added complexity there (see
    this module's own development notes).

    Args:
        energy_MeV: Query energies, shape ``(n_E,)``.
        tab_E_nu_MeV: Strictly increasing tabulated energy grid, shape
            ``(n_tab,)``.
        tab_values: Table values at ``tab_E_nu_MeV``, shape ``(n_tab,
            n_T)``.

    Returns:
        Tensor shaped ``(n_E, n_T)``.
    """
    n = tab_E_nu_MeV.shape[0]
    idx_hi = searchsorted_contiguous(tab_E_nu_MeV, energy_MeV).clamp(1, n - 1)
    idx_lo = idx_hi - 1

    E_lo, E_hi = tab_E_nu_MeV[idx_lo], tab_E_nu_MeV[idx_hi]
    v_lo, v_hi = tab_values[idx_lo], tab_values[idx_hi]

    frac = ((energy_MeV - E_lo) / (E_hi - E_lo).clamp_min(torch.finfo(tab_E_nu_MeV.dtype).eps)).clamp(0.0, 1.0)
    return v_lo + frac[:, None] * (v_hi - v_lo)


def cc_cross_section_grid(E_nu_grid_MeV: Tensor, T_grid_MeV: Tensor, *, context: RuntimeContext) -> Tensor:
    """Real (Nakamura et al. 2002) differential CC cross section dsigma_CC/dT on an (E_nu_grid, T_grid) pair.

    Tabulated dsigma/dp_e values are converted to dsigma/dT_e and
    linearly interpolated between neighboring neutrino energies. Each
    bracketing curve is itself linearly interpolated onto
    ``T_grid_MeV`` first (zero outside that curve's own kinematically
    allowed T_e range, which grows with E_nu), then the two resulting
    arrays are linearly interpolated in E_nu (see
    ``_interp_energy_axis``); energies outside the tabulated range use
    the nearest edge curve unchanged (flat extrapolation).

    Args:
        E_nu_grid_MeV: True neutrino energy grid, shape ``(n_E,)``.
        T_grid_MeV: Electron kinetic energy grid, shape ``(n_T,)``.
        context: RuntimeContext the tabulated data is loaded/cached on.

    Returns:
        Real tensor shaped ``(n_E, n_T)``, in units of 1e-42 cm^2/MeV, floored to exactly
        0 for E_nu below ``tpeanuts.util.constant.Q_CC_DEUTERON_MEV``.
    """
    tab_E_nu_MeV, curves = _load_cc_electron_curves(context)
    zero = torch.zeros((), device=context.device, dtype=context.dtype)

    curves_on_grid = torch.stack([
        interp1d_linear(T_grid_MeV, T_e, dsigma, left=zero, right=zero) for T_e, dsigma in curves
    ])  # (n_tab, n_T)

    grid = _interp_energy_axis(E_nu_grid_MeV, tab_E_nu_MeV, curves_on_grid)  # (n_E, n_T)
    below_threshold = E_nu_grid_MeV < constant.Q_CC_DEUTERON_MEV
    return torch.where(below_threshold[:, None], torch.zeros_like(grid), grid)
