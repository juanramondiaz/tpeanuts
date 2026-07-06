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

"""I/O helpers for the PREM500 tabulated density profile.

Loads the PREM500 CSV (Dziewonski & Anderson 1981, *Phys. Earth Planet. Int.*
25, 297–356; IRIS SPUD dataset ID 9785674), converts mass density to electron
density in mol/cm³ using layer-appropriate ⟨Z/A⟩ values, and builds the
piecewise-linear-in-r² shell representation required by
``PremTabulatedProfile``.

The file has no header and nine comma-separated columns; only the first two
(radius in metres and mass density in kg/m³) are used here.

Module functions:
    load_prem500_profile(...): Read PREM500 CSV → ``(rj, coefficients)``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import tpeanuts.util.default as default
from tpeanuts.util.torch_util import default_device


# ── Z/A conversion factors ────────────────────────────────────────────────────
# Electron fraction (electrons per nucleon) by PREM structural layer.
# Inner core: solid Fe-Ni at ~85/15 wt ratio.  Fe: 26/56 = 0.4643, Ni: 28/58 = 0.4828.
_ZA_INNER_CORE: float = 0.4656
# Outer core: liquid Fe-Ni with dissolved light elements (O, S, Si).
_ZA_OUTER_CORE: float = 0.4678
# Mantle + crust: silicate minerals (MgSiO₃, Mg₂SiO₄, SiO₂, …).
_ZA_MANTLE_CRUST: float = 0.4940
# Ocean: seawater approximated as H₂O  (Z = 10, A = 18).
_ZA_OCEAN: float = 0.5556

# PREM discontinuity radii used to assign Z/A (km).
_R_ICB_KM: float = 1221.5    # inner-core boundary
_R_CMB_KM: float = 3480.0    # core-mantle boundary
_R_OCEAN_KM: float = 6368.0  # ocean floor


def _za_from_radius_km(radius_km: np.ndarray) -> np.ndarray:
    """Return the ⟨Z/A⟩ ratio for each tabulated PREM radius (in km)."""
    return np.where(
        radius_km <= _R_ICB_KM,
        _ZA_INNER_CORE,
        np.where(
            radius_km <= _R_CMB_KM,
            _ZA_OUTER_CORE,
            np.where(radius_km <= _R_OCEAN_KM, _ZA_MANTLE_CRUST, _ZA_OCEAN),
        ),
    )


def load_prem500_profile(
    prem_file: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = default.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load PREM500 CSV and build the piecewise-linear-in-r² shell profile.

    Each shell spans consecutive tabulated radii.  Discontinuities (adjacent
    rows with the same radius) are handled correctly: the zero-thickness gap
    between the two rows is skipped, so the shell ending at the interface uses
    the "below" density and the shell starting at the interface uses the
    "above" density.

    Within shell *i* (inner boundary rᵢ, outer boundary rᵢ₊₁):

    .. math::

        n_e(r^2) = A_i + B_i \\cdot r^2,

    where :math:`A_i` and :math:`B_i` are determined by linear interpolation
    between the electron densities at the two endpoints.

    Args:
        prem_file: Path to the PREM500 CSV file (nine columns, no header).
        device: Torch device for the returned tensors.
        dtype: Real dtype for the returned tensors.

    Returns:
        ``(rj, coefficients)`` where

        * ``rj`` — 1D tensor of shape ``(n_shells,)`` with the normalized
          (by R_E) outer boundary of each shell, strictly increasing in
          ``(0, 1]``.
        * ``coefficients`` — tensor of shape ``(n_shells, 2)`` with columns
          ``[A, B]`` for the linear-in-r² density fit of each shell.

    Raises:
        FileNotFoundError: If ``prem_file`` does not exist.
        ValueError: If no valid shells can be built from the file.
    """
    device = default_device(device)

    if not os.path.isfile(prem_file):
        raise FileNotFoundError(f"PREM500 file not found: {prem_file}")

    data = np.loadtxt(prem_file, delimiter=",")
    radius_m = data[:, 0]
    density_kg_m3 = data[:, 1]

    radius_km = radius_m / 1.0e3
    density_g_cm3 = density_kg_m3 / 1.0e3

    ne_mol_cm3 = _za_from_radius_km(radius_km) * density_g_cm3

    # Normalize radius by the maximum tabulated value (Earth radius).
    r_norm = radius_m / radius_m.max()

    # Build one shell per pair of consecutive rows with distinct radii.
    r_inner_list: list[float] = []
    r_outer_list: list[float] = []
    ne_inner_list: list[float] = []
    ne_outer_list: list[float] = []

    for i in range(len(r_norm) - 1):
        if r_norm[i] == r_norm[i + 1]:
            continue  # skip zero-thickness gap at a PREM discontinuity
        r_inner_list.append(r_norm[i])
        r_outer_list.append(r_norm[i + 1])
        ne_inner_list.append(ne_mol_cm3[i])
        ne_outer_list.append(ne_mol_cm3[i + 1])

    if not r_inner_list:
        raise ValueError(
            "No valid shells could be built from the PREM500 file.  "
            "Check that the file has the expected format."
        )

    r_inner = np.asarray(r_inner_list)
    r_outer = np.asarray(r_outer_list)
    ne_inner = np.asarray(ne_inner_list)
    ne_outer = np.asarray(ne_outer_list)

    # Fit linear-in-r²: n_e(r²) = A + B·r²  for each shell.
    r_inner_sq = r_inner ** 2
    r_outer_sq = r_outer ** 2
    dr_sq = r_outer_sq - r_inner_sq          # always > 0 for non-degenerate pairs

    B = (ne_outer - ne_inner) / dr_sq
    A = ne_inner - B * r_inner_sq

    rj = torch.as_tensor(r_outer, device=device, dtype=dtype)
    coefficients = torch.as_tensor(
        np.stack([A, B], axis=-1),
        device=device,
        dtype=dtype,
    )
    return rj, coefficients
