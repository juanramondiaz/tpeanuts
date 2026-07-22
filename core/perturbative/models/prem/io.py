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

"""I/O helpers for canonical PREM radial-density tables.

Loads the canonical PREM CSV (Dziewonski & Anderson 1981, *Phys. Earth Planet. Int.*
25, 297–356; IRIS SPUD dataset ID 9785674), converts mass density to electron
density in mol/cm³ using layer-appropriate ⟨Z/A⟩ values, and builds the
piecewise-linear-in-r² shell representation required by
``PremTabulatedProfile``.

The file has no header and nine comma-separated columns; only the first two
(radius in metres and mass density in kg/m³) are used here.

The same layer-appropriate ⟨Z/A⟩ values also give the neutron fraction
(1 - ⟨Z/A⟩), since each shell is treated as pure nucleonic matter (electrons
+ neutrons per nucleon, ignoring binding-energy mass defects). This enables
the 3+1 sterile extension's neutral-current matter term (see
``core.common.hamiltonian.hamiltonian_matter_reduced``).

Module functions:
    load_prem_profile(...): Read canonical PREM CSV → ``(rj, coefficients)`` for
        electron density.
    load_prem_neutron_profile(...): Read canonical PREM CSV →
        ``(rj, coefficients)`` for neutron density.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import tpeanuts.config.default as default
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


def _load_prem_radius_density(density_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the canonical PREM CSV and return radius and mass density.

    Args:
        density_path: Path to the canonical PREM density CSV.

    Returns:
        Tuple of 1D arrays: tabulated radius in km and mass density in
        g/cm³, in file row order.

    Raises:
        FileNotFoundError: If ``density_path`` does not exist.
    """
    if not os.path.isfile(density_path):
        raise FileNotFoundError(f"PREM density file not found: {density_path}")
    data = np.genfromtxt(density_path, delimiter=",", names=True)
    required = {"radius_km", "mass_density_g_cm3"}
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(
            "Canonical PREM table requires radius_km and mass_density_g_cm3 columns."
        )
    return data["radius_km"], data["mass_density_g_cm3"]


def _fit_linear_in_r_squared_shells(
    radius_km: np.ndarray,
    n_mol_cm3: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a piecewise-linear-in-r² molar density profile from tabulated shells.

    Each shell spans consecutive tabulated radii. Discontinuities (adjacent
    rows with the same radius) are handled correctly: the zero-thickness gap
    between the two rows is skipped, so the shell ending at the interface uses
    the "below" density and the shell starting at the interface uses the
    "above" density.

    Within shell *i* (inner boundary rᵢ, outer boundary rᵢ₊₁):

    .. math::

        n(r^2) = A_i + B_i \\cdot r^2,

    where :math:`A_i` and :math:`B_i` are determined by linear interpolation
    between the tabulated densities at the two endpoints.

    Args:
        radius_km: Tabulated radius in km, in file row order (same length as
            ``n_mol_cm3``).
        n_mol_cm3: Tabulated molar density in mol/cm³, aligned with
            ``radius_km``.
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
        ValueError: If no valid shells can be built from the tabulated data.
    """
    radius_m = radius_km * 1.0e3

    # Normalize radius by the maximum tabulated value (Earth radius).
    r_norm = radius_m / radius_m.max()

    # Build one shell per pair of consecutive rows with distinct radii.
    r_inner_list: list[float] = []
    r_outer_list: list[float] = []
    n_inner_list: list[float] = []
    n_outer_list: list[float] = []

    for i in range(len(r_norm) - 1):
        if r_norm[i] == r_norm[i + 1]:
            continue  # skip zero-thickness gap at a PREM discontinuity
        r_inner_list.append(r_norm[i])
        r_outer_list.append(r_norm[i + 1])
        n_inner_list.append(n_mol_cm3[i])
        n_outer_list.append(n_mol_cm3[i + 1])

    if not r_inner_list:
        raise ValueError(
            "No valid shells could be built from the canonical PREM file.  "
            "Check that the file has the expected format."
        )

    r_inner = np.asarray(r_inner_list)
    r_outer = np.asarray(r_outer_list)
    n_inner = np.asarray(n_inner_list)
    n_outer = np.asarray(n_outer_list)

    # Fit linear-in-r²: n(r²) = A + B·r²  for each shell.
    r_inner_sq = r_inner ** 2
    r_outer_sq = r_outer ** 2
    dr_sq = r_outer_sq - r_inner_sq          # always > 0 for non-degenerate pairs

    B = (n_outer - n_inner) / dr_sq
    A = n_inner - B * r_inner_sq

    rj = torch.as_tensor(r_outer, device=device, dtype=dtype)
    coefficients = torch.as_tensor(
        np.stack([A, B], axis=-1),
        device=device,
        dtype=dtype,
    )
    return rj, coefficients


def load_prem_profile(
    density_path: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = default.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load canonical PREM data and build the electron-density profile.

    Args:
        density_path: Path to the canonical PREM density CSV.
        device: Torch device for the returned tensors.
        dtype: Real dtype for the returned tensors.

    Returns:
        ``(rj, coefficients)``, see ``_fit_linear_in_r_squared_shells``, built
        from the electron density n_e(r) = ⟨Z/A⟩(r) · ρ(r).

    Raises:
        FileNotFoundError: If ``density_path`` does not exist.
        ValueError: If no valid shells can be built from the file.
    """
    device = default_device(device)
    radius_km, density_g_cm3 = _load_prem_radius_density(density_path)
    ne_mol_cm3 = _za_from_radius_km(radius_km) * density_g_cm3
    return _fit_linear_in_r_squared_shells(
        radius_km, ne_mol_cm3, device=device, dtype=dtype
    )


def load_prem_neutron_profile(
    density_path: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = default.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load canonical PREM data and build the neutron-density profile.

    Uses the complementary fraction (1 - ⟨Z/A⟩) of the same layer-appropriate
    electron fraction used by ``load_prem_profile``, i.e. n_n(r) =
    (1 - ⟨Z/A⟩(r)) · ρ(r). The returned shell boundaries ``rj`` are identical
    to those of ``load_prem_profile`` (same tabulated radii).

    Args:
        density_path: Path to the canonical PREM density CSV.
        device: Torch device for the returned tensors.
        dtype: Real dtype for the returned tensors.

    Returns:
        ``(rj, coefficients)``, see ``_fit_linear_in_r_squared_shells``, built
        from the neutron density n_n(r) = (1 - ⟨Z/A⟩(r)) · ρ(r).

    Raises:
        FileNotFoundError: If ``density_path`` does not exist.
        ValueError: If no valid shells can be built from the file.
    """
    device = default_device(device)
    radius_km, density_g_cm3 = _load_prem_radius_density(density_path)
    nn_mol_cm3 = (1.0 - _za_from_radius_km(radius_km)) * density_g_cm3
    return _fit_linear_in_r_squared_shells(
        radius_km, nn_mol_cm3, device=device, dtype=dtype
    )
