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
Landau-Zener non-adiabatic transition corrections for solar MSW propagation.

In the perfectly adiabatic limit (gamma -> infinity), a neutrino produced
in the solar interior stays in its local matter eigenstate as it propagates
outward through the decreasing electron density.  At finite adiabaticity the
Landau-Zener (LZ) mechanism allows a sudden jump between the two matter
mass-eigenstates as the neutrino traverses the MSW resonance radius.

The exponential approximation (Parke 1986, Phys. Rev. Lett. 57, 1275) gives:

    P_LZ = exp(-pi/2 * gamma_res)

with the adiabaticity parameter evaluated at the resonance:

    gamma_res = Delta_m^2_21 sin^2(2 theta_12) / (2 E hbar_c cos(2 theta_12))
                * L_n(r_res)

where L_n = |n_e / (dn_e/dl)|_res is the density scale height at the MSW
resonance in metres. The resonance radius r_res(E) is defined by the condition

    V_k(Delta_m^2_21, E, n_e(r_res)) = cos(2 theta_12),

i.e. the radius where the solar matter mixing angle theta_12^M = pi/4.

Scope and limitations
---------------------
- Only the theta_12 sector resonance is tracked. The theta_13 resonance lies
  deep in the solar core at very high density and is negligible for all solar
  sources in the standard MSW picture.
- P_LZ = 0 is returned where no resonance exists in the solar volume: energies
  below the resonance threshold, antineutrinos (no MSW resonance in the solar
  interior for standard parameters), and LMA-Dark parameters (theta_12 > pi/4,
  cos2theta_12 < 0, resonance at unphysical negative density).
- For standard LMA parameters and 8B / pp solar neutrinos, gamma_res >> 1
  (typically 10^3 - 10^4), so P_LZ ~ exp(-1500) ~ 0 and the adiabatic
  approximation is excellent. This module becomes relevant for non-standard
  parameter explorations or precision studies near 1-3 MeV.
- Solar NSI is NOT included: the resonance condition is based on the standard
  MSW potential V_k computed in matter_mixing.Vk.

Module functions:
    density_gradient(solar_profile)
        Numerical derivative dn_e/d(r/R_sun) on the profile grid.
    resonance_radius(oscillation, E, solar_profile)
        Radius r_res(E) in solar-radius units where the MSW resonance occurs.
    plz(oscillation, E, solar_profile)
        Landau-Zener transition probability P_LZ(E).
"""



from __future__ import annotations

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.matter_mixing import Vk
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.type import TensorLike


def density_gradient(solar_profile: object) -> torch.Tensor:
    """Compute dn_e/d(r_hat) on the solar profile grid.

    Uses central differences at interior points and one-sided differences
    at the two boundary points. The coordinate r_hat = r/R_sun is the
    dimensionless solar radius stored in ``solar_profile.radius``.

    Args:
        solar_profile: SolarProfile-like object exposing ``radius`` and
            ``density`` 1-D tensors of the same length.

    Returns:
        Tensor of shape ``(n_r,)`` with dn_e/dr_hat in mol/cm^3 per R_sun.
    """
    r = solar_profile.radius    # (n_r,)
    ne = solar_profile.density  # (n_r,)

    # Central differences at interior nodes
    interior = (ne[2:] - ne[:-2]) / (r[2:] - r[:-2])  # (n_r - 2,)

    # One-sided differences at the boundaries
    left = (ne[1:2] - ne[:1]) / (r[1:2] - r[:1])        # (1,)
    right = (ne[-1:] - ne[-2:-1]) / (r[-1:] - r[-2:-1]) # (1,)

    return torch.cat([left, interior, right], dim=0)  # (n_r,)


def resonance_radius(
    oscillation: OscillationParameters,
    E: TensorLike,
    solar_profile: object,
) -> torch.Tensor:
    """Locate the MSW resonance radius for each neutrino energy.

    The theta_12 resonance occurs at the solar radius r_res(E) where the
    dimensionless matter-potential ratio V_k equals cos(2 theta_12):

        V_k(Delta_m^2_21, E, n_e(r_res)) = cos(2 theta_12)

    The function performs a linear interpolation between adjacent grid points
    where the sign of V_k - cos(2 theta_12) changes from positive to negative
    (the condition for a crossing in the standard LMA scenario where the
    density decreases monotonically from the solar centre outward).

    Args:
        oscillation: Oscillation parameters supplying theta_12 and DeltamSq21.
        E: Neutrino energy or 1-D energy grid in MeV.
        solar_profile: SolarProfile-like object with ``radius`` and
            ``density`` 1-D tensors.

    Returns:
        Tensor matching the shape of ``E`` with the resonance radius in solar
        radius units. Entries are NaN for energies without a resonance in the
        solar volume (below threshold, LMA-Dark parameters, antineutrinos,
        etc.).
    """
    th12 = oscillation.pmns.params.theta12
    dm21 = oscillation.DeltamSq21
    cos2th12 = torch.cos(2.0 * th12)  # scalar; negative for LMA-Dark

    radius = solar_profile.radius    # (n_r,)
    density = solar_profile.density  # (n_r,)

    E_t = torch.as_tensor(E, device=radius.device, dtype=radius.dtype)
    scalar_in = E_t.ndim == 0
    E_1d = E_t.reshape(-1)   # (n_E,)
    n_E = E_1d.shape[0]

    # Vk at all (energy, radius) combinations
    vk = Vk(dm21, E_1d[:, None], density[None, :])  # (n_E, n_r)

    # Distance from resonance condition (positive inside resonance, negative outside)
    diff = vk - cos2th12  # (n_E, n_r)

    # Detect the first sign change from + to - along the radius axis
    crossing = (diff[:, :-1] > 0) & (diff[:, 1:] <= 0)  # (n_E, n_r-1)
    has_res = crossing.any(dim=-1)                        # (n_E,)

    # Index of the first crossing for each energy (argmax on a bool tensor)
    idx = crossing.long().argmax(dim=-1).clamp(0, radius.shape[0] - 2)  # (n_E,)
    batch = torch.arange(n_E, device=radius.device)

    # Bracket values for linear interpolation
    r0 = radius[idx]          # (n_E,)
    r1 = radius[idx + 1]      # (n_E,)
    d0 = diff[batch, idx]     # (n_E,)
    d1 = diff[batch, idx + 1] # (n_E,)

    denom = d0 - d1
    safe_denom = torch.where(denom.abs() > 0, denom, torch.ones_like(denom))
    frac = (d0 / safe_denom).clamp(0.0, 1.0)
    r_res = r0 + frac * (r1 - r0)

    # Mark energies without a resonance as NaN
    r_res = torch.where(has_res, r_res, torch.full_like(r_res, float("nan")))

    return r_res.squeeze(0) if scalar_in else r_res


def plz(
    oscillation: OscillationParameters,
    E: TensorLike,
    solar_profile: object,
) -> torch.Tensor:
    """Compute the Landau-Zener transition probability P_LZ(E).

    Uses the exponential approximation of Parke (1986):

        P_LZ = exp(-pi/2 * gamma_res)

    where the adiabaticity parameter at the MSW resonance is

        gamma_res = Delta_m^2_21 sin^2(2 theta_12)
                    / (2 E hbar_c cos(2 theta_12))
                    * L_n(r_res)

    and L_n = |n_e / (dn_e / dl)|_res is the electron-density scale height
    in metres evaluated at the resonance radius.

    P_LZ is set to 0 (fully adiabatic) wherever no resonance exists in the
    solar volume — including below-threshold energies, LMA-Dark parameters,
    and antineutrinos.

    Args:
        oscillation: Oscillation parameters supplying theta_12 and DeltamSq21.
        E: Neutrino energy or 1-D energy grid in MeV.
        solar_profile: SolarProfile-like object exposing ``radius`` and
            ``density``.

    Returns:
        Tensor matching the shape of ``E`` with P_LZ in [0, 1]. For standard
        LMA parameters at solar-neutrino energies, the returned values are
        numerically indistinguishable from zero.
    """
    th12 = oscillation.pmns.params.theta12
    dm21 = oscillation.DeltamSq21
    sin2th12 = torch.sin(2.0 * th12)
    cos2th12 = torch.cos(2.0 * th12)

    radius = solar_profile.radius
    density = solar_profile.density

    E_t = torch.as_tensor(E, device=radius.device, dtype=radius.dtype)
    scalar_in = E_t.ndim == 0
    E_1d = E_t.reshape(-1)  # (n_E,)

    dne_dr = density_gradient(solar_profile)                       # (n_r,)
    r_res = resonance_radius(oscillation, E_1d, solar_profile)     # (n_E,), NaN if absent
    has_res = torch.isfinite(r_res)                                # (n_E,)

    # Replace NaN with a safe interior index so that interp1d does not produce
    # NaN output; these entries will be masked to 0 afterwards.
    r_safe = torch.where(
        has_res,
        r_res,
        radius[radius.shape[0] // 2].expand_as(r_res),
    )

    kw = dict(left=None, right=None, device=radius.device, dtype=radius.dtype)
    ne_res = interp1d_linear(r_safe, radius, density, **kw)     # (n_E,)
    dne_dr_res = interp1d_linear(r_safe, radius, dne_dr, **kw)  # (n_E,)

    # Density scale height at the resonance in metres:
    #   L_n [m] = |n_e / (dn_e / dr_hat)| * R_sun
    # where r_hat = r / R_sun is dimensionless (stored in solar_profile.radius).
    tiny = torch.finfo(radius.dtype).tiny
    L_n_m = (ne_res.abs() / (dne_dr_res.abs() + tiny)) * constant.R_SUN

    # Adiabaticity parameter at resonance (Giunti & Kim 2007, Eq. 13.46):
    #   gamma_res = dm21 [eV^2] * sin^2(2th12) / cos(2th12)
    #               / (2 * E [MeV] * 1e6 [eV/MeV] * hbarc [eV*m])
    #               * L_n [m]
    # Units: eV^2 / (eV * eV*m) * m = dimensionless. ✓
    hbarc_evm = constant.HBARC_MeV_m * 1e6  # hbar*c in eV*m
    kin_per_m = dm21 / (2.0 * E_1d * 1e6 * hbarc_evm)  # (n_E,) [1/m]

    gamma_res = (kin_per_m * (sin2th12 ** 2 / cos2th12) * L_n_m).clamp(min=0.0)

    p = torch.exp(-0.5 * torch.pi * gamma_res)
    p = torch.where(has_res, p, torch.zeros_like(p))

    return p.squeeze(0) if scalar_in else p
