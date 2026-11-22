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

"""Local Landau-Zener correction for the adiabatic 1-2 MSW resonance.

The Parke approximation uses ``P_LZ = exp(-pi*gamma_res/2)`` with the
density scale height evaluated at resonance. Its position includes the
effective theta13 correction used by ``matter_mixing.theta12_in_matter``;
``gamma_res`` retains the standard two-level form.

Moved here from ``tpeanuts.medium.solar.landau_zener`` -- this
calculation was already medium-agnostic in every way except its own
argument type: it originally took a whole
``tpeanuts.medium.solar.solar.SolarMedium``, which is exactly
backwards for a module living under ``tpeanuts.physics`` (``physics``
must never import from ``tpeanuts.medium`` -- the reverse of the
already-established "``medium`` never imports
``tpeanuts.physics.model.PropagationModel``" rule, equally important).
Every function here now takes the density grid as plain ``radius``/
``density`` tensors instead -- the caller (``tpeanuts.pipeline.solar``)
passes ``SolarMedium.radius``/``.density`` in directly; nothing
about the calculation itself changes.

Uses the full structural density grid it is given. Does not describe
generic NSI, sterile multi-level crossings, the 1-3 resonance, or
general nonlinear-profile corrections. If no 1-2 resonance is crossed,
including much of the standard solar pp spectrum, ``P_LZ`` is zero.

Ported from v1_0's ``medium/solar/landau_zener.py``, translated to this
rewrite's ``MixingMatrix``/``MassSpectrum`` signatures. v1_0's
``legacy_precision`` toggle is not ported, same reasoning as
``matter_mixing``/``adiabatic``. ``interp1d_linear`` (``tpeanuts.util.math``
in v1_0, no equivalent in this rewrite) is replaced by a small
module-private helper, keeping v1_0's exact "clamp to the boundary value
outside the tabulated domain" convention (``left``/``right``, default
``fp[0]``/``fp[-1]``) -- a resonance radius is occasionally exactly at a
grid boundary from floating-point rounding, and clamping (not raising)
is the physically sensible fallback there.

Module contents:
    density_gradient(...)
        dn_e/d(r_hat) on a solar-like structural grid.
    resonance_radius(...)
        Locate the 1-2 MSW resonance for each energy.
    landau_zener_probability(...)
        Local Parke/Landau-Zener crossing probability at resonance.
    landau_zener_spatial_correction(...)
        Broadcast ``landau_zener_probability`` across production radii -- the ``(E, r_prod)``
        mask ``adiabatic.mass_weights_adiabatic_approximated``'s ``p_lz``
        argument expects.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

import tpeanuts.util.constant as constant
from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.evolution.adiabatic.matter_mixing import DeltamSqee, Vk, theta13_in_matter
from tpeanuts.util.math import searchsorted_contiguous
from tpeanuts.util.tensor import TensorLike, as_tensor


def _interp1d_linear(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    *,
    left: Optional[torch.Tensor] = None,
    right: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linearly interpolate one-dimensional tabulated data, clamped outside ``xp``'s domain.

    Args:
        x: Query points, any shape.
        xp: Strictly increasing 1-D grid the table is defined on.
        fp: Table values at ``xp``, shape ``xp.shape``.
        left: Value returned where ``x < xp[0]``. None defaults to ``fp[0]``.
        right: Value returned where ``x > xp[-1]``. None defaults to ``fp[-1]``.

    Returns:
        Interpolated values, shaped like ``x``.
    """
    left = fp[0] if left is None else left
    right = fp[-1] if right is None else right

    n = xp.shape[0]
    idx_right = searchsorted_contiguous(xp, x, right=False).clamp(1, n - 1)
    idx_left = idx_right - 1
    x0, x1 = xp[idx_left], xp[idx_right]
    f0, f1 = fp[idx_left], fp[idx_right]
    frac = (x - x0) / (x1 - x0).clamp_min(torch.finfo(xp.dtype).eps)
    interpolated = f0 + frac * (f1 - f0)

    return torch.where(x < xp[0], left, torch.where(x > xp[-1], right, interpolated))


def density_gradient(radius: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """Compute dn_e/d(r_hat) on a solar-like structural density grid.

    Uses central differences at interior points and one-sided differences
    at the two boundary points. The coordinate r_hat is whatever
    dimensionless radius ``radius`` is expressed in (e.g. ``r / R_sun``).

    Args:
        radius: Strictly increasing 1-D dimensionless radius grid.
        density: Electron density at each ``radius``, same shape.

    Returns:
        Tensor of shape ``(n_r,)`` with dn_e/dr_hat, on ``radius``'s own
        grid.
    """
    interior = (density[2:] - density[:-2]) / (radius[2:] - radius[:-2])

    left = (density[1:2] - density[:1]) / (radius[1:2] - radius[:1])
    right = (density[-1:] - density[-2:-1]) / (radius[-1:] - radius[-2:-1])

    return torch.cat([left, interior, right], dim=0)


def resonance_radius(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    radius: torch.Tensor,
    density: torch.Tensor,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """Locate the 1-2 MSW resonance for each energy.

    The resonance satisfies ``V'_k = cos(2*theta12)``, using the same
    theta13-corrected potential as ``matter_mixing.theta12_in_matter``. Its radius
    is interpolated across the first outward sign change on the
    structural grid.

    Args:
        mixing: Built MixingMatrix supplying theta12/theta13.
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.
        E_MeV: Neutrino energy or 1-D energy grid in MeV.
        radius: Strictly increasing 1-D dimensionless radius grid.
        density: Electron density at each ``radius``, same shape.
        antinu: If True, flip the matter-potential sign for antineutrinos.

    Returns:
        Tensor matching the shape of ``E_MeV`` with the resonance radius,
        in ``radius``'s own units. Entries are NaN for energies without a
        resonance on the grid (below threshold, LMA-Dark parameters,
        antineutrinos, etc.).
    """
    th12 = mixing.standard.theta12
    th13 = mixing.standard.theta13
    dm21 = mass_spectrum.DeltamSq21
    dm_ee = DeltamSqee(mixing, mass_spectrum)
    cos2th12 = torch.cos(2.0 * th12)

    E_t = as_tensor(E_MeV, device=radius.device, dtype=radius.dtype)
    scalar_in = E_t.ndim == 0
    E_1d = E_t.reshape(-1)
    n_E = E_1d.shape[0]

    energy_grid = E_1d[:, None]
    density_grid = density[None, :]

    th13m = theta13_in_matter(mixing, mass_spectrum, energy_grid, density_grid, antinu=antinu)
    vk = Vk(dm21, energy_grid, density_grid, antinu=antinu)
    vk_prime = vk * torch.cos(th13m) ** 2 + dm_ee / dm21 * torch.sin(th13m - th13) ** 2

    diff = vk_prime - cos2th12

    crossing = (diff[:, :-1] > 0) & (diff[:, 1:] <= 0)
    has_res = crossing.any(dim=-1)

    idx = crossing.long().argmax(dim=-1).clamp(0, radius.shape[0] - 2)
    batch = torch.arange(n_E, device=radius.device)

    r0 = radius[idx]
    r1 = radius[idx + 1]
    d0 = diff[batch, idx]
    d1 = diff[batch, idx + 1]

    denom = d0 - d1
    safe_denom = torch.where(denom.abs() > 0, denom, torch.ones_like(denom))
    frac = (d0 / safe_denom).clamp(0.0, 1.0)
    r_res = r0 + frac * (r1 - r0)

    r_res = torch.where(has_res, r_res, torch.full_like(r_res, float("nan")))

    return r_res.squeeze(0) if scalar_in else r_res


def landau_zener_probability(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    radius: torch.Tensor,
    density: torch.Tensor,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """Evaluate the local Parke/Landau-Zener crossing probability.

    Uses ``P_LZ = exp(-pi*gamma_res/2)`` and the density scale height at
    the 1-2 resonance. This is a locally linear, two-level approximation
    rather than a general solution for arbitrary density profiles.

    P_LZ is set to 0 (fully adiabatic) wherever no resonance exists on
    the grid -- including below-threshold energies, LMA-Dark parameters,
    and antineutrinos.

    Args:
        mixing: Built MixingMatrix supplying theta12/theta13.
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.
        E_MeV: Neutrino energy or 1-D energy grid in MeV.
        radius: Strictly increasing 1-D dimensionless radius grid.
        density: Electron density at each ``radius``, same shape.
        antinu: If True, flip the matter-potential sign for antineutrinos.

    Returns:
        Tensor matching the shape of ``E_MeV`` with P_LZ in [0, 1]. For
        standard LMA parameters at solar-neutrino energies, the returned
        values are numerically indistinguishable from zero.
    """
    th12 = mixing.standard.theta12
    dm21 = mass_spectrum.DeltamSq21
    sin2th12 = torch.sin(2.0 * th12)
    cos2th12 = torch.cos(2.0 * th12)

    E_t = as_tensor(E_MeV, device=radius.device, dtype=radius.dtype)
    scalar_in = E_t.ndim == 0
    E_1d = E_t.reshape(-1)

    dne_dr = density_gradient(radius, density)
    r_res = resonance_radius(mixing, mass_spectrum, E_1d, radius, density, antinu=antinu)
    has_res = torch.isfinite(r_res)

    r_safe = torch.where(
        has_res,
        r_res,
        radius[radius.shape[0] // 2].expand_as(r_res),
    )

    ne_res = _interp1d_linear(r_safe, radius, density)
    dne_dr_res = _interp1d_linear(r_safe, radius, dne_dr)

    # Density scale height at the resonance in metres:
    #   L_n [m] = |n_e / (dn_e / dr_hat)| * R_sun
    # where r_hat is the dimensionless radius ``radius`` is expressed in.
    tiny = torch.finfo(radius.dtype).tiny
    L_n_m = (ne_res.abs() / (dne_dr_res.abs() + tiny)) * constant.R_SUN

    # Adiabaticity parameter at resonance (Giunti & Kim 2007, Eq. 13.46):
    #   gamma_res = dm21 [eV^2] * sin^2(2th12) / cos(2th12)
    #               / (2 * E [MeV] * 1e6 [eV/MeV] * hbarc [eV*m])
    #               * L_n [m]
    # Units: eV^2 / (eV * eV*m) * m = dimensionless.
    hbarc_evm = constant.HBARC_MeV_m * 1e6
    kin_per_m = dm21 / (2.0 * E_1d * 1e6 * hbarc_evm)

    gamma_res = (kin_per_m * (sin2th12**2 / cos2th12) * L_n_m).clamp(min=0.0)

    p = torch.exp(-0.5 * torch.pi * gamma_res)
    p = torch.where(has_res, p, torch.zeros_like(p))

    return p.squeeze(0) if scalar_in else p


def landau_zener_spatial_correction(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    radius: torch.Tensor,
    density: torch.Tensor,
    radius_samples: torch.Tensor,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """Broadcast the local Landau-Zener probability across production radii.

    Applies ``landau_zener_probability(E)`` only to production radii above the 1-2 resonance
    density (``r_prod < r_res(E)``); production points below the
    resonance (or energies with no resonance on the grid) get a zero
    correction. This is the ``(E, r_prod)`` mask
    ``adiabatic.mass_weights_adiabatic_approximated``'s ``p_lz`` argument
    expects.

    Args:
        mixing: Built MixingMatrix supplying theta12/theta13.
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.
        E_MeV: Neutrino energy, scalar or 1-D grid in MeV.
        radius: Strictly increasing 1-D dimensionless structural radius
            grid.
        density: Electron density at each ``radius``, same shape.
        radius_samples: Production radii, in the same dimensionless
            units as ``radius``, shaped ``(n_r,)`` -- e.g.
            ``tpeanuts.source.solar.SolarSource.production_radius``.
        antinu: If True, flip the matter-potential sign for antineutrinos.

    Returns:
        Tensor shaped ``(n_r,)`` for scalar ``E_MeV``, or ``(n_E, n_r)``
        for a 1-D energy grid, with the spatially resolved
        crossing-probability correction in [0, 1].
    """
    E_t = as_tensor(E_MeV, device=radius_samples.device, dtype=radius_samples.dtype)
    E_1d = E_t.reshape(-1) if E_t.ndim == 1 else E_t.reshape(1)

    r_res = resonance_radius(mixing, mass_spectrum, E_1d, radius, density, antinu=antinu)
    p_lz_e = landau_zener_probability(mixing, mass_spectrum, E_1d, radius, density, antinu=antinu)

    # above_res[e, r]: True where r_prod < r_res(E) (above resonance
    # density). NaN comparisons evaluate to False, so energies without a
    # resonance contribute a zero mask automatically.
    above_res = radius_samples[None, :] < r_res[:, None]
    p_lz_2d = p_lz_e[:, None] * above_res.to(dtype=p_lz_e.dtype)

    return p_lz_2d.squeeze(0) if E_t.ndim == 0 else p_lz_2d
