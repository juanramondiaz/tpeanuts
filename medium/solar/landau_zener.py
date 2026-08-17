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

"""Local Landau--Zener correction for the solar 1--2 MSW resonance.

The Parke approximation uses ``P_LZ = exp(-pi*gamma_res/2)`` with the density
scale height evaluated at resonance. Its position includes the effective
theta13 correction used by ``matter_mixing.th12_M``; ``gamma_res`` retains
the standard two-level form.

The calculation uses the full structural density grid. It does not describe
generic NSI, sterile multi-level crossings, the 1--3 resonance or general
nonlinear-profile corrections. If no 1--2 resonance is crossed inside the
Sun, including much of the standard pp spectrum, ``P_LZ`` is zero.
"""



from __future__ import annotations

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.matter_mixing import DeltamSqee, Vk, th13_M
from tpeanuts.util.math import interp1d_linear
from tpeanuts.util.type import TensorLike


def _full_density_grid(medium: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the profile's full structural radius/density grid."""
    return medium.radius, medium.density


def density_gradient(medium: object) -> torch.Tensor:
    """Compute dn_e/d(r_hat) on the solar profile's full density grid.

    Uses central differences at interior points and one-sided differences
    at the two boundary points. The coordinate r_hat = r/R_sun is the
    dimensionless solar radius (see ``_full_density_grid``).

    Args:
        medium: SolarMediumProfile-like object exposing ``radius``/
            ``density`` 1-D tensors of matching length.

    Returns:
        Tensor of shape ``(n_r,)`` with dn_e/dr_hat in mol/cm^3 per R_sun,
        on the same grid returned by ``_full_density_grid``.
    """
    r, ne = _full_density_grid(medium)  # (n_r,), (n_r,)

    # Central differences at interior nodes
    interior = (ne[2:] - ne[:-2]) / (r[2:] - r[:-2])  # (n_r - 2,)

    # One-sided differences at the boundaries
    left = (ne[1:2] - ne[:1]) / (r[1:2] - r[:1])        # (1,)
    right = (ne[-1:] - ne[-2:-1]) / (r[-1:] - r[-2:-1]) # (1,)

    return torch.cat([left, interior, right], dim=0)  # (n_r,)


def resonance_radius(
    oscillation: OscillationParameters,
    E: TensorLike,
    medium: object,
    *,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Locate the solar 1--2 MSW resonance for each energy.

    The resonance satisfies ``V'_k = cos(2*theta12)``, using the same
    theta13-corrected potential as ``matter_mixing.th12_M``. Its radius is
    interpolated across the first outward sign change on the structural grid.

    Args:
        oscillation: Oscillation parameters supplying theta_12, theta_13, and
            mass_spectrum.DeltamSq21/DeltamSq3l.
        E: Neutrino energy or 1-D energy grid in MeV.
        medium: SolarMediumProfile-like object with ``radius``/``density``
            1-D tensors.
        legacy_precision: If True, evaluate the internal ``Vk``/``th13_M``
            calls with the legacy peanuts combined prefactor for
            bit-comparable validation (see ``matter_mixing.Vk``).

    Returns:
        Tensor matching the shape of ``E`` with the resonance radius in solar
        radius units. Entries are NaN for energies without a resonance in the
        solar volume (below threshold, LMA-Dark parameters, antineutrinos,
        etc.).
    """
    th12 = oscillation.pmns.params.theta12
    th13 = oscillation.pmns.params.theta13
    dm21 = oscillation.mass_spectrum.DeltamSq21
    dm_ee = DeltamSqee(oscillation)
    cos2th12 = torch.cos(2.0 * th12)  # scalar; negative for LMA-Dark

    radius, density = _full_density_grid(medium)  # (n_r,), (n_r,)

    E_t = torch.as_tensor(E, device=radius.device, dtype=radius.dtype)
    scalar_in = E_t.ndim == 0
    E_1d = E_t.reshape(-1)   # (n_E,)
    n_E = E_1d.shape[0]

    energy_grid = E_1d[:, None]    # (n_E, 1)
    density_grid = density[None, :]  # (1, n_r)

    # Full effective potential V'_k, matching th12_M's resonance condition
    # (broadcasts to (n_E, n_r)).
    th13m = th13_M(oscillation, energy_grid, density_grid, legacy_precision=legacy_precision)
    vk = Vk(dm21, energy_grid, density_grid, antinu=oscillation.antinu, legacy_precision=legacy_precision)
    vk_prime = vk * torch.cos(th13m) ** 2 + dm_ee / dm21 * torch.sin(th13m - th13) ** 2

    # Distance from resonance condition (positive inside resonance, negative outside)
    diff = vk_prime - cos2th12  # (n_E, n_r)

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
    medium: object,
    *,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Evaluate the local Parke/Landau--Zener crossing probability.

    Uses ``P_LZ = exp(-pi*gamma_res/2)`` and the density scale height at the
    1--2 resonance. This is a locally linear, two-level approximation rather
    than a general solution for arbitrary density profiles.

    P_LZ is set to 0 (fully adiabatic) wherever no resonance exists in the
    solar volume — including below-threshold energies, LMA-Dark parameters,
    and antineutrinos.

    Args:
        oscillation: Oscillation parameters supplying theta_12, theta_13, and
            mass_spectrum.DeltamSq21/DeltamSq3l.
        E: Neutrino energy or 1-D energy grid in MeV.
        medium: SolarMediumProfile-like object exposing ``radius``/
            ``density``.
        legacy_precision: If True, evaluate the internal ``resonance_radius``
            call (and its ``Vk``/``th13_M`` evaluations) with the legacy
            peanuts combined prefactor for bit-comparable validation.

    Returns:
        Tensor matching the shape of ``E`` with P_LZ in [0, 1]. For standard
        LMA parameters at solar-neutrino energies, the returned values are
        numerically indistinguishable from zero.
    """
    th12 = oscillation.pmns.params.theta12
    dm21 = oscillation.mass_spectrum.DeltamSq21
    sin2th12 = torch.sin(2.0 * th12)
    cos2th12 = torch.cos(2.0 * th12)

    radius, density = _full_density_grid(medium)

    E_t = torch.as_tensor(E, device=radius.device, dtype=radius.dtype)
    scalar_in = E_t.ndim == 0
    E_1d = E_t.reshape(-1)  # (n_E,)

    dne_dr = density_gradient(medium)                       # (n_r,)
    r_res = resonance_radius(
        oscillation, E_1d, medium, legacy_precision=legacy_precision,
    )  # (n_E,), NaN if absent
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
    # where r_hat = r / R_sun is dimensionless (stored in medium.radius).
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


def landau_zener_spatial_correction(
    oscillation: OscillationParameters,
    E: TensorLike,
    medium: object,
    radius_samples: torch.Tensor,
    *,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Broadcast the local Landau-Zener probability across production radii.

    Applies ``plz(E)`` only to production radii above the 1--2 resonance
    density (``r_prod < r_res(E)``); production points below the resonance
    (or energies with no resonance in the solar volume) get a zero
    correction. This is the (E, r_prod) mask ``mass_weights_adiabatic_
    approximated``'s ``p_lz`` argument expects.

    Args:
        oscillation: Oscillation parameters supplying theta_12, theta_13, and
            mass_spectrum.DeltamSq21/DeltamSq3l.
        E: Neutrino energy, scalar or 1-D grid in MeV.
        medium: SolarMediumProfile-like object exposing the full structural
            ``radius``/``density`` grid (see ``resonance_radius``/``plz``).
        radius_samples: Production radii in solar-radius units, shaped
            ``(n_r,)``.
        legacy_precision: If True, evaluate the internal ``resonance_radius``/
            ``plz`` calls with the legacy peanuts combined prefactor for
            bit-comparable validation.

    Returns:
        Tensor shaped ``(n_r,)`` for scalar ``E``, or ``(n_E, n_r)`` for a
        1-D energy grid, with the spatially resolved crossing-probability
        correction in [0, 1].
    """
    E_t = torch.as_tensor(E, device=radius_samples.device, dtype=radius_samples.dtype)
    E_1d = E_t.reshape(-1) if E_t.ndim == 1 else E_t.reshape(1)  # (n_E,)

    r_res = resonance_radius(
        oscillation, E_1d, medium, legacy_precision=legacy_precision,
    )  # (n_E,) NaN if absent
    p_lz_e = plz(
        oscillation, E_1d, medium, legacy_precision=legacy_precision,
    )  # (n_E,)

    # above_res[e, r]: True where r_prod < r_res(E) (above resonance
    # density). NaN comparisons evaluate to False, so energies without a
    # resonance contribute a zero mask automatically.
    above_res = radius_samples[None, :] < r_res[:, None]          # (n_E, n_r)
    p_lz_2d = p_lz_e[:, None] * above_res.to(dtype=p_lz_e.dtype)  # (n_E, n_r)

    return p_lz_2d.squeeze(0) if E_t.ndim == 0 else p_lz_2d
