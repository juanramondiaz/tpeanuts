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
Torch-native solar matter mixing utilities.

This module evaluates the matter-modified solar mixing angles used by the
adiabatic solar-probability approximation. The formulas mirror the original
peanuts matter_mixing.py implementation while preserving torch device and
dtype propagation.

Module functions:
    Vk(...)
        Compute the dimensionless matter-potential ratio for one splitting.
    DeltamSqee(...)
        Compute the effective atmospheric mass splitting Delta m^2_ee.
    th13_M(...)
        Compute the matter-modified theta13 angle.
    th12_M(...)
        Compute the matter-modified theta12 angle.
"""



from __future__ import annotations

from typing import Optional

import torch

import tpeanuts.util.constant as constant
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike, as_tensor_like, first_tensor

from tpeanuts.core.common.potential import kinetic_potential, matter_potential_cc

# ---------------------------------------------------------------------------
# Legacy peanuts precomputed Vk prefactor (peanuts/matter_mixing.py):
#
#     Vk = (3.868e-7 / 2.533) * (ne * E / Deltam2)
#
# Both 3.868e-7 (matter-potential prefactor, see core.common.potential) and
# 2.533 (~= 1 / (2 * hbar*c * 1e12), the kinetic-term conversion factor) are
# independently hardcoded there with 4 significant digits instead of being
# derived from full-precision physical constants. Their compounded rounding
# gives a combined Vk prefactor that differs from the full-precision value by
# a relative ~3.6e-4 — an order of magnitude larger than the matter-potential
# rounding alone. Kept only for bit-comparable validation against legacy.
# ---------------------------------------------------------------------------
_VK_FACTOR_LEGACY: float = 3.868e-7 / 2.533


def Vk(
    Deltam2: TensorLike,
    E: TensorLike,
    ne: TensorLike,
    *,
    antinu: bool | torch.Tensor = False,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compute the dimensionless solar matter-potential ratio.

    Formula:
        V_k = V_mat / K = 2 E hbar c V_cc / Delta m^2.

    Args:
        Deltam2: Mass-squared splitting in eV^2.
        E: Neutrino energy in MeV.
        ne: Electron density in mol/cm^3.
        antinu: If True, flip the matter-potential sign for antineutrinos.
        legacy_precision: If True, use the legacy peanuts combined prefactor
            ``_VK_FACTOR_LEGACY`` (hardcoded to 4 significant digits in
            ``peanuts/matter_mixing.py``) instead of deriving the matter and
            kinetic prefactors separately from full-precision physical
            constants. Intended only for bit-comparable validation against
            legacy peanuts; the two prefactors differ by a relative ~3.6e-4.

    Returns:
        Dimensionless ratio between the charged-current matter potential and
        the kinetic splitting used by the solar mixing formulas.
    """
    reference = first_tensor(E, ne, Deltam2)
    E_t = as_tensor_like(E, reference)
    dm_t = as_tensor_like(Deltam2, E_t)
    ne_t = as_tensor_like(ne, E_t)

    shape = torch.broadcast_shapes(ne_t.shape, E_t.shape, dm_t.shape)
    E_b = torch.broadcast_to(E_t, shape)
    dm_b = torch.broadcast_to(dm_t, shape)
    ne_b = torch.broadcast_to(ne_t, shape)

    if legacy_precision:
        if isinstance(antinu, bool):
            sign = -1.0 if antinu else 1.0
        else:
            antinu_t = antinu.to(device=ne_b.device, dtype=torch.bool)
            while antinu_t.ndim < ne_b.ndim:
                antinu_t = antinu_t.unsqueeze(-1)
            sign = torch.where(
                antinu_t,
                torch.full_like(ne_b, -1.0),
                torch.ones_like(ne_b),
            )
        return sign * _VK_FACTOR_LEGACY * ne_b * E_b / dm_b

    context = RuntimeContext(device=E_b.device, dtype=E_b.dtype)
    V = matter_potential_cc(
        ne_b[..., None],
        antinu=antinu,
        evolution_scale_m=constant.R_E,
        context=context,
    )
    K = kinetic_potential(
        dm_b[..., None],
        E_b,
        evolution_scale_m=constant.R_E,
        context=context,
    )

    return (V / K).squeeze(-1)


def DeltamSqee(oscillation: OscillationParameters) -> torch.Tensor:
    """Compute the effective atmospheric splitting Delta m^2_ee.

    Formula:
        Delta m^2_ee = cos^2(theta12) Delta m^2_31
        + sin^2(theta12) Delta m^2_32.

    Args:
        oscillation: Oscillation parameters supplying theta12 and
            mass_spectrum.DeltamSq21/DeltamSq3l.

    Returns:
        Effective Delta m^2_ee tensor, with ordering selected by the sign of
        DeltamSq3l.
    """
    th12  = oscillation.pmns.params.theta12
    dm21  = oscillation.mass_spectrum.DeltamSq21
    dm3l  = oscillation.mass_spectrum.DeltamSq3l

    dm31 = torch.where(dm3l > 0.0, dm3l, dm3l + dm21)
    dm32 = torch.where(dm3l < 0.0, dm3l, dm3l - dm21)

    return torch.cos(th12) ** 2 * dm31 + torch.sin(th12) ** 2 * dm32


def th13_M(
    oscillation: OscillationParameters,
    E: TensorLike,
    ne: TensorLike,
    *,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compute the matter-modified theta13 mixing angle.

    Formula:
        theta13^M = 1/2 arccos(
            (cos(2 theta13) - V_k(Delta m^2_ee))
            / sqrt((cos(2 theta13) - V_k(Delta m^2_ee))^2
                   + sin^2(2 theta13))
        ).

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13, and
            mass_spectrum.DeltamSq21/DeltamSq3l.
        E: Neutrino energy in MeV.
        ne: Electron density in mol/cm^3.
        legacy_precision: If True, evaluate the internal ``Vk`` call with the
            legacy peanuts combined prefactor for bit-comparable validation.

    Returns:
        Matter-modified theta13 angle in radians.
    """
    th13_t = oscillation.pmns.params.theta13
    E_t    = as_tensor_like(E, th13_t)
    ne_t   = as_tensor_like(ne, E_t)

    vk = Vk(
        DeltamSqee(oscillation),
        E_t,
        ne_t,
        antinu=oscillation.antinu,
        legacy_precision=legacy_precision,
    )
    numerator = torch.cos(2.0 * th13_t) - vk
    denominator = torch.sqrt(numerator**2 + torch.sin(2.0 * th13_t) ** 2)
    arg = torch.clamp(numerator / denominator, min=-1.0, max=1.0)

    # arg is clamped to [-1, 1] above, so arccos(arg) is guaranteed in
    # [0, pi] and 0.5*arccos(arg) already lies in the closed interval
    # [0, pi/2] -- no wrapping is needed. A `torch.remainder(..., pi/2)`
    # used to sit here, but since remainder(pi/2, pi/2) == 0 exactly, it
    # silently mapped the fully-resonant angle pi/2 (denominator == 0, i.e.
    # the matter-dominated limit) to 0 (the vacuum-like limit) -- the
    # opposite of the correct value -- instead of being the no-op it was
    # presumably intended as.
    return 0.5 * torch.arccos(arg)


def th12_M(
    oscillation: OscillationParameters,
    E: TensorLike,
    ne: TensorLike,
    *,
    legacy_precision: bool = False,
    th13m: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the matter-modified theta12 mixing angle.

    Formula:
        theta12^M = 1/2 arccos(
            (cos(2 theta12) - V'_k)
            / sqrt((cos(2 theta12) - V'_k)^2
                   + sin^2(2 theta12) cos^2(theta13^M - theta13))
        ),
        where V'_k = V_k(Delta m^2_21) cos^2(theta13^M)
        + (Delta m^2_ee / Delta m^2_21) sin^2(theta13^M - theta13).

    Args:
        oscillation: Oscillation parameters supplying theta12, theta13, and
            mass_spectrum.DeltamSq21/DeltamSq3l.
        E: Neutrino energy in MeV.
        ne: Electron density in mol/cm^3.
        legacy_precision: If True, evaluate every internal ``Vk``/``th13_M``
            call with the legacy peanuts combined prefactor for
            bit-comparable validation.
        th13m: Optional precomputed ``th13_M(oscillation, E, ne,
            legacy_precision=legacy_precision)`` result. Callers that already
            evaluated ``th13_M`` for the same ``(E, ne)`` grid can pass it
            here to avoid recomputing it. When omitted, it is computed
            internally as before.

    Returns:
        Matter-modified theta12 angle in radians.
    """
    th12_t = oscillation.pmns.params.theta12
    th13_t = oscillation.pmns.params.theta13
    E_t    = as_tensor_like(E, th12_t)
    ne_t   = as_tensor_like(ne, E_t)

    if th13m is None:
        th13m = th13_M(oscillation, E_t, ne_t, legacy_precision=legacy_precision)
    dm21  = oscillation.mass_spectrum.DeltamSq21
    dm_ee = DeltamSqee(oscillation)

    vk_prime = (
        Vk(
            dm21,
            E_t,
            ne_t,
            antinu=oscillation.antinu,
            legacy_precision=legacy_precision,
        ) * torch.cos(th13m) ** 2
        + dm_ee / dm21 * torch.sin(th13m - th13_t) ** 2
    )

    numerator = torch.cos(2.0 * th12_t) - vk_prime
    denominator = torch.sqrt(
        numerator**2 + torch.sin(2.0 * th12_t) ** 2 * torch.cos(th13m - th13_t) ** 2
    )
    arg = torch.clamp(numerator / denominator, min=-1.0, max=1.0)

    # arg is clamped to [-1, 1] above, so arccos(arg) is guaranteed in
    # [0, pi] and 0.5*arccos(arg) already lies in the closed interval
    # [0, pi/2] -- no wrapping is needed. A `torch.remainder(..., pi/2)`
    # used to sit here, but since remainder(pi/2, pi/2) == 0 exactly, it
    # silently mapped the fully-resonant angle pi/2 (denominator == 0, i.e.
    # the matter-dominated limit) to 0 (the vacuum-like limit) -- the
    # opposite of the correct value -- instead of being the no-op it was
    # presumably intended as.
    return 0.5 * torch.arccos(arg)
