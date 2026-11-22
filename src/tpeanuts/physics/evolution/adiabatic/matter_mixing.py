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

"""Closed-form matter-mixing angles (MSW), 3-flavour Standard Model only.

Originally built as part of the solar-neutrino migration (ported from
v1_0's ``medium/solar/matter_mixing.py``) and moved here from
``tpeanuts.medium.solar.matter_mixing``: these formulas are not
solar-specific physics at all -- they are a general adiabatic
matter-mixing technique that happens, in this project, to only be
consumed by the solar propagation path (``tpeanuts.pipeline.solar``).
Living under ``tpeanuts.physics.evolution`` keeps every propagation
*algorithm* in one place regardless of which medium uses it, mirroring
``tpeanuts.physics.evolution.numerical``/``analytic``/``perturbative`` (each its own folder):
``tpeanuts.medium.*`` stays pure density/geometry data, and
``tpeanuts.pipeline.*`` stays pure source/medium/method composition --
see ``tpeanuts.physics.evolution.adiabatic``'s own package docstring for
the full reasoning.

``Vk``/``DeltamSqee``/``theta13_in_matter``/``theta12_in_matter`` only ever need a built
``MixingMatrix``'s ``standard`` angles and a ``MassSpectrum``'s
splittings -- never an extension, a dissipator, or the composition root
(``tpeanuts.physics.model.PropagationModel``) itself.

v1_0's ``legacy_precision`` toggle (a hardcoded 4-significant-digit
``Vk`` prefactor, ``_VK_FACTOR_LEGACY``) is not ported: its only
consumer in v1_0 is ``medium/solar/validation.py``, a bit-comparable
check against the pre-v1_0 legacy ``peanuts`` package, which is not
being ported (this rewrite verifies against v1_0 itself instead). The
formulas below are v1_0's non-legacy branch, unchanged.

Module contents:
    Vk(...)
        Compute the dimensionless matter-potential ratio for one splitting.
    DeltamSqee(...)
        Compute the effective atmospheric mass splitting Delta m^2_ee.
    theta13_in_matter(...)
        Compute the matter-modified theta13 angle.
    theta12_in_matter(...)
        Compute the matter-modified theta12 angle.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tpeanuts.physics.common.mass_spectrum import MassSpectrum
from tpeanuts.physics.common.mixing import MixingMatrix
from tpeanuts.physics.common.potential import kinetic_potential, matter_potential_cc
from tpeanuts.util.tensor import TensorLike, as_tensor, infer_device_dtype


def Vk(
    Deltam2_eV2: TensorLike,
    E_MeV: TensorLike,
    ne_mol_cm3: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """Compute the dimensionless solar matter-potential ratio.

    Formula:
        V_k = V_mat / K = 2 E hbar c V_cc / Delta m^2.

    ``evolution_scale_m`` is left at its shared default
    (``tpeanuts.util.constant.R_E``) for both the matter and kinetic
    potentials: since ``Vk`` is a pure ratio, any positive length scale
    cancels exactly as long as the same one is used for both.

    Args:
        Deltam2_eV2: Mass-squared splitting in eV^2.
        E_MeV: Neutrino energy in MeV.
        ne_mol_cm3: Electron density in mol/cm^3.
        antinu: If True, flip the matter-potential sign for antineutrinos.

    Returns:
        Dimensionless ratio between the charged-current matter potential and
        the kinetic splitting used by the solar mixing formulas.
    """
    device, dtype = infer_device_dtype(E_MeV, ne_mol_cm3, Deltam2_eV2)
    E_t = as_tensor(E_MeV, device=device, dtype=dtype)
    ne_t = as_tensor(ne_mol_cm3, device=device, dtype=dtype)
    dm_t = as_tensor(Deltam2_eV2, device=device, dtype=dtype)

    shape = torch.broadcast_shapes(E_t.shape, ne_t.shape, dm_t.shape)
    E_b = torch.broadcast_to(E_t, shape)
    ne_b = torch.broadcast_to(ne_t, shape)
    dm_b = torch.broadcast_to(dm_t, shape)

    # kinetic_potential treats mSq_eV2's own last axis as a flavour axis
    # and unsqueezes E to match it -- a scalar-per-point splitting (not a
    # flavour vector) needs an explicit trailing singleton axis here, or
    # E_b's own unsqueeze(-1) broadcasts against dm_b's *unexpanded* last
    # axis instead of matching it elementwise, giving a cross ``(N, N)``
    # shape instead of the intended ``(N,)``. Both V and K get the same
    # treatment so the final ``squeeze(-1)`` undoes it cleanly for both.
    V = matter_potential_cc(ne_b[..., None], antinu=antinu)
    K = kinetic_potential(dm_b[..., None], E_b)

    return (V / K).squeeze(-1)


def DeltamSqee(mixing: MixingMatrix, mass_spectrum: MassSpectrum) -> torch.Tensor:
    """Compute the effective atmospheric splitting Delta m^2_ee.

    Formula:
        Delta m^2_ee = cos^2(theta12) Delta m^2_31
        + sin^2(theta12) Delta m^2_32.

    Args:
        mixing: Built MixingMatrix supplying theta12.
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.

    Returns:
        Effective Delta m^2_ee tensor, with ordering selected by the sign of
        DeltamSq3l.
    """
    th12 = mixing.standard.theta12
    dm21 = mass_spectrum.DeltamSq21
    dm3l = mass_spectrum.DeltamSq3l

    dm31 = torch.where(dm3l > 0.0, dm3l, dm3l + dm21)
    dm32 = torch.where(dm3l < 0.0, dm3l, dm3l - dm21)

    return torch.cos(th12) ** 2 * dm31 + torch.sin(th12) ** 2 * dm32


def theta13_in_matter(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    ne_mol_cm3: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
) -> torch.Tensor:
    """Compute the matter-modified theta13 mixing angle.

    Formula:
        theta13^M = 1/2 arccos(
            (cos(2 theta13) - V_k(Delta m^2_ee))
            / sqrt((cos(2 theta13) - V_k(Delta m^2_ee))^2
                   + sin^2(2 theta13))
        ).

    Args:
        mixing: Built MixingMatrix supplying theta12/theta13.
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.
        E_MeV: Neutrino energy in MeV.
        ne_mol_cm3: Electron density in mol/cm^3.
        antinu: If True, flip the matter-potential sign for antineutrinos.

    Returns:
        Matter-modified theta13 angle in radians.
    """
    th13_t = mixing.standard.theta13
    E_t = as_tensor(E_MeV, device=th13_t.device, dtype=th13_t.dtype)
    ne_t = as_tensor(ne_mol_cm3, device=E_t.device, dtype=E_t.dtype)

    vk = Vk(DeltamSqee(mixing, mass_spectrum), E_t, ne_t, antinu=antinu)
    numerator = torch.cos(2.0 * th13_t) - vk
    denominator = torch.sqrt(numerator**2 + torch.sin(2.0 * th13_t) ** 2)
    arg = torch.clamp(numerator / denominator, min=-1.0, max=1.0)

    # arg is clamped to [-1, 1] above, so arccos(arg) is guaranteed in
    # [0, pi] and 0.5*arccos(arg) already lies in the closed interval
    # [0, pi/2] -- no wrapping is needed. v1_0 once had a
    # `torch.remainder(..., pi/2)` here, but since remainder(pi/2, pi/2)
    # == 0 exactly, it silently mapped the fully-resonant angle pi/2
    # (denominator == 0, i.e. the matter-dominated limit) to 0 (the
    # vacuum-like limit) -- the opposite of the correct value -- instead
    # of being the no-op it was presumably intended as. Ported without
    # it, matching v1_0's own already-fixed version.
    return 0.5 * torch.arccos(arg)


def theta12_in_matter(
    mixing: MixingMatrix,
    mass_spectrum: MassSpectrum,
    E_MeV: TensorLike,
    ne_mol_cm3: TensorLike,
    *,
    antinu: Union[bool, torch.Tensor] = False,
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
        mixing: Built MixingMatrix supplying theta12/theta13.
        mass_spectrum: Built MassSpectrum supplying DeltamSq21/DeltamSq3l.
        E_MeV: Neutrino energy in MeV.
        ne_mol_cm3: Electron density in mol/cm^3.
        antinu: If True, flip the matter-potential sign for antineutrinos.
        th13m: Optional precomputed ``theta13_in_matter(mixing, mass_spectrum, E_MeV,
            ne_mol_cm3, antinu=antinu)`` result. Callers that already
            evaluated ``theta13_in_matter`` for the same ``(E, ne)`` grid can pass it
            here to avoid recomputing it. When omitted, it is computed
            internally.

    Returns:
        Matter-modified theta12 angle in radians.
    """
    th12_t = mixing.standard.theta12
    th13_t = mixing.standard.theta13
    E_t = as_tensor(E_MeV, device=th12_t.device, dtype=th12_t.dtype)
    ne_t = as_tensor(ne_mol_cm3, device=E_t.device, dtype=E_t.dtype)

    if th13m is None:
        th13m = theta13_in_matter(mixing, mass_spectrum, E_t, ne_t, antinu=antinu)
    dm21 = mass_spectrum.DeltamSq21
    dm_ee = DeltamSqee(mixing, mass_spectrum)

    vk_prime = (
        Vk(dm21, E_t, ne_t, antinu=antinu) * torch.cos(th13m) ** 2
        + dm_ee / dm21 * torch.sin(th13m - th13_t) ** 2
    )

    numerator = torch.cos(2.0 * th12_t) - vk_prime
    denominator = torch.sqrt(
        numerator**2 + torch.sin(2.0 * th12_t) ** 2 * torch.cos(th13m - th13_t) ** 2
    )
    arg = torch.clamp(numerator / denominator, min=-1.0, max=1.0)

    # See theta13_in_matter's own comment: arg is clamped to [-1, 1], so no
    # remainder-based wrapping is needed or applied here either.
    return 0.5 * torch.arccos(arg)
