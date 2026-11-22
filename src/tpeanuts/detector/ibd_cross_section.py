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

"""Interaction cross sections, reusable independently of any one detector's own wiring.

The inverse beta decay (IBD, ``nu_e_bar + p -> e+ + n``) reaction --
``tpeanuts.detector.ibd.IBDDetector``'s own reaction, and the natural
pairing with ``tpeanuts.source.reactor.reactor.ReactorSource``'s pure
electron-antineutrino flux -- at two levels of approximation, both ported
from Vogel & Beacom, Phys. Rev. D60, 053003 (1999) [VB99]:

    ``ibd_cross_section``/``ibd_cross_section_grid``
        The zeroth-order (no-recoil) approximation, Eqs. (10)-(11):
        ``sigma(E_nu) ~= sigma0 * E_e * p_e``, neglecting order-1/M recoil,
        weak magnetism, and angular dependence.
    ``ibd_differential_cross_section_precise``/``ibd_cross_section_grid_precise``
        The order-1/M formula, Eqs. (13)-(15), integrated numerically over
        the positron angle -- what a real detector's own event-rate chain
        (e.g. Daya Bay) needs, since the zeroth-order approximation's few
        -percent bias is not negligible at that precision.

Both grid functions build the visible *prompt* energy
``E_prompt = T_e+ + 2*m_e = E_e + m_e`` (the annihilation photons
contribute the extra ``2*m_e``), not the raw positron kinetic energy.

Separated from ``tpeanuts.detector.ibd`` because a cross section is
reusable physics, independent of how any one concrete detector composes it
into an event rate -- the same separation ``tpeanuts.physics.common.
potential`` (bare formulas) already keeps from ``tpeanuts.physics.model``
(composition).

Units: every cross section here (and in ``neutrino_electron``/``deuteron``)
is returned in units of ``constant.SIGMA_UNIT_CM2`` = 1e-42 cm^2 (``dsigma/dT``
in 1e-42 cm^2/MeV), because the physical values (1e-42 and below) are
subnormal or zero in float32. Pair them with
``tpeanuts.detector.common.target.reduced_target_count``.

Module contents:
    ibd_cross_section(...)
        Zeroth-order (no-recoil) total IBD cross section, in 1e-42 cm^2.
    ibd_cross_section_grid(...)
        Zeroth-order differential cross section dsigma/dT on an
        ``(E_nu_grid, T_grid)`` pair, as a triangular kernel.
    ibd_differential_cross_section_precise(...)
        Order-1/M angular differential cross section dsigma/dcos(theta).
    ibd_cross_section_grid_precise(...)
        Order-1/M differential cross section dsigma/dT, by numerical
        integration of ``ibd_differential_cross_section_precise`` over cos(theta).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from tpeanuts.detector.common.response import scatter_add_linear
from tpeanuts.util import constant

# Cabibbo angle cos(theta_C) and the (energy-independent) inner radiative
# correction Delta_inner^R, both from VB99, used only to compute the bare
# sigma0 of VB99 Eq. (9) for the order-1/M formula below -- the zeroth
# -order ibd_cross_section instead uses the already-combined
# constant.SIGMA0_IBD_CM2_PER_MEV2 (VB99 Eq. 11).
_COS_THETA_CABIBBO = 0.974
_DELTA_INNER_RADIATIVE = 0.024


def _sigma0_bare_units_per_mev4() -> float:
    """The coupling-independent order-1/M cross-section coefficient, in SIGMA_UNIT_CM2/MeV^4 (VB99 Eq. 9)."""
    hbarc_MeV_cm = constant.HBARC_MeV_m * 100.0
    sigma0_natural = (
        constant.G_F_MEV_M2 ** 2 * _COS_THETA_CABIBBO ** 2 / math.pi * (1.0 + _DELTA_INNER_RADIATIVE)
    )
    return sigma0_natural * hbarc_MeV_cm ** 2 / constant.SIGMA_UNIT_CM2


_SIGMA0_BARE_UNITS_PER_MEV4: float = _sigma0_bare_units_per_mev4()


def ibd_cross_section(energy_MeV: torch.Tensor) -> torch.Tensor:
    """Zeroth-order (no-recoil) total IBD cross section, in units of 1e-42 cm^2.

    ``sigma(E_nu) / SIGMA_UNIT_CM2 = (SIGMA0_IBD_CM2_PER_MEV2 / SIGMA_UNIT_CM2) * E_e * p_e`` for
    ``E_nu > IBD_THRESHOLD_MEV``, else 0, with ``E_e = E_nu - DELTA_NP_MEV``
    and ``p_e = sqrt(max(E_e^2 - M_ELECTRON_MEV^2, 0))``.

    Note: in raw cm^2 this is ~1e-43 to 1e-42 for reactor-relevant energies,
    deep in float32's *subnormal* range (smallest normal ~1.18e-38; CUDA flushes
    subnormals to zero), which is why it is returned in units of
    ``constant.SIGMA_UNIT_CM2`` (O(1) values, normal in float32).

    Args:
        energy_MeV: Antineutrino energy, in MeV, any shape.

    Returns:
        Real tensor, same shape as ``energy_MeV``, zero below
        ``tpeanuts.util.constant.IBD_THRESHOLD_MEV``.
    """
    E_e = energy_MeV - constant.DELTA_NP_MEV
    p_e_sq = (E_e ** 2 - constant.M_ELECTRON_MEV ** 2).clamp_min(0.0)
    p_e = torch.sqrt(p_e_sq)

    sigma = (constant.SIGMA0_IBD_CM2_PER_MEV2 / constant.SIGMA_UNIT_CM2) * E_e * p_e
    below_threshold = energy_MeV < constant.IBD_THRESHOLD_MEV
    return torch.where(below_threshold, torch.zeros_like(sigma), sigma)


def ibd_cross_section_grid(E_nu_grid_MeV: torch.Tensor, T_grid_MeV: torch.Tensor) -> torch.Tensor:
    """Zeroth-order differential IBD cross section dsigma/dT on an ``(E_nu_grid, T_grid)`` pair.

    Approximates the visible prompt signal as monoenergetic at
    ``E_prompt = E_nu - DELTA_NP_MEV + M_ELECTRON_MEV`` (no-recoil
    kinematics), represented as a triangular kernel one ``T_grid_MeV``
    spacing wide, so integrating the returned column over ``T_grid_MeV``
    reproduces ``ibd_cross_section(E_nu)`` to the grid's own resolution.

    Args:
        E_nu_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        T_grid_MeV: Visible prompt-energy grid, shape ``(n_T,)``,
            uniformly spaced.

    Returns:
        Real tensor shaped ``(n_E, n_T)``, in units of 1e-42 cm^2/MeV.
    """
    dT = T_grid_MeV[1] - T_grid_MeV[0]
    E_prompt = (E_nu_grid_MeV - constant.DELTA_NP_MEV + constant.M_ELECTRON_MEV).clamp_min(0.0)

    delta = T_grid_MeV[None, :] - E_prompt[:, None]  # (n_E, n_T)
    triangle = (1.0 - delta.abs() / dT).clamp_min(0.0) / dT  # integrates to 1 over T for each row

    sigma_tot = ibd_cross_section(E_nu_grid_MeV)  # (n_E,)
    return sigma_tot[:, None] * triangle


def ibd_differential_cross_section_precise(
    energy_MeV: torch.Tensor,
    cos_theta: torch.Tensor,
    *,
    f: float = 1.0,
    g: float = 1.2701,
    f2: float = 3.706,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Order-1/M differential IBD cross section dsigma/dcos(theta), VB99 Eqs. (14)-(15).

    ``f``, ``g`` and ``f2`` are the vector, axial-vector, and anomalous
    isovector magnetic couplings (VB99 Eq. 4). The recoil term (VB99
    Eq. 15's ``Gamma``) is evaluated as ``p_e^(0) * Gamma`` in an
    algebraically rearranged form that removes the ``1/v_e^(0)`` terms'
    apparent singularity at threshold (each such term is exactly cancelled
    by the ``p_e^(0)`` prefactor multiplying ``Gamma`` in Eq. 14; the
    rearrangement makes that cancellation manifest instead of relying on
    floating-point near-cancellation of a large and a small number).

    Args:
        energy_MeV: Antineutrino energy, shape broadcastable with ``cos_theta``.
        cos_theta: cos(theta) between the incident antineutrino and
            outgoing positron directions (lab frame), same shape as
            ``energy_MeV``.
        f, g, f2: Vector, axial-vector, and anomalous-magnetic-moment
            coupling constants (VB99 Eq. 4).

    Returns:
        ``(dsigma_dcostheta, E_e1)``: ``dsigma_dcostheta`` in units of 1e-42 cm^2 (per
        unit cos(theta)), zero below ``constant.IBD_THRESHOLD_MEV``;
        ``E_e1`` the order-1/M positron total energy (VB99 Eq. 13), MeV --
        both the same shape as ``energy_MeV``.
    """
    Delta = constant.DELTA_NP_MEV
    M = 0.5 * (constant.M_NEUTRON_MEV + constant.M_PROTON_MEV)
    m_e = constant.M_ELECTRON_MEV

    E_e0 = energy_MeV - Delta
    valid = E_e0 > m_e
    E_e0c = E_e0.clamp_min(m_e)
    p_e0_sq = (E_e0c ** 2 - m_e ** 2).clamp_min(0.0)
    p_e0 = torch.sqrt(p_e0_sq)

    y2 = (Delta ** 2 - m_e ** 2) / 2.0
    E_e1 = E_e0c * (1.0 - (energy_MeV / M) * (1.0 - (p_e0 / E_e0c) * cos_theta)) - y2 / M
    p_e1_sq = (E_e1 ** 2 - m_e ** 2).clamp_min(0.0)
    p_e1 = torch.sqrt(p_e1_sq)

    main = 0.5 * _SIGMA0_BARE_UNITS_PER_MEV4 * (
        (f ** 2 + 3.0 * g ** 2) * E_e1 * p_e1 + (f ** 2 - g ** 2) * cos_theta * p_e1_sq
    )

    # p_e0 * Gamma (VB99 Eq. 15), algebraically cleaned of 1/v_e0 terms (see docstring).
    term1 = 2.0 * (f + f2) * g * (
        p_e0 * (2.0 * E_e0c + Delta)
        - cos_theta * (p_e0 ** 2 / E_e0c) * (2.0 * E_e0c + Delta)
        - p_e0 * m_e ** 2 / E_e0c
    )
    term2 = (f ** 2 + g ** 2) * (
        Delta * p_e0 + cos_theta * Delta * (p_e0 ** 2 / E_e0c) + p_e0 * m_e ** 2 / E_e0c
    )
    term3 = (f ** 2 + 3.0 * g ** 2) * (p_e0 * E_e0c - cos_theta * E_e0c * (E_e0c + Delta))
    term4 = (f ** 2 - g ** 2) * p_e0 * cos_theta * (p_e0 - cos_theta * (E_e0c + Delta))
    p_e0_gamma = term1 + term2 + term3 + term4

    recoil = 0.5 * _SIGMA0_BARE_UNITS_PER_MEV4 / M * E_e0c * p_e0_gamma

    dsigma_dcos = (main - recoil).clamp_min(0.0)
    dsigma_dcos = torch.where(valid, dsigma_dcos, torch.zeros_like(dsigma_dcos))
    return dsigma_dcos, E_e1


def ibd_cross_section_grid_precise(
    E_nu_grid_MeV: torch.Tensor,
    T_grid_MeV: torch.Tensor,
    *,
    f: float = 1.0,
    g: float = 1.2701,
    f2: float = 3.706,
    n_cos: int = 41,
) -> torch.Tensor:
    """Order-1/M differential IBD cross section dsigma/dT, VB99 Eqs. (14)-(15).

    Built by numerically integrating ``ibd_differential_cross_section_precise`` over
    cos(theta) on a uniform ``n_cos``-point trapezoidal grid. Each angular
    contribution is redistributed linearly onto ``T_grid_MeV`` at its
    order-1/M prompt energy, ``E_prompt = E_e1 + M_ELECTRON_MEV``, via
    ``tpeanuts.detector.common.response.scatter_add_linear``.

    Args:
        E_nu_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        T_grid_MeV: Visible prompt-energy grid, shape ``(n_T,)``,
            uniformly spaced, strictly increasing.
        f, g, f2: Coupling constants, see ``ibd_differential_cross_section_precise``.
        n_cos: Number of cos(theta) quadrature nodes.

    Returns:
        Real tensor shaped ``(n_E, n_T)``, in units of 1e-42 cm^2/MeV. Integrating a row over
        ``T_grid_MeV`` (trapezoidal quadrature) reproduces the order-1/M
        total cross section at that row's ``E_nu`` to the quadrature's
        accuracy.
    """
    dtype, device = E_nu_grid_MeV.dtype, E_nu_grid_MeV.device
    cos_grid = torch.linspace(-1.0, 1.0, n_cos, dtype=dtype, device=device)
    dc = cos_grid[1] - cos_grid[0]
    weights = torch.full_like(cos_grid, float(dc))
    weights[0] *= 0.5
    weights[-1] *= 0.5

    n_E = E_nu_grid_MeV.shape[0]
    E_nu = E_nu_grid_MeV[:, None].expand(n_E, n_cos)
    cos_theta = cos_grid[None, :].expand(n_E, n_cos)

    dsigma_dcos, E_e1 = ibd_differential_cross_section_precise(E_nu, cos_theta, f=f, g=g, f2=f2)
    mass = dsigma_dcos * weights[None, :]  # (n_E, n_cos), 1e-42 cm^2 per node
    E_prompt = E_e1 + constant.M_ELECTRON_MEV

    dT = T_grid_MeV[1] - T_grid_MeV[0]
    mass_on_grid = scatter_add_linear(T_grid_MeV, E_prompt, mass)  # (n_E, n_T), 1e-42 cm^2
    return mass_on_grid / dT
