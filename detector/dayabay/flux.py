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
Daya Bay reactor antineutrino flux: Huber-Mueller spectrum, normalization and corrections.

The per-isotope spectrum interpolation and fission-fraction-weighted
combination are generic reactor physics, reused from
``tpeanuts.source.reactor.huber_mueller`` (Huber 2011 for U235/Pu239/Pu241;
Mueller et al. 2011 for U238); this module adds what is specific to Daya
Bay's own 6 cores -- time-averaged fission fractions, normalization by
thermal power and energy-per-fission, and three further per-reactor/per-
isotope corrections:

    S_i^corrected(E) = S_i(E) * (1 + C_i^neq(E))     i in {U235, Pu239, Pu241}
    Phi_face,r(E) = (P_th / <e_fission>) * w_r * (1 + C_r^snf(E))
                    * sum_i f_i S_i^corrected(E)                    [nu/s/MeV]
    Phi_detector,r(E, L) = Phi_face,r(E) / (4 pi L^2)                [nu/s/MeV/cm^2]

with ``<e_fission> = sum_i f_i * e_fission,i`` the fission-fraction-weighted
mean energy per fission, ``C_i^neq`` the per-isotope non-equilibrium
correction (long-lived fission-daughter beta decays not yet in secular
equilibrium; not published for U238), ``C_r^snf`` the per-reactor spent-
nuclear-fuel correction, and ``w_r`` the per-reactor relative power/output
weight for the 8AD period (see ``parameters.REACTOR_RELATIVE_WEIGHT``).

Module contents:
    reactor_spectrum_shape(...)
        Fission-fraction-weighted sum over the 4 isotopes, with the
        non-equilibrium correction folded into U235/Pu239/Pu241.
    reactor_flux_at_face(...)
        Absolute flux at one reactor core, before geometric dilution.
    flux_at_detector(...)
        Phi_detector,r(E, L), see above.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

import tpeanuts.util.constant as constant
from tpeanuts.detector.dayabay.io import load_huber_mueller_spectra
from tpeanuts.detector.dayabay.parameters import (
    ENERGY_PER_FISSION_MEV,
    FISSION_FRACTIONS,
    ISOTOPES,
    NONEQUILIBRIUM_CORRECTION,
    REACTOR_RELATIVE_WEIGHT,
    SNF_CORRECTION,
    THERMAL_POWER_GW,
)
from tpeanuts.source.reactor import weighted_spectrum_shape


@functools.lru_cache(maxsize=1)
def _hm_curves() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return load_huber_mueller_spectra(dtype=torch.float64)


@functools.lru_cache(maxsize=1)
def _corrected_hm_curves() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Huber-Mueller curves with the non-equilibrium correction folded into U235/Pu239/Pu241.

    Each corrected isotope's curve becomes ``S_i(E) * (1 + C_i^neq(E))``,
    with ``C_i^neq`` linearly interpolated onto that isotope's own tabulated
    energy grid (0 outside the correction's range). U238 has no published
    correction and is passed through unchanged.
    """
    curves = dict(_hm_curves())
    for isotope, (E_corr, C_corr) in NONEQUILIBRIUM_CORRECTION.items():
        E_tab, N_tab = curves[isotope]
        C_on_tab = np.interp(
            E_tab.cpu().numpy(), E_corr.cpu().numpy(), C_corr.cpu().numpy(), left=0.0, right=0.0,
        )
        C_on_tab_t = torch.as_tensor(C_on_tab, dtype=N_tab.dtype, device=N_tab.device)
        curves[isotope] = (E_tab, N_tab * (1.0 + C_on_tab_t))
    return curves


def reactor_spectrum_shape(E_grid_MeV: torch.Tensor) -> torch.Tensor:
    """Fission-fraction-weighted sum over the 4 isotopes, antineutrinos/fission/MeV.

    Includes the non-equilibrium correction (see ``_corrected_hm_curves``);
    shared by all 6 reactor cores (the correction is per-isotope, not
    per-reactor).
    """
    return weighted_spectrum_shape(_corrected_hm_curves(), FISSION_FRACTIONS, E_grid_MeV)


def mean_energy_per_fission_MeV() -> float:
    """Fission-fraction-weighted mean energy released per fission, MeV."""
    return sum(FISSION_FRACTIONS[iso] * ENERGY_PER_FISSION_MEV[iso] for iso in ISOTOPES)


def reactor_flux_at_face(E_grid_MeV: torch.Tensor, reactor: str) -> torch.Tensor:
    """Absolute antineutrino flux at one reactor core, before geometric dilution.

    Phi_face,r(E) = (P_th / <e_fission>) * w_r * (1 + C_r^snf(E))
                    * reactor_spectrum_shape(E), nu/s/MeV -- see module
    docstring for ``w_r`` and ``C_r^snf``.

    Args:
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        reactor: Reactor name, e.g. "R1".

    Returns:
        Tensor shaped ``(n_E,)``, nu/s/MeV.
    """
    fissions_per_s = (THERMAL_POWER_GW * 1.0e9) / (mean_energy_per_fission_MeV() * constant.MEV_TO_JOULE)

    E_snf, C_snf = SNF_CORRECTION[reactor]
    C_snf_on_grid = np.interp(
        E_grid_MeV.cpu().numpy(), E_snf.cpu().numpy(), C_snf.cpu().numpy(), left=0.0, right=0.0,
    )
    snf_factor = 1.0 + torch.as_tensor(C_snf_on_grid, dtype=E_grid_MeV.dtype, device=E_grid_MeV.device)

    return (
        fissions_per_s * REACTOR_RELATIVE_WEIGHT[reactor] * snf_factor * reactor_spectrum_shape(E_grid_MeV)
    )


def flux_at_detector(E_grid_MeV: torch.Tensor, baseline_km: torch.Tensor, reactor: str) -> torch.Tensor:
    """Phi_detector,r(E, L) = Phi_face,r(E) / (4 pi L^2), nu/s/MeV/cm^2.

    Args:
        E_grid_MeV: True antineutrino energy grid, shape ``(n_E,)``.
        baseline_km: Reactor-to-detector distance, km (scalar tensor).
        reactor: Reactor name, e.g. "R1".

    Returns:
        Tensor shaped ``(n_E,)``, nu/s/MeV/cm^2.
    """
    L_cm = baseline_km * 1.0e5
    return reactor_flux_at_face(E_grid_MeV, reactor) / (4.0 * torch.pi * L_cm ** 2)
