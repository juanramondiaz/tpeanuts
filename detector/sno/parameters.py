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
SNO heavy-water target composition and energy grids.

Module contents:
    D2O_MOLAR_MASS_G_MOL, TARGET_MASS_TON
        Heavy-water stoichiometry inputs.
    N_TARGET_DEUTERONS
        Target deuteron count at ``TARGET_MASS_TON``, derived from
        stoichiometry (2 deuterons per D2O molecule), not quoted from
        memory as a detector-specific number. Consumed by
        ``detector.sno.event_rate.cc_event_rate``/``nc_event_rate``.
    N_TARGET_ELECTRONS
        Target electron count at ``TARGET_MASS_TON``, from the same D2O
        stoichiometry (10 electrons/molecule: 2 x Z(H) + Z(O)). Consumed by
        ``detector.sno.event_rate.es_event_rate`` -- SNO's elastic-
        scattering channel sees the water's electrons, not its deuterons.
    E_NU_GRID_MEV, T_GRID_MEV, TPRIME_GRID_MEV
        Default integration grids spanning the day/night spectrum's
        electron-energy range (5-20 MeV, see
        ``data/detector/sno/observation/day_night_spectrum.csv``).
    ENERGY_RESOLUTION_C0, ENERGY_RESOLUTION_C1, ENERGY_RESOLUTION_C2
        Real SNO Phase-I energy-resolution parametrization coefficients --
        see ``detector.sno.response`` for the formula and citation. Shared
        by the CC/ES continuous response and the NC capture-gamma response
        (same Cherenkov detector technology).
    ENERGY_RESOLUTION_FLOOR_MEV
        Numerical floor on sigma(T) (see ``detector.sno.response``), not a
        physical parameter.
    NC_CAPTURE_ENERGY_MEV
        Real (not illustrative) Q-value of n + d -> t + gamma, the
        radiative neutron-capture reaction SNO Phase-I's pure-D2O NC signal
        is detected through.
    NC_CAPTURE_EFFICIENCY
        Phase-I neutron detection efficiency after the fiducial-volume and
        5-MeV kinetic-energy cuts.
"""

from __future__ import annotations

import torch

import tpeanuts.util.constant as constant
from tpeanuts.detector.common.target import n_electrons

# D2O (heavy water) molar mass, g/mol: 2 * deuterium (2.0141) + oxygen (15.9994).
D2O_MOLAR_MASS_G_MOL: float = 2.0 * 2.0141 + 15.9994

# Deuterium's nucleus (a single proton) has the same electron structure as
# ordinary hydrogen (Z=1); detector.common.target.ATOMIC_NUMBER counts
# electrons per neutral atom, which does not depend on the isotope, so "H"
# is reused rather than adding a "D" entry that would carry an identical value.
D2O_COMPOSITION: dict[str, int] = {"H": 2, "O": 1}

# SNO's approximate heavy-water target mass (order-of-magnitude figure widely
# quoted for the experiment; not verified here against a primary SNO paper).
TARGET_MASS_TON: float = 1000.0

_N_D2O_MOLECULES = (TARGET_MASS_TON * 1.0e6 / D2O_MOLAR_MASS_G_MOL) * constant.N_A
N_TARGET_DEUTERONS: torch.Tensor = torch.tensor(2.0 * _N_D2O_MOLECULES, dtype=torch.float64)

N_TARGET_ELECTRONS: torch.Tensor = n_electrons(D2O_COMPOSITION, D2O_MOLAR_MASS_G_MOL, TARGET_MASS_TON)

# Real SNO Phase-I kinetic-energy resolution parametrization,
# sigma_T(Te) = C0 + C1*sqrt(Te) + C2*Te [MeV, Te in MeV] -- the widely-quoted
# SNO Phase-I resolution function used in solar-neutrino oscillation-fit
# literature (see detector.sno.response for the full formula and its
# numerical floor). Unlike detector.borexino.parameters.ENERGY_RESOLUTION_A,
# this is real, not illustrative.
ENERGY_RESOLUTION_C0: float = -0.0684
ENERGY_RESOLUTION_C1: float = 0.331
ENERGY_RESOLUTION_C2: float = 0.0425

# Numerical floor on sigma(T), MeV -- the formula above goes non-positive
# for Te below ~0.04 MeV (well under SNO's actual analysis threshold), so
# this only guards T_GRID_MEV's low-T tail against feeding a non-positive
# sigma into gaussian_response_matrix; it carries no physical meaning.
ENERGY_RESOLUTION_FLOOR_MEV: float = 1.0e-3

E_NU_GRID_MEV: torch.Tensor = torch.linspace(1.0, 20.0, 400, dtype=torch.float64)
T_GRID_MEV: torch.Tensor = torch.linspace(0.0, 20.0, 500, dtype=torch.float64)
TPRIME_GRID_MEV: torch.Tensor = T_GRID_MEV

# Q-value of n + d -> t + gamma, MeV -- the real (well-established nuclear
# data, not illustrative) energy of the single gamma SNO Phase-I's pure-D2O
# NC channel is detected through (neutron thermalizes and is radiatively
# captured on a second deuteron; the gamma Compton-scatters an electron,
# detected the same way as CC/ES Cherenkov light).
NC_CAPTURE_ENERGY_MEV: float = 6.25

# SNO Phase-I neutron detection efficiency within the fiducial volume and
# above the 5-MeV kinetic-energy threshold: 14.4%. This is the effective
# analysis efficiency, not merely the larger probability (~29.9%) that a
# neutron captures on deuterium. SNO Collaboration, Phys. Rev. Lett. 89,
# 011301 (2002), DOI 10.1103/PhysRevLett.89.011301.
NC_CAPTURE_EFFICIENCY: float = 0.144
