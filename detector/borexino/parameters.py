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
Borexino scintillator composition, reference normalization, and energy grids.

Reference target mass/exposure (100 t, 1 day) are chosen to match the units
of the real published spectrum this package is validated against,
``data/detector/borexino/observation/nature2018_low_energy_spectrum.csv``
("Events/[day x 100t x N_h]", see ``detector.borexino.io``'s docstring for
the ``N_h`` caveat) -- not Borexino's true absolute fiducial mass or total
live time, which are not needed at that normalization and are not quoted
here to avoid stating a number from memory that cannot be checked against
the primary paper in this session.

Module contents:
    PSEUDOCUMENE_COMPOSITION, PSEUDOCUMENE_MOLAR_MASS_G_MOL
        Borexino's scintillator solvent (1,2,4-trimethylbenzene, C9H12);
        textbook stoichiometry, not Borexino-specific measurements. The PPO
        fluor (~1.5-2.0 g/L) and quenchers are neglected: their electron
        contribution is a small correction relative to the solvent's own
        mass fraction.
    REFERENCE_TARGET_MASS_TON, REFERENCE_EXPOSURE_DAYS
        The (100 t, 1 day) normalization above.
    N_TARGET_ELECTRONS
        Target electron count at the reference mass, from
        ``detector.common.target.n_electrons``.
    ENERGY_RESOLUTION_A
        Illustrative energy-resolution parametrization -- see
        ``detector.borexino.response`` for the formula and, importantly,
        the caveat that this value is *not* verified against the Nature
        2018 paper's own reported resolution.
    E_NU_GRID_MEV, T_GRID_MEV, TPRIME_GRID_MEV
        Default integration grids spanning the pp-chain neutrino energy
        range (matching ``notebooks/inference/inference1_borexino.ipynb``'s
        0.15-16 MeV convention) and the corresponding electron recoil range.
"""

from __future__ import annotations

import torch

from tpeanuts.detector.common.target import n_electrons

PSEUDOCUMENE_COMPOSITION: dict[str, int] = {"C": 9, "H": 12}
PSEUDOCUMENE_MOLAR_MASS_G_MOL: float = 120.19

REFERENCE_TARGET_MASS_TON: float = 100.0
REFERENCE_EXPOSURE_DAYS: float = 1.0

N_TARGET_ELECTRONS: torch.Tensor = n_electrons(
    PSEUDOCUMENE_COMPOSITION, PSEUDOCUMENE_MOLAR_MASS_G_MOL, REFERENCE_TARGET_MASS_TON,
)

# Illustrative sigma(T)/T = ENERGY_RESOLUTION_A / sqrt(T[MeV]) energy-resolution
# parametrization, the standard functional form for a photoelectron-counting
# liquid-scintillator detector (Poisson-limited light yield). The value
# 0.05 (5%/sqrt(MeV)) is a representative order-of-magnitude figure for this
# detector class, NOT a number read off the Nature 2018 paper -- treat any
# fit that is sensitive to its exact value as illustrative until it is
# replaced with the paper's own quoted resolution.
ENERGY_RESOLUTION_A: float = 0.05

E_NU_GRID_MEV: torch.Tensor = torch.linspace(0.15, 16.0, 400, dtype=torch.float64)
T_GRID_MEV: torch.Tensor = torch.linspace(0.05, 14.0, 400, dtype=torch.float64)
TPRIME_GRID_MEV: torch.Tensor = T_GRID_MEV
