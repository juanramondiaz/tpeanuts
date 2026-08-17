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
#      August 2026
# =============================================================================

"""
SNO salt-phase (Phase II) constants: observable ordering, binning, response, hep flux.

All real values here are transcribed from the primary source (see
``tpeanuts.detector.sno_ii``'s package docstring) via
``data/detector/sno_ii/metadata/source.json``, not illustrative.

Module contents:
    CHANNEL_ORDER
        The 19-entry observable ordering (NC, CC1..CC17, ES) used
        consistently by every table in ``data/detector/sno_ii/`` and by
        ``tpeanuts.detector.sno_ii.inference_model``'s predicted vector.
    N_CC_BINS, N_CHANNELS, N_OBSERVABLES_PER_PERIOD, N_OBSERVABLES_TOTAL
        Dimensions of the observable vector (17 CC bins + NC + ES = 19 per
        period; day and night = 38 total).
    CC_BIN_EDGES_MEV
        The 17 CC recoil-electron kinetic-energy bin edges (Table XXX):
        16 bins of 0.5 MeV from 5.5-13.5 MeV, plus one wide 13.5-20.0 MeV
        bin.
    ENERGY_RESOLUTION_C0, ENERGY_RESOLUTION_C1, ENERGY_RESOLUTION_C2
        Salt-phase kinetic-energy resolution parametrization coefficients,
        sigma_T(Te) = C0 + C1*sqrt(Te) + C2*Te [MeV], Eq. A3 of the primary
        source -- distinct from (and more recent/precise than)
        ``tpeanuts.detector.sno.parameters``' Phase-I coefficients; do not
        conflate the two.
    ENERGY_RESOLUTION_FLOOR_MEV
        Numerical floor on sigma(T), not a physical parameter.
    T_GRID_MEV, TPRIME_GRID_MEV
        Default true/reconstructed electron kinetic-energy integration
        grids, spanning ``CC_BIN_EDGES_MEV``'s own range.
    E_NU_GRID_MEV
        Default true neutrino energy integration grid.
    ANALYSIS_THRESHOLD_MEV
        SNO salt-phase analysis threshold (5.5 MeV kinetic energy), the
        lower edge of ``CC_BIN_EDGES_MEV`` and the equivalent-flux
        denominator's own integration range in Eq. A1 of the primary
        source.
    HEP_FLUX_CM2S
        Fixed hep solar-neutrino flux SNO's own salt-phase analysis
        assumed (not a free parameter), verified at the primary source.
    N_TARGET_DEUTERONS, N_TARGET_ELECTRONS
        Re-exported from ``tpeanuts.detector.sno.parameters``: the salt
        phase's ~2 t NaCl addition to the ~1000 t D2O target is a
        negligible dilution of the target stoichiometry, so Phase-I's
        already-derived target composition is reused rather than
        recomputed.
    DETECTOR_DEPTH_M
        Same physical detector as Phase I, see
        ``tpeanuts.detector.sno.inference_model.SNODayNightModel``.
"""

from __future__ import annotations

import torch

from tpeanuts.detector.sno.parameters import (
    N_TARGET_DEUTERONS,
    N_TARGET_ELECTRONS,
)

__all__ = [
    "CHANNEL_ORDER",
    "N_CC_BINS",
    "N_CHANNELS",
    "N_OBSERVABLES_PER_PERIOD",
    "N_OBSERVABLES_TOTAL",
    "CC_BIN_EDGES_MEV",
    "ENERGY_RESOLUTION_C0",
    "ENERGY_RESOLUTION_C1",
    "ENERGY_RESOLUTION_C2",
    "ENERGY_RESOLUTION_FLOOR_MEV",
    "T_GRID_MEV",
    "TPRIME_GRID_MEV",
    "E_NU_GRID_MEV",
    "ANALYSIS_THRESHOLD_MEV",
    "HEP_FLUX_CM2S",
    "N_TARGET_DEUTERONS",
    "N_TARGET_ELECTRONS",
    "DETECTOR_DEPTH_M",
]

# Observable ordering, matching every table in data/detector/sno_ii/ and
# Tables XXX/XXXII/XXXIII/XXXIV of the primary source exactly.
CHANNEL_ORDER: tuple[str, ...] = (
    "NC",
    "CC1", "CC2", "CC3", "CC4", "CC5", "CC6", "CC7", "CC8", "CC9",
    "CC10", "CC11", "CC12", "CC13", "CC14", "CC15", "CC16", "CC17",
    "ES",
)

N_CC_BINS: int = 17
N_CHANNELS: int = len(CHANNEL_ORDER)  # 19
N_OBSERVABLES_PER_PERIOD: int = N_CHANNELS  # 19 (NC + 17 CC bins + ES)
N_OBSERVABLES_TOTAL: int = 2 * N_OBSERVABLES_PER_PERIOD  # 38 (day + night)

# Table XXX: 16 bins of 0.5 MeV from 5.5 MeV, plus one wide bin to 20.0 MeV.
CC_BIN_EDGES_MEV: torch.Tensor = torch.cat([
    torch.arange(5.5, 13.5001, 0.5, dtype=torch.float64),
    torch.tensor([20.0], dtype=torch.float64),
])

# Eq. A3 of the primary source: sigma_T(Te) = -0.131 + 0.383*sqrt(Te) + 0.03731*Te [MeV].
ENERGY_RESOLUTION_C0: float = -0.131
ENERGY_RESOLUTION_C1: float = 0.383
ENERGY_RESOLUTION_C2: float = 0.03731

# Numerical floor on sigma(T), MeV -- as in detector.sno.parameters, this
# formula goes non-positive below Te ~= 0.12 MeV (well under the 5.5 MeV
# analysis threshold), so this only guards the low-T tail of T_GRID_MEV
# against feeding a non-positive sigma into gaussian_response_matrix; it
# carries no physical meaning.
ENERGY_RESOLUTION_FLOOR_MEV: float = 1.0e-3

# Default true/reconstructed electron-energy integration grids, spanning
# the CC spectrum's own range (see CC_BIN_EDGES_MEV above).
T_GRID_MEV: torch.Tensor = torch.linspace(0.0, 20.0, 500, dtype=torch.float64)
TPRIME_GRID_MEV: torch.Tensor = T_GRID_MEV
E_NU_GRID_MEV: torch.Tensor = torch.linspace(1.0, 20.0, 400, dtype=torch.float64)

# Table XXX's own quoted analysis threshold: CC_BIN_EDGES_MEV[0].
ANALYSIS_THRESHOLD_MEV: float = 5.5

# Fixed hep contribution assumed throughout SNO's own salt-phase flux
# extraction (Section XI of the primary source): "a fixed hep contribution
# of 9.3e3 cm^-2 s^-1 has been assumed", in cm^-2 s^-1.
HEP_FLUX_CM2S: float = 9.3e3

# Same physical detector as Phase I; see
# tpeanuts.detector.sno.inference_model.SNODayNightModel's own caveat about
# this being an approximate, not primary-source-verified, depth.
DETECTOR_DEPTH_M: float = 2039.0
