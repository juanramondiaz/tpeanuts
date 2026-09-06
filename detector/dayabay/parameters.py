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
Daya Bay geometry/isotope constants and parameters loaded at import time.

Module contents:
    DETECTORS, REACTORS, BG_CATEGORIES
        Names, in the data release's own order.
    NEAR_DETECTORS, FAR_DETECTORS
        Experimental-hall grouping (EH1+EH2 vs EH3).
    ISOTOPES
        Re-exported from ``tpeanuts.source.reactor``.
    BASELINES_KM
        (detector, reactor) baseline matrix.
    FISSION_FRACTIONS, ENERGY_PER_FISSION_MEV, THERMAL_POWER_GW
        Reactor parameters, shared by all 6 cores.
    N_PROTONS
        Target proton count per detector.
    ERES_A, ERES_B, ERES_C
        Energy-resolution formula coefficients.
    FINAL_EREC_BIN_EDGES_MEV
        Analysis binning.
    E_NU_GRID_MEV, T_GRID_MEV, TPRIME_GRID_MEV
        Grids used to fold the physics: E_NU_GRID_MEV spans the IBD-
        relevant Huber-Mueller range; T_GRID_MEV/TPRIME_GRID_MEV are the
        observed spectrum's own 0.05 MeV bin centers, so predictions and
        data share the same fine binning before both are rebinned onto
        FINAL_EREC_BIN_EDGES_MEV.
    DETECTOR_EFFICIENCY
        IBD-selection efficiency, applied on top of exposure.
    IBD_CONSTANTS
        IBD cross-section coupling constants (f, g, f2, PhaseSpaceFactor).
    IAV_MATRIX, LSNL_CURVE_E_MEV, LSNL_CURVE_F, LSNL_CURVE_PULLS
        Detector-response corrections (see ``response``).
    NONEQUILIBRIUM_CORRECTION, SNF_CORRECTION
        Per-isotope/per-reactor flux corrections (see ``flux``).
    REACTOR_RELATIVE_WEIGHT
        Per-reactor relative power/output weight for the 8AD period,
        derived from the weekly antineutrino-rate history (mean over
        ``n_det==8`` weeks, normalized to a 6-reactor average of 1).
    GLOBAL_NORMALIZATION_NOMINAL
        Daya Bay's own nominal global-normalization value (1.0), the
        starting point for the free normalization nuisance.
    BACKGROUND_CATEGORY_SIGMA
        Per-category background-rate fractional uncertainty: one
        correlated scale nuisance per category rather than per detector/
        hall (see ``_background_category_sigma`` for the simplification
        relative to the official per-detector/per-hall correlation
        structure), used as the Gaussian-prior width for the background-
        rate nuisance parameters.
"""

from __future__ import annotations

import torch

from tpeanuts.detector.dayabay.io import (
    load_background_rates,
    load_baselines,
    load_detector_efficiency,
    load_eres_parameters,
    load_final_erec_bin_edges,
    load_global_normalization,
    load_ibd_constants,
    load_iav_matrix,
    load_lsnl_curve,
    load_lsnl_curve_pulls,
    load_n_protons,
    load_neutrino_rate_weekly,
    load_nonequilibrium_correction,
    load_reactor_parameters,
    load_snf_correction,
)
from tpeanuts.source.reactor import ISOTOPES

DETECTORS: tuple[str, ...] = ("AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34")
REACTORS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6")
BG_CATEGORIES: tuple[str, ...] = ("accidentals", "alpha_neutron", "amc", "fast_neutrons", "lithium_helium")

# Experimental-hall grouping (detector name's first digit = hall number):
# EH1 (AD11/AD12) and EH2 (AD21/AD22) are the two near halls (flux-weighted
# baselines ~560/600 m); EH3 (AD31-AD34) is the far hall (~1640 m).
NEAR_DETECTORS: tuple[str, ...] = ("AD11", "AD12", "AD21", "AD22")
FAR_DETECTORS: tuple[str, ...] = ("AD31", "AD32", "AD33", "AD34")

BASELINES_KM: dict[str, dict[str, torch.Tensor]] = load_baselines(dtype=torch.float64)
FISSION_FRACTIONS, ENERGY_PER_FISSION_MEV, THERMAL_POWER_GW = load_reactor_parameters()
N_PROTONS: dict[str, torch.Tensor] = load_n_protons(dtype=torch.float64)
ERES_A, ERES_B, ERES_C = load_eres_parameters()
FINAL_EREC_BIN_EDGES_MEV: torch.Tensor = load_final_erec_bin_edges(dtype=torch.float64)

# IBD-selection efficiency (Gd-capture + delayed-coincidence + analysis
# cuts), separate from and multiplicative with the daily livetime exposure.
DETECTOR_EFFICIENCY: float = load_detector_efficiency()

# True antineutrino energy grid: the tabulated Huber-Mueller range relevant
# to IBD (threshold ~1.8 MeV up to where the flux is negligible).
E_NU_GRID_MEV: torch.Tensor = torch.linspace(1.8, 9.0, 145, dtype=torch.float64)

# Reconstructed/true prompt-energy grid, matching the observed spectrum's
# own bin edges (0.05 MeV steps, 0-12 MeV, 241 points).
T_GRID_MEV: torch.Tensor = torch.linspace(0.0, 12.0, 241, dtype=torch.float64)
TPRIME_GRID_MEV: torch.Tensor = T_GRID_MEV

# IBD cross-section coupling constants: vector coupling f, axial-vector
# coupling g, anomalous nucleon isovector magnetic moment f2.
IBD_CONSTANTS: dict[str, float] = load_ibd_constants()

# IAV energy-redistribution matrix (240x240, probability mass, columns sum
# to 1) and nominal LSNL energy-scale nonlinearity curve.
IAV_MATRIX: torch.Tensor = load_iav_matrix(dtype=torch.float64)
LSNL_CURVE_E_MEV, LSNL_CURVE_F = load_lsnl_curve(dtype=torch.float64)

# LSNL systematic-variation ("pull") curves, f_k(E), k=0..3.
LSNL_CURVE_PULLS: list[tuple[torch.Tensor, torch.Tensor]] = load_lsnl_curve_pulls(dtype=torch.float64)

# Per-isotope non-equilibrium (U235/Pu239/Pu241 only) and per-reactor SNF
# flux corrections, {key: (E_MeV, C)}.
NONEQUILIBRIUM_CORRECTION: dict[str, tuple[torch.Tensor, torch.Tensor]] = load_nonequilibrium_correction(
    dtype=torch.float64,
)
SNF_CORRECTION: dict[str, tuple[torch.Tensor, torch.Tensor]] = load_snf_correction(dtype=torch.float64)


def _reactor_relative_weights() -> dict[str, float]:
    """Per-reactor relative power/output weight for the 8AD period."""
    table = load_neutrino_rate_weekly()
    table_8ad = table[table["n_det"] == 8]
    mean_rate = table_8ad.groupby("reactor")["neutrino_rate_per_s"].mean()
    weights = mean_rate / mean_rate.mean()
    return {reactor: float(weights[reactor]) for reactor in weights.index}


REACTOR_RELATIVE_WEIGHT: dict[str, float] = _reactor_relative_weights()

GLOBAL_NORMALIZATION_NOMINAL: float = load_global_normalization()


def _background_category_sigma() -> dict[str, float]:
    """Per-category background-rate fractional uncertainty, one correlated scale per category.

    The official release documents a genuine per-detector/per-hall
    correlation structure (accidentals uncorrelated between detectors, AmC
    correlated across all detectors, 9Li/8He and fast neutrons correlated
    within a hall). This project simplifies that to one fully correlated
    scale nuisance per category, with prior width

        sigma_category = sqrt(sum_d uncertainty_d^2) / sum_d rate_d,

    a rate-weighted combination of the per-detector (rate, uncertainty)
    pairs into a single relative uncertainty on that category's total rate.
    """
    rates = load_background_rates()
    sigmas: dict[str, float] = {}
    for category in BG_CATEGORIES:
        rate = rates.loc[f"{category}_rate"].astype(float)
        uncertainty = rates.loc[f"{category}_uncertainty"].astype(float)
        sigmas[category] = float((uncertainty ** 2).sum() ** 0.5 / rate.sum())
    return sigmas


BACKGROUND_CATEGORY_SIGMA: dict[str, float] = _background_category_sigma()
