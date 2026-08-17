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
Daya Bay geometry/isotope constants and real parameters loaded at import time.

Module contents:
    DETECTORS, REACTORS, BG_CATEGORIES
        Real names, in the data release's own order.
    NEAR_DETECTORS, FAR_DETECTORS
        Real experimental-hall grouping (EH1+EH2 vs EH3).
    ISOTOPES
        Re-exported from ``tpeanuts.source.reactor`` (generic reactor
        physics, not specific to this data release).
    BASELINES_KM
        Real (detector, reactor) baseline matrix.
    FISSION_FRACTIONS, ENERGY_PER_FISSION_MEV, THERMAL_POWER_GW
        Real reactor parameters (shared by all 6 cores, see package
        docstring).
    N_PROTONS
        Real target proton count per detector.
    ERES_A, ERES_B, ERES_C
        Real energy-resolution formula coefficients.
    FINAL_EREC_BIN_EDGES_MEV
        Real analysis binning.
    E_NU_GRID_MEV, T_GRID_MEV, TPRIME_GRID_MEV
        Grids used to fold the physics: E_NU_GRID_MEV spans the IBD-
        relevant Huber-Mueller range; T_GRID_MEV/TPRIME_GRID_MEV are the
        real observed spectrum's own 0.05 MeV bin centers, 0-12 MeV, so
        model predictions and real data share the same fine binning before
        both are rebinned onto FINAL_EREC_BIN_EDGES_MEV.
    DETECTOR_EFFICIENCY
        Real IBD-selection efficiency, applied by
        ``detector.dayabay.event_rate.ibd_event_rate`` on top of the real
        exposure (see that module's docstring).
    IBD_CONSTANTS
        Real IBD cross-section coupling constants (f, g, f2,
        PhaseSpaceFactor), used by
        ``detector.interaction.inverse_beta_decay.ibd_cross_section_grid_precise``.
    IAV_MATRIX, LSNL_CURVE_E_MEV, LSNL_CURVE_F, LSNL_CURVE_PULLS
        Real detector-response corrections, used by
        ``detector.dayabay.response``.
    NONEQUILIBRIUM_CORRECTION, SNF_CORRECTION
        Real per-isotope/per-reactor flux corrections, used by
        ``detector.dayabay.flux``.
    REACTOR_RELATIVE_WEIGHT
        Real per-reactor relative power/output weight for the 8AD period,
        derived from the real weekly antineutrino-rate history (mean over
        ``n_det==8`` weeks, normalized to a 6-reactor average of 1), used by
        ``detector.dayabay.flux``.
    GLOBAL_NORMALIZATION_NOMINAL
        Daya Bay's own real nominal global-normalization value (1.0), the
        starting point for the free normalization nuisance parameter in
        ``detector.dayabay.inference_model``.
    BACKGROUND_CATEGORY_SIGMA
        Real per-category background-rate fractional uncertainty (one
        correlated scale nuisance per category, not per detector/hall --
        see the docstring on ``_background_category_sigma`` for the exact
        real-data aggregation and its simplification relative to the
        official analysis's per-detector/per-hall correlation structure),
        used as the real Gaussian-prior width for the background-rate
        nuisance parameters in ``detector.dayabay.inference_model``.
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

# Real experimental-hall grouping (detector name's first digit = hall number,
# the official naming convention): EH1 (AD11/AD12) and EH2 (AD21/AD22) are
# the two real near halls (flux-weighted baselines ~560/600 m); EH3
# (AD31-AD34) is the real far hall (~1640 m) -- see
# detector.dayabay.inference_model.NearFarRatioDayaBayModel.
NEAR_DETECTORS: tuple[str, ...] = ("AD11", "AD12", "AD21", "AD22")
FAR_DETECTORS: tuple[str, ...] = ("AD31", "AD32", "AD33", "AD34")

BASELINES_KM: dict[str, dict[str, torch.Tensor]] = load_baselines(dtype=torch.float64)
FISSION_FRACTIONS, ENERGY_PER_FISSION_MEV, THERMAL_POWER_GW = load_reactor_parameters()
N_PROTONS: dict[str, torch.Tensor] = load_n_protons(dtype=torch.float64)
ERES_A, ERES_B, ERES_C = load_eres_parameters()
FINAL_EREC_BIN_EDGES_MEV: torch.Tensor = load_final_erec_bin_edges(dtype=torch.float64)

# Real IBD-selection efficiency (Gd-capture + delayed-coincidence +
# analysis cuts), separate from and multiplicative with the daily
# eff_livetime exposure -- see detector.dayabay.io.load_detector_efficiency.
DETECTOR_EFFICIENCY: float = load_detector_efficiency()

# True antineutrino energy grid: the tabulated Huber-Mueller range relevant
# to IBD (threshold ~1.8 MeV up to where the flux is negligible), same 50
# keV resolution as the source tables.
E_NU_GRID_MEV: torch.Tensor = torch.linspace(1.8, 9.0, 145, dtype=torch.float64)

# Reconstructed/true prompt-energy grid: positron kinetic energy plus the
# two 511-keV annihilation photons, matching the published observable. The
# grid uses the real observed spectrum's own
# bin edges, 0.05 MeV steps, 0-12 MeV (241 points) -- edges rather than
# centers so FINAL_EREC_BIN_EDGES_MEV's own range (0.7-12.0) lies strictly
# within this grid's range, as detector.common.event_rate.bin_counts
# requires.
T_GRID_MEV: torch.Tensor = torch.linspace(0.0, 12.0, 241, dtype=torch.float64)
TPRIME_GRID_MEV: torch.Tensor = T_GRID_MEV

# Real IBD cross-section coupling constants (Daya Bay's own
# parameters/ibd_constants.yaml): vector coupling f, axial-vector coupling
# g, anomalous nucleon isovector magnetic moment f2, and the phase-space
# factor (unused here -- see detector.interaction.inverse_beta_decay's own
# first-principles sigma0 derivation).
IBD_CONSTANTS: dict[str, float] = load_ibd_constants()

# Real IAV energy-redistribution matrix (240x240, probability mass, columns
# sum to 1) and real LSNL nominal energy-scale nonlinearity curve -- see
# detector.dayabay.response for how they are combined with the Gaussian
# resolution formula above.
IAV_MATRIX: torch.Tensor = load_iav_matrix(dtype=torch.float64)
LSNL_CURVE_E_MEV, LSNL_CURVE_F = load_lsnl_curve(dtype=torch.float64)

# Real LSNL systematic-variation ("pull") curves, f_k(E), k=0..3, each a
# real (E_MeV, f) pair on LSNL_CURVE_E_MEV's own energy grid -- see
# detector.dayabay.response and detector.dayabay.inference_model.
LSNL_CURVE_PULLS: list[tuple[torch.Tensor, torch.Tensor]] = load_lsnl_curve_pulls(dtype=torch.float64)

# Real per-isotope non-equilibrium (U235/Pu239/Pu241 only) and per-reactor
# SNF flux corrections, {key: (E_MeV, C)} -- see detector.dayabay.flux.
NONEQUILIBRIUM_CORRECTION: dict[str, tuple[torch.Tensor, torch.Tensor]] = load_nonequilibrium_correction(
    dtype=torch.float64,
)
SNF_CORRECTION: dict[str, tuple[torch.Tensor, torch.Tensor]] = load_snf_correction(dtype=torch.float64)


def _reactor_relative_weights() -> dict[str, float]:
    """Real per-reactor relative power/output weight for the 8AD period (see module docstring)."""
    table = load_neutrino_rate_weekly()
    table_8ad = table[table["n_det"] == 8]
    mean_rate = table_8ad.groupby("reactor")["neutrino_rate_per_s"].mean()
    weights = mean_rate / mean_rate.mean()
    return {reactor: float(weights[reactor]) for reactor in weights.index}


# Real per-reactor relative weight (8AD-period average of the real weekly
# rate, normalized to a 6-reactor mean of 1) -- see detector.dayabay.flux.
REACTOR_RELATIVE_WEIGHT: dict[str, float] = _reactor_relative_weights()

# Daya Bay's own real nominal global-normalization value (parameters/
# detector_normalization.yaml, 1.0) -- see detector.dayabay.inference_model.
GLOBAL_NORMALIZATION_NOMINAL: float = load_global_normalization()


def _background_category_sigma() -> dict[str, float]:
    """Real per-category background-rate fractional uncertainty, one correlated scale per category.

    The official release documents a genuine per-detector/per-hall
    correlation structure (accidentals uncorrelated between detectors, AmC
    correlated across all detectors, ⁹Li/⁸He and fast neutrons correlated
    within a hall -- see the data release's own README). This project uses
    a single simplification instead: one fully correlated scale nuisance
    per category, real-data-derived rather than invented, with prior width

        sigma_category = sqrt(sum_d uncertainty_d^2) / sum_d rate_d,

    a rate-weighted combination of the real per-detector (rate,
    uncertainty) pairs (``background_rates_8AD.csv``) into a single
    relative uncertainty on that category's real total rate.
    """
    rates = load_background_rates()
    sigmas: dict[str, float] = {}
    for category in BG_CATEGORIES:
        rate = rates.loc[f"{category}_rate"].astype(float)
        uncertainty = rates.loc[f"{category}_uncertainty"].astype(float)
        sigmas[category] = float((uncertainty ** 2).sum() ** 0.5 / rate.sum())
    return sigmas


# Real per-category background-rate fractional uncertainty (see docstring
# above) -- the real Gaussian-prior sigma for the background-rate nuisance
# parameters in detector.dayabay.inference_model.
BACKGROUND_CATEGORY_SIGMA: dict[str, float] = _background_category_sigma()
