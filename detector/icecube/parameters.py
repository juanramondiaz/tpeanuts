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
IceCube DeepCore real analysis binning and detector constants.

Source: IceCube Collaboration, *Measurement of atmospheric neutrino mixing
with improved IceCube DeepCore calibration and data processing*, Phys. Rev.
D 108, 012014 (2023) (arXiv:2304.12236), and its official public data
release (Harvard Dataverse, DOI 10.7910/DVN/B4RITM) -- 9 years of golden
(track-like, up-going/near-horizontal) DeepCore events.

Module contents:
    RECO_ENERGY_BIN_EDGES_GEV, RECO_COSZEN_BIN_EDGES, PID_BIN_EDGES
        Real analysis binning, exactly reproducing the data release's own
        ``readme.md`` recipe (log-spaced energy, one wide top bin merged
        for statistics, linear coszen, 2 PID categories).
    DETECTOR_DEPTH_M
        Approximate DeepCore depth below the South Pole ice surface.
    NOMINAL_SYSTEMATICS, BESTFIT_SYSTEMATICS
        Real detector-systematics hypersurface reference points (Table IV
        of the publication).
    THETA12_DEG, THETA13_DEG, DELTA13_DEG, DELTAMSQ21
        Fixed solar/reactor oscillation inputs (NuFit), not re-fit from
        atmospheric data (see ``tpeanuts.inference.model_atmosphere``).
    PUBLISHED_SIN2_THETA23, PUBLISHED_DELTAMSQ32, PUBLISHED_MUON_SCALE
        The publication's own best-fit point (normal ordering), for
        reference/comparison only -- not used as a prior anywhere in this
        package.
"""

from __future__ import annotations

import numpy as np
import torch

# Real analysis binning, from the data release's own readme.md:
#   "Energy (GeV): 10 bins from 6.31 to 158.49" (12 log-spaced edges, one
#   internal edge merged away "to contain sufficient statistics" -- the
#   release's own choice, reproduced here exactly rather than re-derived).
#   "Cosine of zenith: 11 bins from -1.0 (upgoing) to 0.1."
#   "Particle ID: [0.55, 0.75, 1.0]" (2 bins, tracks only -- the cascade
#   bin [0, 0.55] is not part of this release).
_EN_MIN, _EN_MAX = 6.31, 158.49
_CZ_MIN, _CZ_MAX = -1.0, 0.1

RECO_ENERGY_BIN_EDGES_GEV: torch.Tensor = torch.as_tensor(
    np.delete(np.logspace(np.log10(_EN_MIN), np.log10(_EN_MAX), num=12), -2),
    dtype=torch.float64,
)
RECO_COSZEN_BIN_EDGES: torch.Tensor = torch.linspace(_CZ_MIN, _CZ_MAX, 11, dtype=torch.float64)
PID_BIN_EDGES: torch.Tensor = torch.tensor([0.55, 0.75, 1.0], dtype=torch.float64)

N_ENERGY_BINS: int = RECO_ENERGY_BIN_EDGES_GEV.numel() - 1
N_COSZEN_BINS: int = RECO_COSZEN_BIN_EDGES.numel() - 1
N_PID_BINS: int = PID_BIN_EDGES.numel() - 1
N_BINS: int = N_ENERGY_BINS * N_COSZEN_BINS * N_PID_BINS

# DeepCore's approximate depth below the South Pole ice surface (~1450-2450 m
# for the instrumented DOMs; not verified here against a primary IceCube
# geometry paper -- the physical effect of this value on Earth-crossing
# geometry is tiny relative to the Earth's ~6371 km radius, the same
# reasoning already used for SNO's/Daya Bay's own approximate depths in
# this project).
DETECTOR_DEPTH_M: float = 1950.0

# Real detector-systematics hypersurface reference points (readme.md /
# Table IV of the publication): nominal (nu_e/nu_mu/nu_tau simulation
# baseline) and best-fit (the paper's own fitted point). Held fixed at
# BESTFIT_SYSTEMATICS in this package's forward model rather than
# refit as free nuisance parameters -- see detector.icecube.event_rate's
# module docstring for that scope decision.
NOMINAL_SYSTEMATICS: dict[str, float] = {
    "dom_eff": 1.00,
    "hole_ice_p0": 0.10,
    "hole_ice_p1": -0.05,
    "bulk_ice_abs": 1.00,
    "bulk_ice_scatter": 1.00,
}
BESTFIT_SYSTEMATICS: dict[str, float] = {
    "dom_eff": 1.06,
    "hole_ice_p0": -0.27,
    "hole_ice_p1": -0.04,
    "bulk_ice_abs": 0.97,
    "bulk_ice_scatter": 0.99,
}

# Real solar/reactor oscillation inputs (NuFit 6.1, normal ordering),
# fixed rather than re-fit from atmospheric-only data -- see
# tpeanuts.inference.model_atmosphere's module docstring.
THETA12_DEG: float = 33.41
THETA13_DEG: float = 8.58
DELTA13_DEG: float = 197.0
DELTAMSQ21: float = 7.41e-5

# The publication's own best-fit point (Table III, normal ordering), for
# reference/comparison only -- never used as a prior in this package's fit.
PUBLISHED_SIN2_THETA23: float = 0.51
PUBLISHED_DELTAMSQ32: float = 2.41e-3
# DeltamSq3l (this project's Delta m^2_3l = m3^2 - m1^2 convention for
# normal ordering) = DeltamSq32 + DeltamSq21, the same conversion already
# used for Daya Bay's own published truth (see
# notebooks/inference/inference4_dayabay.ipynb Section 2).
PUBLISHED_DELTAMSQ3L: float = PUBLISHED_DELTAMSQ32 + DELTAMSQ21
# Atmospheric-muon-background normalization scale at the paper's own best
# fit (Table IV); this package's forward model leaves it free by default.
PUBLISHED_MUON_SCALE: float = 1.39
