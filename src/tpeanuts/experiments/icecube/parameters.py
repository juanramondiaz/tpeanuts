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

"""IceCube DeepCore real analysis binning and detector constants.

Ported from v1_0's ``detector/icecube/parameters.py``. Source: IceCube
Collaboration, *Measurement of atmospheric neutrino mixing with
improved IceCube DeepCore calibration and data processing*, Phys. Rev.
D 108, 012014 (2023) (arXiv:2304.12236), and its official public data
release (Harvard Dataverse, DOI 10.7910/DVN/B4RITM) -- 9 years of
golden (track-like, up-going/near-horizontal) DeepCore events.

Every module-level constant here is a fixed value with no ``RuntimeContext``
dependency; the binning edges are the only tensors, built once with
``torch.logspace``/``torch.linspace`` directly (no numpy, unlike
v1_0's own ``np.logspace``/``np.delete``).
``IceCubeParameters.default(context=...)`` bundles them, re-cast to a
caller's ``context``, with the detector depth, the systematics reference
points and the fixed oscillation inputs -- the same shape as
``DayaBayParameters``/``BorexinoParameters``/``SNOParameters``.

Module contents:
    RECO_ENERGY_BIN_EDGES_GEV, RECO_COSZEN_BIN_EDGES, PID_BIN_EDGES
        Real analysis binning, exactly reproducing the data release's
        own ``readme.md`` recipe (log-spaced energy, one wide top bin
        merged for statistics, linear coszen, 2 PID categories).
    DETECTOR_DEPTH_M
        Approximate DeepCore depth below the South Pole ice surface.
    NOMINAL_SYSTEMATICS, BESTFIT_SYSTEMATICS
        Real detector-systematics hypersurface reference points (Table
        IV of the publication).
    THETA12_DEG, THETA13_DEG, DELTA13_DEG, DELTAMSQ21
        Fixed solar/reactor oscillation inputs (NuFit), not re-fit from
        atmospheric data.
    PUBLISHED_SIN2_THETA23, PUBLISHED_DELTAMSQ32, PUBLISHED_DELTAMSQ3L,
    PUBLISHED_MUON_SCALE
        The publication's own best-fit point (normal ordering), for
        reference/comparison only -- not used as a prior anywhere in
        this package.
    IceCubeParameters
        Frozen dataclass bundling the binning edges (on a ``RuntimeContext``), the detector depth, the systematics
        reference points and the fixed oscillation inputs above, with a ``.default(context=...)`` classmethod.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch

from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor

# Real analysis binning, from the data release's own readme.md:
#   "Energy (GeV): 10 bins from 6.31 to 158.49" (12 log-spaced edges, one
#   internal edge merged away "to contain sufficient statistics" -- the
#   release's own choice, reproduced here exactly rather than re-derived).
#   "Cosine of zenith: 11 bins from -1.0 (upgoing) to 0.1."
#   "Particle ID: [0.55, 0.75, 1.0]" (2 bins, tracks only -- the cascade
#   bin [0, 0.55] is not part of this release).
_EN_MIN, _EN_MAX = 6.31, 158.49
_CZ_MIN, _CZ_MAX = -1.0, 0.1

_full_energy_edges = torch.logspace(math.log10(_EN_MIN), math.log10(_EN_MAX), 12, dtype=torch.float64)
RECO_ENERGY_BIN_EDGES_GEV: Tensor = torch.cat([_full_energy_edges[:-2], _full_energy_edges[-1:]])
RECO_COSZEN_BIN_EDGES: Tensor = torch.linspace(_CZ_MIN, _CZ_MAX, 11, dtype=torch.float64)
PID_BIN_EDGES: Tensor = torch.tensor([0.55, 0.75, 1.0], dtype=torch.float64)

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
# refit as free nuisance parameters -- see
# tpeanuts.experiments.icecube.event_rate's own module docstring for that
# scope decision.
NOMINAL_SYSTEMATICS: Dict[str, float] = {
    "dom_eff": 1.00,
    "hole_ice_p0": 0.10,
    "hole_ice_p1": -0.05,
    "bulk_ice_abs": 1.00,
    "bulk_ice_scatter": 1.00,
}
BESTFIT_SYSTEMATICS: Dict[str, float] = {
    "dom_eff": 1.06,
    "hole_ice_p0": -0.27,
    "hole_ice_p1": -0.04,
    "bulk_ice_abs": 0.97,
    "bulk_ice_scatter": 0.99,
}

# Real solar/reactor oscillation inputs (NuFit 6.1, normal ordering),
# fixed rather than re-fit from atmospheric-only data.
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
# used for Daya Bay's own published truth.
PUBLISHED_DELTAMSQ3L: float = PUBLISHED_DELTAMSQ32 + DELTAMSQ21
# Atmospheric-muon-background normalization scale at the paper's own best
# fit (Table IV); this package's forward model leaves it free by default.
PUBLISHED_MUON_SCALE: float = 1.39


@dataclass(frozen=True)
class IceCubeParameters:
    """Every IceCube DeepCore binning/depth/systematics/oscillation constant, bundled on one device/dtype.

    Attributes:
        context: RuntimeContext the binning tensors are built on.
        reco_energy_bin_edges_GeV: Reconstructed-energy bin edges, GeV, see ``RECO_ENERGY_BIN_EDGES_GEV``.
        reco_coszen_bin_edges: Reconstructed cos(zenith) bin edges, see ``RECO_COSZEN_BIN_EDGES``.
        pid_bin_edges: Particle-ID bin edges, see ``PID_BIN_EDGES``.
        detector_depth_m: Approximate DeepCore depth, metres, see ``DETECTOR_DEPTH_M``.
        nominal_systematics, bestfit_systematics: Detector-systematics reference points, see ``NOMINAL_SYSTEMATICS``
            and ``BESTFIT_SYSTEMATICS``.
        theta12_deg, theta13_deg, delta13_deg, deltamsq21: Fixed solar/reactor oscillation inputs (NuFit 6.1), degrees
            and eV^2, see the constants of the same name.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> from tpeanuts.experiments.icecube.parameters import IceCubeParameters
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> params = IceCubeParameters.default(context=context)
        >>> params.reco_energy_bin_edges_GeV.numel() - 1, params.detector_depth_m
        (10, 1950.0)
    """

    context: RuntimeContext
    reco_energy_bin_edges_GeV: Tensor
    reco_coszen_bin_edges: Tensor
    pid_bin_edges: Tensor
    detector_depth_m: float = DETECTOR_DEPTH_M
    nominal_systematics: Dict[str, float] = field(default_factory=lambda: dict(NOMINAL_SYSTEMATICS))
    bestfit_systematics: Dict[str, float] = field(default_factory=lambda: dict(BESTFIT_SYSTEMATICS))
    theta12_deg: float = THETA12_DEG
    theta13_deg: float = THETA13_DEG
    delta13_deg: float = DELTA13_DEG
    deltamsq21: float = DELTAMSQ21

    @classmethod
    def default(cls, *, context: RuntimeContext) -> "IceCubeParameters":
        """Build the default IceCube parameter bundle on ``context``.

        Args:
            context: RuntimeContext the returned tensors are built on.

        Returns:
            IceCubeParameters with the data release's own binning and the publication's reference points.
        """
        return cls(
            context=context,
            reco_energy_bin_edges_GeV=RECO_ENERGY_BIN_EDGES_GEV.to(device=context.device, dtype=context.dtype),
            reco_coszen_bin_edges=RECO_COSZEN_BIN_EDGES.to(device=context.device, dtype=context.dtype),
            pid_bin_edges=PID_BIN_EDGES.to(device=context.device, dtype=context.dtype),
        )
