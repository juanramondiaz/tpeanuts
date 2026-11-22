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

"""SNO salt-phase (Phase II) constants: observable ordering, binning, response, hep flux.

Ported from v1_0's ``experiments/sno_ii/parameters.py``. All real values
here are transcribed from the primary source (see
``tpeanuts.experiments.sno_ii``'s own package docstring) via
``data/experiments/sno_ii/metadata/source.json``, not illustrative.

Unlike v1_0's own version, ``N_TARGET_DEUTERONS``/``N_TARGET_ELECTRONS``
are not re-exported here: v2's target-particle counts live inside
``tpeanuts.experiments.sno.parameters.SNOParameters`` (a
``RuntimeContext``-bound dataclass, not bare module-level constants),
so ``tpeanuts.experiments.sno_ii.inference_model`` builds its own
``SNOParameters`` and reads ``.n_target_deuterons``/
``.n_target_electrons`` from it directly instead.

Module contents:
    CHANNEL_ORDER
        The 19-entry observable ordering (NC, CC1..CC17, ES) used
        consistently by every table in ``data/experiments/sno_ii/`` and
        by ``tpeanuts.experiments.sno_ii.inference_model``'s predicted
        vector.
    N_CC_BINS, N_CHANNELS, N_OBSERVABLES_PER_PERIOD, N_OBSERVABLES_TOTAL
        Dimensions of the observable vector (17 CC bins + NC + ES = 19
        per period; day and night = 38 total).
    CC_BIN_EDGES_MEV
        The 17 CC recoil-electron kinetic-energy bin edges (Table
        XXX): 16 bins of 0.5 MeV from 5.5-13.5 MeV, plus one wide
        13.5-20.0 MeV bin.
    ENERGY_RESOLUTION_C0, ENERGY_RESOLUTION_C1, ENERGY_RESOLUTION_C2
        Salt-phase kinetic-energy resolution parametrization
        coefficients, ``sigma_T(Te) = C0 + C1*sqrt(Te) + C2*Te`` [MeV],
        Eq. A3 of the primary source -- distinct from (and more
        recent/precise than) ``tpeanuts.experiments.sno.parameters``'s
        Phase-I coefficients; do not conflate the two.
    ENERGY_RESOLUTION_FLOOR_MEV
        Numerical floor on sigma(T), not a physical parameter.
    T_GRID_MEV, TPRIME_GRID_MEV
        Default true/reconstructed electron kinetic-energy integration
        grids, spanning ``CC_BIN_EDGES_MEV``'s own range.
    E_NU_GRID_MEV
        Default true neutrino energy integration grid.
    ANALYSIS_THRESHOLD_MEV
        SNO salt-phase analysis threshold (5.5 MeV kinetic energy),
        the lower edge of ``CC_BIN_EDGES_MEV`` and the equivalent-flux
        denominator's own integration range in Eq. A1 of the primary
        source.
    HEP_FLUX_CM2S
        Fixed hep solar-neutrino flux SNO's own salt-phase analysis
        assumed (not a free parameter), verified at the primary
        source.
    DETECTOR_DEPTH_M
        Same physical detector as Phase I, see
        ``tpeanuts.experiments.sno.inference_model.SNOModel``.
    SNOPhaseIIParameters
        Frozen dataclass bundling the grids, bin edges, resolution coefficients, depth, hep flux and analysis
        threshold above on one ``RuntimeContext``, with a ``.default(context=...)`` classmethod -- the same shape as
        ``tpeanuts.experiments.sno.parameters.SNOParameters`` (which still owns the target-particle counts).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor

# Observable ordering, matching every table in data/experiments/sno_ii/ and
# Tables XXX/XXXII/XXXIII/XXXIV of the primary source exactly.
CHANNEL_ORDER = (
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
CC_BIN_EDGES_MEV: Tensor = torch.cat([
    torch.arange(5.5, 13.5001, 0.5, dtype=torch.float64),
    torch.tensor([20.0], dtype=torch.float64),
])

# Eq. A3 of the primary source: sigma_T(Te) = -0.131 + 0.383*sqrt(Te) + 0.03731*Te [MeV].
ENERGY_RESOLUTION_C0: float = -0.131
ENERGY_RESOLUTION_C1: float = 0.383
ENERGY_RESOLUTION_C2: float = 0.03731

# Numerical floor on sigma(T), MeV -- as in experiments.sno.parameters, this
# formula goes non-positive below Te ~= 0.12 MeV (well under the 5.5 MeV
# analysis threshold), so this only guards the low-T tail of T_GRID_MEV
# against feeding a non-positive sigma into gaussian_response_matrix; it
# carries no physical meaning.
ENERGY_RESOLUTION_FLOOR_MEV: float = 1.0e-3

# Default true/reconstructed electron-energy integration grids, spanning
# the CC spectrum's own range (see CC_BIN_EDGES_MEV above).
T_GRID_MEV: Tensor = torch.linspace(0.0, 20.0, 500, dtype=torch.float64)
TPRIME_GRID_MEV: Tensor = T_GRID_MEV
E_NU_GRID_MEV: Tensor = torch.linspace(1.0, 20.0, 400, dtype=torch.float64)

# Table XXX's own quoted analysis threshold: CC_BIN_EDGES_MEV[0].
ANALYSIS_THRESHOLD_MEV: float = 5.5

# Fixed hep contribution assumed throughout SNO's own salt-phase flux
# extraction (Section XI of the primary source): "a fixed hep contribution
# of 9.3e3 cm^-2 s^-1 has been assumed", in cm^-2 s^-1.
HEP_FLUX_CM2S: float = 9.3e3

# Same physical detector as Phase I; see
# tpeanuts.experiments.sno.inference_model.SNOModel's own caveat about
# this being an approximate, not primary-source-verified, depth.
DETECTOR_DEPTH_M: float = 2039.0


@dataclass(frozen=True)
class SNOPhaseIIParameters:
    """Every SNO salt-phase grid/binning/response constant, bundled on one device/dtype.

    Attributes:
        context: RuntimeContext every tensor here is built on.
        e_nu_grid_MeV: True neutrino energy grid, shape ``(400,)``, see ``E_NU_GRID_MEV``.
        T_grid_MeV: True electron kinetic-energy grid, shape ``(500,)``, see ``T_GRID_MEV``.
        Tprime_grid_MeV: Reconstructed electron kinetic-energy grid, equal to ``T_grid_MeV``.
        cc_bin_edges_MeV: The 18 edges of the 17 CC bins, see ``CC_BIN_EDGES_MEV``.
        detector_depth_m: Detector depth below the surface, metres, see ``DETECTOR_DEPTH_M``.
        hep_flux_cm2s: Fixed hep flux, cm^-2 s^-1, see ``HEP_FLUX_CM2S``.
        analysis_threshold_MeV: Analysis threshold, see ``ANALYSIS_THRESHOLD_MEV``.
        energy_resolution_c0, energy_resolution_c1, energy_resolution_c2: Resolution-formula coefficients, see
            ``ENERGY_RESOLUTION_C0``/``C1``/``C2``.
        energy_resolution_floor_MeV: See ``ENERGY_RESOLUTION_FLOOR_MEV``.

    Examples:
        >>> import torch
        >>> import tpeanuts as tp
        >>> from tpeanuts.experiments.sno_ii.parameters import SNOPhaseIIParameters
        >>> context = tp.RuntimeContext.resolve("cpu", torch.float64)
        >>> params = SNOPhaseIIParameters.default(context=context)
        >>> tuple(params.cc_bin_edges_MeV.shape), params.analysis_threshold_MeV
        ((18,), 5.5)
    """

    context: RuntimeContext
    e_nu_grid_MeV: Tensor
    T_grid_MeV: Tensor
    Tprime_grid_MeV: Tensor
    cc_bin_edges_MeV: Tensor
    detector_depth_m: float = DETECTOR_DEPTH_M
    hep_flux_cm2s: float = HEP_FLUX_CM2S
    analysis_threshold_MeV: float = ANALYSIS_THRESHOLD_MEV
    energy_resolution_c0: float = ENERGY_RESOLUTION_C0
    energy_resolution_c1: float = ENERGY_RESOLUTION_C1
    energy_resolution_c2: float = ENERGY_RESOLUTION_C2
    energy_resolution_floor_MeV: float = ENERGY_RESOLUTION_FLOOR_MEV

    @classmethod
    def default(cls, *, context: RuntimeContext) -> "SNOPhaseIIParameters":
        """Build the default salt-phase parameter bundle on ``context``.

        Args:
            context: RuntimeContext every returned tensor is built on.

        Returns:
            SNOPhaseIIParameters with the published grids, bin edges and constants.
        """
        T_grid_MeV = T_GRID_MEV.to(device=context.device, dtype=context.dtype)
        return cls(
            context=context,
            e_nu_grid_MeV=E_NU_GRID_MEV.to(device=context.device, dtype=context.dtype),
            T_grid_MeV=T_grid_MeV,
            Tprime_grid_MeV=T_grid_MeV,
            cc_bin_edges_MeV=CC_BIN_EDGES_MEV.to(device=context.device, dtype=context.dtype),
        )
