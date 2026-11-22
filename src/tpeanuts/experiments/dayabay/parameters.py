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

"""``DayaBayParameters``: the dense, loaded-once bundle of Daya Bay geometry/data-release constants.

Ported from v1_0's ``experiments/dayabay/parameters.py``, which instead
loaded everything as bare module-level constants at import time,
hardcoded to ``dtype=torch.float64`` with no ``device`` at all for most
values -- incompatible with this project's own ``RuntimeContext``
convention (every tensor explicit about device/dtype, no import-time
global state). ``DayaBayParameters.default(context=...)`` is the
replacement: a frozen dataclass built lazily, on request, on whatever
device/dtype the caller needs, mirroring
``tpeanuts.source.atmosphere.atmosphere.AtmosphericSource.default``/
``tpeanuts.source.solar.solar.SolarSource.default``'s own
"call every loader once, bundle the results" shape.

``DETECTORS``/``REACTORS``/``BG_CATEGORIES``/``NEAR_DETECTORS``/
``FAR_DETECTORS`` stay plain module-level constants (fixed names, not
loaded data).

Module contents:
    DETECTORS, REACTORS, BG_CATEGORIES
        Names, in the data release's own order.
    NEAR_DETECTORS, FAR_DETECTORS
        Experimental-hall grouping (EH1+EH2 vs EH3).
    DayaBayParameters
        Frozen dataclass bundling every loaded constant, with a
        ``.default(context=...)`` classmethod.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from tpeanuts.experiments.dayabay.io import (
    load_background_rates,
    load_baselines,
    load_detector_efficiency,
    load_eres_parameters,
    load_final_erec_bin_edges,
    load_global_normalization,
    load_iav_matrix,
    load_ibd_constants,
    load_lsnl_curve,
    load_lsnl_curve_pulls,
    load_n_protons,
    load_neutrino_rate_weekly,
    load_snf_correction,
)
from tpeanuts.source.reactor.io import (
    load_huber_mueller_spectra,
    load_nonequilibrium_correction,
    load_reactor_parameters,
)
from tpeanuts.util.context import RuntimeContext

Tensor = torch.Tensor

DETECTORS: Tuple[str, ...] = ("AD11", "AD12", "AD21", "AD22", "AD31", "AD32", "AD33", "AD34")
REACTORS: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6")
BG_CATEGORIES: Tuple[str, ...] = ("accidentals", "alpha_neutron", "amc", "fast_neutrons", "lithium_helium")

# Experimental-hall grouping (detector name's first digit = hall number):
# EH1 (AD11/AD12) and EH2 (AD21/AD22) are the two near halls (flux-weighted
# baselines ~560/600 m); EH3 (AD31-AD34) is the far hall (~1640 m).
NEAR_DETECTORS: Tuple[str, ...] = ("AD11", "AD12", "AD21", "AD22")
FAR_DETECTORS: Tuple[str, ...] = ("AD31", "AD32", "AD33", "AD34")

# True antineutrino energy grid: the tabulated Huber-Mueller range relevant
# to IBD (threshold ~1.8 MeV up to where the flux is negligible).
_E_NU_GRID_MIN_MEV = 1.8
_E_NU_GRID_MAX_MEV = 9.0
_E_NU_GRID_N = 145

# Reconstructed/true prompt-energy grid, matching the observed spectrum's
# own bin edges (0.05 MeV steps, 0-12 MeV, 241 points).
_T_GRID_MIN_MEV = 0.0
_T_GRID_MAX_MEV = 12.0
_T_GRID_N = 241


def _reactor_relative_weights(weekly: Dict[str, List[Tuple[int, float]]]) -> Dict[str, float]:
    """Per-reactor relative power/output weight for the 8AD period.

    Mean ``neutrino_rate_per_s`` over weeks with ``n_det == 8``,
    normalized to a 6-reactor average of 1.
    """
    mean_rate = {
        reactor: sum(rate for n_det, rate in entries if n_det == 8) / sum(1 for n_det, _ in entries if n_det == 8)
        for reactor, entries in weekly.items()
    }
    overall_mean = sum(mean_rate.values()) / len(mean_rate)
    return {reactor: rate / overall_mean for reactor, rate in mean_rate.items()}


def _background_category_sigma(rates: Dict[str, Dict[str, float]], categories: Tuple[str, ...]) -> Dict[str, float]:
    """Per-category background-rate fractional uncertainty, one correlated scale per category.

    The official release documents a genuine per-detector/per-hall
    correlation structure (accidentals uncorrelated between detectors,
    AmC correlated across all detectors, 9Li/8He and fast neutrons
    correlated within a hall). This project simplifies that to one
    fully correlated scale nuisance per category, with prior width

        sigma_category = sqrt(sum_d uncertainty_d^2) / sum_d rate_d,

    a rate-weighted combination of the per-detector (rate, uncertainty)
    pairs into a single relative uncertainty on that category's total
    rate.
    """
    sigmas: Dict[str, float] = {}
    for category in categories:
        rate = rates[f"{category}_rate"]
        uncertainty = rates[f"{category}_uncertainty"]
        sigmas[category] = sum(u ** 2 for u in uncertainty.values()) ** 0.5 / sum(rate.values())
    return sigmas


@dataclass(frozen=True)
class DayaBayParameters:
    """Every Daya Bay geometry/data-release constant, bundled on one device/dtype.

    Attributes:
        context: RuntimeContext every tensor here is built on.
        baselines_km: ``{detector: {reactor: baseline_km}}``.
        fission_fractions, energy_per_fission_MeV: ``{isotope: value}``,
            isotope in ``tpeanuts.source.reactor.reactor.ISOTOPES``.
        thermal_power_gw: Scalar float, GW, shared by all 6 reactor cores.
        huber_mueller_curves: ``{isotope: (E_MeV, N_per_fission_per_MeV)}``,
            the raw tabulated per-isotope spectra (see
            ``tpeanuts.source.reactor.reactor.ISOTOPES``) -- before the
            non-equilibrium correction ``tpeanuts.experiments.dayabay.flux``
            folds into U235/Pu239/Pu241.
        n_protons: ``{detector: n_protons}``.
        eres_a, eres_b, eres_c: Energy-resolution formula coefficients.
        final_erec_bin_edges_MeV: Analysis binning, shape ``(27,)``.
        detector_efficiency: IBD-selection efficiency (a bare scalar).
        e_nu_grid_MeV: True antineutrino energy grid, shape ``(145,)``.
        T_grid_MeV: True/reconstructed prompt-energy grid (0.05 MeV
            bins), shape ``(241,)``.
        ibd_constants: ``{"f": ..., "g": ..., "f2": ..., "PhaseSpaceFactor": ...}``.
        iav_matrix: IAV energy-redistribution matrix, shape ``(240, 240)``.
        lsnl_curve_E_MeV, lsnl_curve_F: Nominal LSNL curve.
        lsnl_curve_pulls: 4 ``(E_MeV, f_k)`` pull-curve pairs.
        nonequilibrium_correction: ``{isotope: (E_MeV, C)}`` (U235/Pu239/Pu241 only).
        snf_correction: ``{reactor: (E_MeV, C)}``, all 6 cores.
        reactor_relative_weight: ``{reactor: weight}``, 8AD-period, mean 1.
        global_normalization_nominal: Daya Bay's own nominal
            global-normalization value (1.0).
        background_category_sigma: ``{category: sigma}``, prior width for
            the background-rate nuisance parameters.

    Raises:
        ValueError: If ``baselines_km``/``n_protons`` is missing an entry
            for one of ``DETECTORS``, or a detector's own ``baselines_km``
            entry is missing one of ``REACTORS``.
    """

    context: RuntimeContext
    baselines_km: Dict[str, Dict[str, Tensor]]
    fission_fractions: Dict[str, float]
    energy_per_fission_MeV: Dict[str, float]
    thermal_power_gw: float
    huber_mueller_curves: Dict[str, Tuple[Tensor, Tensor]]
    n_protons: Dict[str, Tensor]
    eres_a: float
    eres_b: float
    eres_c: float
    final_erec_bin_edges_MeV: Tensor
    detector_efficiency: float
    e_nu_grid_MeV: Tensor
    T_grid_MeV: Tensor
    ibd_constants: Dict[str, float]
    iav_matrix: Tensor
    lsnl_curve_E_MeV: Tensor
    lsnl_curve_F: Tensor
    lsnl_curve_pulls: List[Tuple[Tensor, Tensor]]
    nonequilibrium_correction: Dict[str, Tuple[Tensor, Tensor]]
    snf_correction: Dict[str, Tuple[Tensor, Tensor]]
    reactor_relative_weight: Dict[str, float]
    global_normalization_nominal: float
    background_category_sigma: Dict[str, float]

    def __post_init__(self) -> None:
        missing_detectors = set(DETECTORS) - set(self.baselines_km)
        if missing_detectors:
            raise ValueError(f"baselines_km is missing detector(s): {sorted(missing_detectors)}.")
        for detector in DETECTORS:
            missing_reactors = set(REACTORS) - set(self.baselines_km[detector])
            if missing_reactors:
                raise ValueError(f"baselines_km[{detector!r}] is missing reactor(s): {sorted(missing_reactors)}.")
        missing_n_protons = set(DETECTORS) - set(self.n_protons)
        if missing_n_protons:
            raise ValueError(f"n_protons is missing detector(s): {sorted(missing_n_protons)}.")

    @classmethod
    def default(cls, *, context: RuntimeContext) -> "DayaBayParameters":
        """Load every Daya Bay constant from the cached data release, on ``context``.

        Args:
            context: RuntimeContext every returned tensor is built on.

        Returns:
            DayaBayParameters bundling every loaded constant.
        """
        fission_fractions, energy_per_fission_MeV, thermal_power_gw = load_reactor_parameters()
        eres_a, eres_b, eres_c = load_eres_parameters()
        lsnl_curve_E_MeV, lsnl_curve_F = load_lsnl_curve(context=context)
        background_rates = load_background_rates()

        return cls(
            context=context,
            baselines_km=load_baselines(context=context),
            fission_fractions=fission_fractions,
            energy_per_fission_MeV=energy_per_fission_MeV,
            thermal_power_gw=thermal_power_gw,
            huber_mueller_curves=load_huber_mueller_spectra(context=context),
            n_protons=load_n_protons(context=context),
            eres_a=eres_a, eres_b=eres_b, eres_c=eres_c,
            final_erec_bin_edges_MeV=load_final_erec_bin_edges(context=context),
            detector_efficiency=load_detector_efficiency(),
            e_nu_grid_MeV=torch.linspace(
                _E_NU_GRID_MIN_MEV, _E_NU_GRID_MAX_MEV, _E_NU_GRID_N, device=context.device, dtype=context.dtype,
            ),
            T_grid_MeV=torch.linspace(
                _T_GRID_MIN_MEV, _T_GRID_MAX_MEV, _T_GRID_N, device=context.device, dtype=context.dtype,
            ),
            ibd_constants=load_ibd_constants(),
            iav_matrix=load_iav_matrix(context=context),
            lsnl_curve_E_MeV=lsnl_curve_E_MeV,
            lsnl_curve_F=lsnl_curve_F,
            lsnl_curve_pulls=load_lsnl_curve_pulls(context=context),
            nonequilibrium_correction=load_nonequilibrium_correction(context=context),
            snf_correction=load_snf_correction(context=context),
            reactor_relative_weight=_reactor_relative_weights(load_neutrino_rate_weekly()),
            global_normalization_nominal=load_global_normalization(),
            background_category_sigma=_background_category_sigma(background_rates, BG_CATEGORIES),
        )
