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
Composed Daya Bay IBD event-rate function: real multi-reactor fold per detector.

``ibd_event_rate`` is the one place this package's real 6-reactor geometry,
real flux, real IBD cross section, real response, and real background come
together: each detector sees **6 real reactor cores at 6 real, different
baselines**:

    flux_e(E) = sum_r Phi_detector(E, L_{r,d}) * P_ee(E, L_{r,d})

with ``Phi_detector`` from ``detector.dayabay.flux.flux_at_detector`` and
``P_ee(E, L_r)`` supplied by the caller (one tensor per reactor, already
oscillated at that reactor's own baseline -- oscillation itself is not
computed here, matching every other detector package in this project;
see ``tpeanuts.detector.dayabay.inference_model.DayaBayDetectorModel``,
which computes the 6 ``P_ee`` curves via
``tpeanuts.medium.vacuum.oscillation_model.VacuumOscillationModel``).

The real IBD-selection efficiency (``detector.dayabay.parameters
.DETECTOR_EFFICIENCY``, Gd-capture + delayed-coincidence + analysis cuts)
is applied here as a flat efficiency factor, separate from and on top of
the effective livetime, following the official data-release parameterization.
The absolute prediction is not exactly calibrated to the observed total (a
real ~20% gap remains even with every correction below applied -- Daya
Bay's own official analysis does not fix this either, instead leaving a
free ``global_normalization`` nuisance parameter,
``detector.dayabay.parameters.GLOBAL_NORMALIZATION_NOMINAL`` = 1.0, in its
combined fit; ``signal_scale`` below is that same real, official knob, not
an ad hoc rescaling introduced by this project -- see
``detector.dayabay.inference_model``).

The cross section (``detector.interaction.inverse_beta_decay
.ibd_cross_section_grid_precise``, Daya Bay's own real coupling constants
via ``detector.dayabay.parameters.IBD_CONSTANTS``), the response
(``detector.dayabay.response.response_matrix``, real IAV + LSNL + Gaussian
resolution chain), and the flux (``detector.dayabay.flux.flux_at_detector``,
real per-isotope non-equilibrium and per-reactor SNF/relative-power
corrections) are each real, per-reactor where applicable.

The real observed IBD spectrum and real background-shape histograms are
**discrete per-bin counts/probabilities** (confirmed: each column sums to
exactly 1 for the background shapes, and to the real integer candidate
count for the IBD spectra), not densities -- so rebinning them onto the
real (non-uniform) analysis edges is a plain per-bin sum
(``rebin_discrete_counts``), not the trapezoidal-density integral
``detector.common.event_rate.bin_counts`` performs for the (continuous)
folded *signal* spectrum.

Module contents:
    rebin_discrete_counts(...)
        Sum fine (0.05 MeV) bins into the real analysis bin edges.
    real_background_counts(...)
        Real background rate x real shape x real exposure, rebinned.
    real_observed_counts(...)
        Real observed IBD spectrum, rebinned.
    ibd_event_rate(...)
        The full real multi-reactor signal-plus-background fold, one
        detector at a time.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.event_rate import predicted_counts
from tpeanuts.detector.dayabay.flux import flux_at_detector
from tpeanuts.detector.dayabay.io import (
    load_background_rates,
    load_background_shape,
    load_exposure,
    load_ibd_spectrum,
)
from tpeanuts.detector.dayabay.parameters import (
    BASELINES_KM,
    BG_CATEGORIES,
    DETECTOR_EFFICIENCY,
    E_NU_GRID_MEV,
    FINAL_EREC_BIN_EDGES_MEV,
    IBD_CONSTANTS,
    N_PROTONS,
    REACTORS,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)
from tpeanuts.detector.dayabay.response import response_matrix
from tpeanuts.detector.interaction.inverse_beta_decay import ibd_cross_section_grid_precise
from tpeanuts.util.type import TensorLike, as_tensor

_SECONDS_PER_DAY = 86_400.0


def rebin_discrete_counts(
    fine_low_MeV: torch.Tensor,
    fine_high_MeV: torch.Tensor,
    fine_counts: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
) -> torch.Tensor:
    """Sum discrete fine-bin counts into coarser, possibly non-uniform, bins.

    Every real Daya Bay analysis bin edge lands exactly on a 0.05 MeV fine-
    bin boundary (verified against ``final_erec_bin_edges.csv``), so a fine
    bin is assigned to exactly one coarse bin by its center; no fine bin
    straddles two coarse bins.

    Args:
        fine_low_MeV, fine_high_MeV: Fine-bin edges, shape ``(n_fine,)``.
        fine_counts: Fine-bin counts (or probabilities), shape ``(n_fine,)``.
        bin_edges_MeV: Target (coarse) bin edges, shape ``(n_bins + 1,)``.

    Returns:
        Real tensor shaped ``(n_bins,)``.
    """
    fine_center = 0.5 * (fine_low_MeV + fine_high_MeV)
    n_bins = bin_edges_MeV.shape[0] - 1
    out = torch.empty(n_bins, dtype=fine_counts.dtype, device=fine_counts.device)
    for i in range(n_bins):
        mask = (fine_center >= bin_edges_MeV[i]) & (fine_center < bin_edges_MeV[i + 1])
        out[i] = fine_counts[mask].sum()
    return out


def real_background_counts(
    detector: str,
    *,
    exposure_seconds: Optional[torch.Tensor] = None,
    bin_edges_MeV: torch.Tensor = FINAL_EREC_BIN_EDGES_MEV,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    category_scale: Optional[dict[str, TensorLike]] = None,
) -> torch.Tensor:
    """Real background counts per analysis bin: real rate x real shape x real exposure.

    Args:
        detector: Detector name, e.g. "AD11".
        exposure_seconds: Override exposure; None loads the real 8AD-period
            value (``detector.dayabay.io.load_exposure``).
        bin_edges_MeV: Target analysis bin edges.
        device, dtype: Target tensor device/dtype.
        category_scale: Optional ``{category: scale}`` nuisance multiplying
            that category's real rate (nominal 1.0); categories not present
            default to 1.0. Differentiable w.r.t. a tensor ``scale`` --
            see ``detector.dayabay.parameters.BACKGROUND_CATEGORY_SIGMA``
            for the real prior uncertainty to constrain it with.

    Returns:
        Real tensor shaped ``(n_bins,)``, summed over all 5 real background
        categories.
    """
    rates = load_background_rates()
    if exposure_seconds is None:
        exposure_seconds = load_exposure(device=device, dtype=dtype)[detector]
    exposure_days = exposure_seconds / _SECONDS_PER_DAY

    fine_low, fine_high, _ = load_ibd_spectrum(detector, device=device, dtype=dtype)
    total_fine = torch.zeros_like(fine_low)
    for category in BG_CATEGORIES:
        rate_per_day = float(rates.loc[f"{category}_rate", detector])
        shape = load_background_shape(category, detector, device=device, dtype=dtype)
        scale = 1.0 if category_scale is None else category_scale.get(category, 1.0)
        total_fine = total_fine + as_tensor(scale, device=fine_low.device, dtype=dtype) * (
            rate_per_day * exposure_days * shape
        )

    return rebin_discrete_counts(fine_low, fine_high, total_fine, bin_edges_MeV)


def real_observed_counts(
    detector: str,
    *,
    bin_edges_MeV: torch.Tensor = FINAL_EREC_BIN_EDGES_MEV,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Real observed IBD candidate counts per analysis bin (signal + background, as recorded).

    Args:
        detector: Detector name, e.g. "AD11".
        bin_edges_MeV: Target analysis bin edges.
        device, dtype: Target tensor device/dtype.

    Returns:
        Real tensor shaped ``(n_bins,)``.
    """
    fine_low, fine_high, fine_counts = load_ibd_spectrum(detector, device=device, dtype=dtype)
    return rebin_discrete_counts(fine_low, fine_high, fine_counts, bin_edges_MeV)


def ibd_event_rate(
    detector: str,
    p_ee_per_reactor: dict[str, torch.Tensor],
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    bin_edges_MeV: torch.Tensor = FINAL_EREC_BIN_EDGES_MEV,
    n_target: Optional[torch.Tensor] = None,
    exposure_seconds: Optional[torch.Tensor] = None,
    background_counts: Optional[torch.Tensor] = None,
    signal_scale: TensorLike = 1.0,
    background_category_scale: Optional[dict[str, TensorLike]] = None,
    lsnl_pulls: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predicted Daya Bay IBD counts per analysis bin for one detector, real 6-reactor fold.

    Args:
        detector: Detector name, e.g. "AD11".
        p_ee_per_reactor: ``{reactor_name: P_ee(E)}``, one tensor per entry
            of ``detector.dayabay.parameters.REACTORS``, each shape
            ``(n_E,)`` on ``E_nu_grid_MeV``, already oscillated at that
            reactor's own real baseline. Differentiable w.r.t. oscillation
            parameters.
        E_nu_grid_MeV: True antineutrino energy grid.
        T_grid_MeV: True visible prompt-energy grid (real 0.05 MeV bins).
        Tprime_grid_MeV: Reconstructed prompt-energy grid; must
            equal ``T_grid_MeV`` (see ``detector.common.event_rate
            .predicted_counts``).
        bin_edges_MeV: Real analysis bin edges.
        n_target: Target free-proton count; None uses the real
            ``detector.dayabay.parameters.N_PROTONS[detector]``.
        exposure_seconds: Real exposure; None loads
            ``detector.dayabay.io.load_exposure()[detector]``.
        background_counts: Real background; None computes it via
            ``real_background_counts``. If given explicitly,
            ``background_category_scale`` is ignored (it only affects the
            internally computed default).
        signal_scale: Daya Bay's own real ``global_normalization`` nuisance
            (see module docstring), multiplying the predicted signal only
            (not ``background_counts``, which is separately measured).
            Defaults to 1.0 (``detector.dayabay.parameters
            .GLOBAL_NORMALIZATION_NOMINAL``, unscaled).
        background_category_scale: Optional real per-category background-
            rate nuisance, forwarded to ``real_background_counts`` when
            ``background_counts`` is None (see that function and
            ``detector.dayabay.parameters.BACKGROUND_CATEGORY_SIGMA``).
        lsnl_pulls: Optional real LSNL pull-curve nuisances, shape ``(4,)``,
            forwarded to ``detector.dayabay.response.response_matrix``;
            None uses the real nominal LSNL curve unchanged.

    Returns:
        Predicted counts per analysis bin, shape ``(n_bins,)``.
    """
    baselines = BASELINES_KM[detector]
    flux_e = torch.zeros_like(E_nu_grid_MeV)
    for reactor in REACTORS:
        flux_e = flux_e + flux_at_detector(
            E_nu_grid_MeV, baselines[reactor], reactor,
        ) * p_ee_per_reactor[reactor]
    flux_x = torch.zeros_like(flux_e)  # IBD is nu_e_bar-only.

    cross_section_e = ibd_cross_section_grid_precise(
        E_nu_grid_MeV, T_grid_MeV,
        f=IBD_CONSTANTS["f"], g=IBD_CONSTANTS["g"], f2=IBD_CONSTANTS["f2"],
    )
    cross_section_x = torch.zeros_like(cross_section_e)

    R = response_matrix(T_grid_MeV, Tprime_grid_MeV, lsnl_pulls=lsnl_pulls)
    efficiency = DETECTOR_EFFICIENCY * torch.ones_like(Tprime_grid_MeV)

    if n_target is None:
        n_target = N_PROTONS[detector]
    n_target_scaled = as_tensor(signal_scale, device=T_grid_MeV.device, dtype=T_grid_MeV.dtype) * n_target
    if exposure_seconds is None:
        exposure_seconds = load_exposure(device=T_grid_MeV.device, dtype=T_grid_MeV.dtype)[detector]
    if background_counts is None:
        background_counts = real_background_counts(
            detector, exposure_seconds=exposure_seconds, bin_edges_MeV=bin_edges_MeV,
            device=T_grid_MeV.device, dtype=T_grid_MeV.dtype,
            category_scale=background_category_scale,
        )

    return predicted_counts(
        E_nu_grid_MeV, flux_e, flux_x,
        cross_section_e, cross_section_x, n_target_scaled,
        T_grid_MeV, R, efficiency,
        bin_edges_MeV, exposure_seconds,
        background_counts=background_counts,
    )
