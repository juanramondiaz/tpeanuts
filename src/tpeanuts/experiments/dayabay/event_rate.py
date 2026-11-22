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

"""Composed Daya Bay IBD event-rate function: multi-reactor fold per detector.

``ibd_event_rate`` is where this package's 6-reactor geometry, flux, IBD
cross section, response and background come together: each detector
sees 6 reactor cores at 6 different baselines,

    flux_e(E) = sum_r Phi_detector(E, L_{r,d}) * P_ee(E, L_{r,d})

with ``Phi_detector`` from ``tpeanuts.experiments.dayabay.flux.flux_at_detector``
and ``P_ee(E, L_r)`` supplied by the caller as one tensor per reactor,
already oscillated (oscillation itself is not computed here; see
``tpeanuts.experiments.dayabay.inference_model``). Unlike v1_0's own
version of this function (a Python ``for reactor in REACTORS`` loop,
each iteration calling ``flux_at_detector`` once), ``p_ee_per_reactor``
here is a single ``(n_reactor, n_E)`` tensor (``REACTORS`` order), so
the whole reactor sum is one batched
``tpeanuts.experiments.dayabay.flux.flux_at_detector`` call plus one
weighted reduction (see that module's own docstring for why this is not
just a mechanical reshuffle -- it also computes the reactor-independent
part of the flux, ``reactor_spectrum_shape``, exactly once).

Notes:
    - The IBD-selection efficiency is applied as a flat factor on top of
      the effective livetime. The absolute prediction is not exactly
      calibrated to the observed total (a ~20% gap remains even with
      every correction applied); Daya Bay's own analysis leaves this
      open too, via the free ``global_normalization`` nuisance --
      ``signal_scale`` below is that same official knob, not an ad hoc
      rescaling.
    - The observed IBD spectrum and background-shape histograms are
      discrete per-bin counts/probabilities, not densities, so rebinning
      them onto the analysis edges is a plain per-bin sum
      (``rebin_discrete_counts``), not the trapezoidal-density integral
      ``tpeanuts.detector.common.event_rate.bin_counts`` uses for the
      continuous folded signal spectrum.

Module contents:
    rebin_discrete_counts(...)
        Sum fine (0.05 MeV) bins into the analysis bin edges.
    real_background_counts(...)
        Background rate x shape x exposure, rebinned.
    real_observed_counts(...)
        Observed IBD spectrum, rebinned.
    ibd_detector(...)
        One Daya Bay detector as a ``FoldedDetector``: IBD cross section, LSNL/IAV
        response, flat efficiency, exposure and background. ``ibd_detector(...).observe(flux)``
        is the chain's last link for the ``SourceFlux`` of the six cores that has reached it.
    ibd_event_rate(...)
        The full multi-reactor signal-plus-background fold, one detector
        at a time, from bare per-reactor survival probabilities.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from tpeanuts.detector.common.response import DetectorResponse
from tpeanuts.detector.folded import FoldedDetector
from tpeanuts.detector.ibd_cross_section import ibd_cross_section_grid_precise
from tpeanuts.experiments.dayabay.flux import flux_at_detector
from tpeanuts.experiments.dayabay.io import (
    load_background_rates,
    load_background_shape,
    load_exposure,
    load_ibd_spectrum,
)
from tpeanuts.experiments.dayabay.parameters import BG_CATEGORIES, REACTORS, DayaBayParameters
from tpeanuts.experiments.dayabay.response import response_matrix
from tpeanuts.util.tensor import TensorLike, as_tensor

Tensor = torch.Tensor

_SECONDS_PER_DAY = 86_400.0


def rebin_discrete_counts(
    fine_low_MeV: Tensor, fine_high_MeV: Tensor, fine_counts: Tensor, bin_edges_MeV: Tensor,
) -> Tensor:
    """Sum discrete fine-bin counts into coarser, possibly non-uniform, bins.

    Every Daya Bay analysis bin edge lands exactly on a 0.05 MeV
    fine-bin boundary, so a fine bin is assigned to exactly one coarse
    bin by its center; no fine bin straddles two coarse bins.

    Args:
        fine_low_MeV, fine_high_MeV: Fine-bin edges, shape ``(n_fine,)``.
        fine_counts: Fine-bin counts (or probabilities), shape
            ``(n_fine,)``.
        bin_edges_MeV: Target (coarse) bin edges, shape ``(n_bins + 1,)``.

    Returns:
        Tensor shaped ``(n_bins,)``.
    """
    fine_center = 0.5 * (fine_low_MeV + fine_high_MeV)
    n_bins = bin_edges_MeV.shape[0] - 1
    out = torch.empty(n_bins, dtype=fine_counts.dtype, device=fine_counts.device)
    for i in range(n_bins):
        mask = (fine_center >= bin_edges_MeV[i]) & (fine_center < bin_edges_MeV[i + 1])
        out[i] = fine_counts[mask].sum()
    return out


def real_background_counts(
    params: DayaBayParameters,
    detector: str,
    *,
    exposure_seconds: Optional[Tensor] = None,
    bin_edges_MeV: Optional[Tensor] = None,
    category_scale: Optional[Mapping[str, TensorLike]] = None,
) -> Tensor:
    """Background counts per analysis bin: rate x shape x exposure.

    Args:
        params: Loaded Daya Bay parameters.
        detector: Detector name, e.g. ``"AD11"``.
        exposure_seconds: Override exposure; None loads the 8AD-period
            value.
        bin_edges_MeV: Target analysis bin edges; None uses
            ``params.final_erec_bin_edges_MeV``.
        category_scale: Optional ``{category: scale}`` nuisance
            multiplying that category's rate (nominal 1.0); categories
            not present default to 1.0. Differentiable w.r.t. a tensor
            ``scale`` -- see
            ``DayaBayParameters.background_category_sigma`` for its
            prior width.

    Returns:
        Tensor shaped ``(n_bins,)``, summed over all 5 background
        categories.
    """
    context = params.context
    if bin_edges_MeV is None:
        bin_edges_MeV = params.final_erec_bin_edges_MeV
    if exposure_seconds is None:
        exposure_seconds = load_exposure(context=context)[detector]
    exposure_days = exposure_seconds / _SECONDS_PER_DAY

    rates = load_background_rates()
    fine_low, fine_high, _ = load_ibd_spectrum(detector, context=context)
    total_fine = torch.zeros_like(fine_low)
    for category in BG_CATEGORIES:
        rate_per_day = rates[f"{category}_rate"][detector]
        shape = load_background_shape(category, detector, context=context)
        scale = 1.0 if category_scale is None else category_scale.get(category, 1.0)
        total_fine = total_fine + as_tensor(scale, device=fine_low.device, dtype=fine_low.dtype) * (
            rate_per_day * exposure_days * shape
        )

    return rebin_discrete_counts(fine_low, fine_high, total_fine, bin_edges_MeV)


def real_observed_counts(
    params: DayaBayParameters, detector: str, *, bin_edges_MeV: Optional[Tensor] = None,
) -> Tensor:
    """Observed IBD candidate counts per analysis bin (signal + background, as recorded).

    Args:
        params: Loaded Daya Bay parameters.
        detector: Detector name, e.g. ``"AD11"``.
        bin_edges_MeV: Target analysis bin edges; None uses
            ``params.final_erec_bin_edges_MeV``.

    Returns:
        Tensor shaped ``(n_bins,)``.
    """
    if bin_edges_MeV is None:
        bin_edges_MeV = params.final_erec_bin_edges_MeV
    fine_low, fine_high, fine_counts = load_ibd_spectrum(detector, context=params.context)
    return rebin_discrete_counts(fine_low, fine_high, fine_counts, bin_edges_MeV)


def ibd_detector(
    params: DayaBayParameters,
    name: str,
    *,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    bin_edges_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_seconds: Optional[Tensor] = None,
    background_counts: Optional[Tensor] = None,
    signal_scale: TensorLike = 1.0,
    background_category_scale: Optional[Mapping[str, TensorLike]] = None,
    lsnl_pulls: Optional[Tensor] = None,
) -> FoldedDetector:
    """One Daya Bay detector as a ``Detector``: ``ibd_detector(params, "AD11").observe(flux)`` gives the counts per bin.

    The arguments are those of ``ibd_event_rate`` that describe the detector (not the flux); None takes the default
    from ``params`` or the packaged data. ``signal_scale`` (the ``global_normalization`` nuisance) multiplies the
    target count, so it scales the signal and not the background.

    Args:
        params: Loaded Daya Bay parameters.
        name: Detector name, e.g. ``"AD11"``.

    Returns:
        FoldedDetector for antineutrinos, with the precise IBD cross section differential in the positron energy.
    """
    context = params.context
    if T_grid_MeV is None:
        T_grid_MeV = params.T_grid_MeV
    if Tprime_grid_MeV is None:
        Tprime_grid_MeV = params.T_grid_MeV
    if bin_edges_MeV is None:
        bin_edges_MeV = params.final_erec_bin_edges_MeV
    if n_target is None:
        n_target = params.n_protons[name]
    if exposure_seconds is None:
        exposure_seconds = load_exposure(context=context)[name]
    if background_counts is None:
        background_counts = real_background_counts(
            params, name, exposure_seconds=exposure_seconds, bin_edges_MeV=bin_edges_MeV,
            category_scale=background_category_scale,
        )
    constants = params.ibd_constants
    response = DetectorResponse(
        T_grid_MeV=T_grid_MeV,
        response_matrix=response_matrix(params, T_grid_MeV, Tprime_grid_MeV, lsnl_pulls=lsnl_pulls),
        efficiency=params.detector_efficiency * torch.ones_like(Tprime_grid_MeV),
        bin_edges_MeV=bin_edges_MeV,
    )
    return FoldedDetector(
        n_target=n_target, signal_scale=signal_scale, response=response, exposure_s=exposure_seconds,
        cross_section_e=lambda energy, grid: ibd_cross_section_grid_precise(
            energy, grid, f=constants["f"], g=constants["g"], f2=constants["f2"]),
        cross_section_x=None, context=context, antinu=True, background_counts=background_counts,
        observable="positron visible energy",
    )


def ibd_event_rate(
    params: DayaBayParameters,
    detector: str,
    p_ee_per_reactor: Tensor,
    *,
    E_nu_grid_MeV: Optional[Tensor] = None,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    bin_edges_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_seconds: Optional[Tensor] = None,
    background_counts: Optional[Tensor] = None,
    signal_scale: TensorLike = 1.0,
    background_category_scale: Optional[Mapping[str, TensorLike]] = None,
    lsnl_pulls: Optional[Tensor] = None,
) -> Tensor:
    """Predicted Daya Bay IBD counts per analysis bin for one detector, 6-reactor fold.

    Args:
        params: Loaded Daya Bay parameters.
        detector: Detector name, e.g. ``"AD11"``.
        p_ee_per_reactor: Survival probability, shape ``(n_reactor,
            n_E)``, rows in ``tpeanuts.experiments.dayabay.parameters.REACTORS``
            order, each already oscillated at that reactor's own
            baseline. Differentiable w.r.t. oscillation parameters.
        E_nu_grid_MeV: True antineutrino energy grid; None uses
            ``params.e_nu_grid_MeV``.
        T_grid_MeV: True visible prompt-energy grid (0.05 MeV bins);
            None uses ``params.T_grid_MeV``.
        Tprime_grid_MeV: Reconstructed prompt-energy grid; None uses
            ``params.T_grid_MeV`` (must equal ``T_grid_MeV``).
        bin_edges_MeV: Analysis bin edges; None uses
            ``params.final_erec_bin_edges_MeV``.
        n_target: Target free-proton count; None uses
            ``params.n_protons[detector]``.
        exposure_seconds: None loads ``io.load_exposure()[detector]``.
        background_counts: None computes it via
            ``real_background_counts``. If given explicitly,
            ``background_category_scale`` is ignored.
        signal_scale: Daya Bay's own ``global_normalization`` nuisance
            (see module docstring), multiplying the predicted signal
            only (not ``background_counts``, which is separately
            measured). Defaults to 1.0, unscaled.
        background_category_scale: Optional per-category background
            -rate nuisance, forwarded to ``real_background_counts`` when
            ``background_counts`` is None.
        lsnl_pulls: Optional LSNL pull-curve nuisances, shape ``(4,)``;
            None uses the nominal LSNL curve unchanged.

    Returns:
        Predicted counts per analysis bin, shape ``(n_bins,)``.
    """
    if E_nu_grid_MeV is None:
        E_nu_grid_MeV = params.e_nu_grid_MeV

    baselines_km = torch.stack([params.baselines_km[detector][reactor] for reactor in REACTORS])
    flux_per_reactor = flux_at_detector(params, E_nu_grid_MeV, baselines_km, list(REACTORS))  # (n_reactor, n_E)
    flux_e = (flux_per_reactor * p_ee_per_reactor).sum(dim=0)  # (n_E,)
    flux_x = torch.zeros_like(flux_e)  # IBD is nu_e_bar-only.

    folded = ibd_detector(
        params, detector, T_grid_MeV=T_grid_MeV, Tprime_grid_MeV=Tprime_grid_MeV, bin_edges_MeV=bin_edges_MeV,
        n_target=n_target, exposure_seconds=exposure_seconds, background_counts=background_counts,
        signal_scale=signal_scale, background_category_scale=background_category_scale, lsnl_pulls=lsnl_pulls,
    )
    return folded.counts(E_nu_grid_MeV, flux_e, flux_x).counts
