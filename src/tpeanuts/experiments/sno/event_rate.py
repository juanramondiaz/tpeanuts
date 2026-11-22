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

"""Composed SNO event-rate functions: three separate, independently composable channels.

Ported from v1_0's ``experiments/sno/event_rate.py``. SNO measures solar
neutrinos through three physically distinct reactions on the same D2O
target, and this module keeps them as three separate functions (never
auto-summed here) rather than one "total predicted spectrum", so a
caller can combine whichever subset a given analysis needs -- e.g. all
three summed against the real day/night candidate spectrum
(``tpeanuts.experiments.sno.inference_model.SNOModel``, since
SNO's own raw counts are not separated by reaction either), or
``cc_event_rate`` alone for a CC-only study:

    cc_event_rate(...)
        nu_e + d -> p + p + e-  (CC, nu_e-only). Wires
        ``detector.common.event_rate.predicted_counts`` with the real
        Nakamura et al. (2002) cross section
        (``tpeanuts.detector.deuteron``).
    es_event_rate(...)
        nu + e- -> nu + e-  (ES, all flavours, different couplings).
        Same fold as ``tpeanuts.experiments.borexino.event_rate.event_rate``,
        with SNO's own target/response/exposure -- added as a
        genuinely separate channel, not merged into ``cc_event_rate``'s
        prediction.
    nc_event_rate(...)
        nu_x + d -> p + n + nu_x  (NC, flavour-blind), followed by
        neutron capture on a second deuteron and its gamma's
        Compton-scattered electron -- see that function's own
        docstring for why this channel cannot reuse
        ``predicted_counts``'s continuum fold the way the other two do.

All three deliberately do not know about oscillation, the Sun, or the
Earth: they take flavour probabilities/``flux_tot_MeV`` as plain
tensors, built upstream (day or night, see
``tpeanuts.experiments.sno.inference_model.SNOModel``). Every
grid/target/exposure default comes from an explicit ``SNOParameters``
instance (same params-object pattern as
``tpeanuts.experiments.dayabay.event_rate.ibd_event_rate``), unlike
v1_0's own module-level constant defaults fixed at import time.

**Real neutron backgrounds stay backgrounds, not the NC signal
above.** The "neutron" column of the real published
``data/experiments/sno/observation/backgrounds.csv``, loaded by
``tpeanuts.experiments.sno.io.load_backgrounds``, is SNO's own measured
NC-neutron-capture *leakage into the CC/ES electron-energy analysis
window* -- a background correction to the CC(+ES) spectrum fit,
unrelated to ``nc_event_rate`` above (a forward-modeled prediction for
NC's own dedicated visible-energy peak). Neither function reads from
or feeds into the other.

Module contents:
    cc_detector(...), es_detector(...), nc_detector(...)
        The three channels as ``Detector`` objects (``FoldedDetector``/``FixedEnergyDetector``):
        ``cc_detector(params, bin_edges_MeV).observe(flux)`` is the chain's last link for a ``Flux`` per unit area.
    cc_event_rate(...), es_event_rate(...), nc_event_rate(...)
        See above; each builds its detector and applies it to bare probabilities and flux tensors.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.response import DetectorResponse
from tpeanuts.detector.deuteron import cc_cross_section_grid, nc_cross_section
from tpeanuts.detector.folded import FixedEnergyDetector, FoldedDetector
from tpeanuts.detector.neutrino_electron import nue_cross_section_grid, numutau_cross_section_grid
from tpeanuts.experiments.sno.parameters import SNOParameters
from tpeanuts.experiments.sno.response import energy_resolution_MeV as cc_energy_resolution_MeV
from tpeanuts.experiments.sno.response import response_matrix

Tensor = torch.Tensor
_SECONDS_PER_DAY = 86_400.0


def _response(
    params: SNOParameters, bin_edges_MeV: Tensor, T_grid_MeV: Optional[Tensor], Tprime_grid_MeV: Optional[Tensor],
    c0: Optional[float], c1: Optional[float], c2: Optional[float],
) -> DetectorResponse:
    """SNO's response (real resolution formula, no threshold) on the given or default grids."""
    if T_grid_MeV is None:
        T_grid_MeV = params.T_grid_MeV
    if Tprime_grid_MeV is None:
        Tprime_grid_MeV = params.Tprime_grid_MeV
    return DetectorResponse(
        T_grid_MeV=T_grid_MeV,
        response_matrix=response_matrix(
            T_grid_MeV, Tprime_grid_MeV,
            c0=params.energy_resolution_c0 if c0 is None else c0,
            c1=params.energy_resolution_c1 if c1 is None else c1,
            c2=params.energy_resolution_c2 if c2 is None else c2,
        ),
        efficiency=torch.ones_like(Tprime_grid_MeV),
        bin_edges_MeV=bin_edges_MeV,
    )


def cc_detector(
    params: SNOParameters,
    bin_edges_MeV: Tensor,
    *,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    c0: Optional[float] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    background_counts: Optional[Tensor] = None,
) -> FoldedDetector:
    """SNO's charged-current channel (nu_e + d -> p + p + e-) as a ``Detector``.

    The arguments are those of ``cc_event_rate`` that describe the detector; None takes the default from ``params``
    (``exposure_days`` defaults to 1 day).

    Returns:
        FoldedDetector with the deuteron target, the Nakamura cross section (nu_e only) and the real response.
    """
    return FoldedDetector(
        n_target=params.n_target_deuterons if n_target is None else n_target,
        response=_response(params, bin_edges_MeV, T_grid_MeV, Tprime_grid_MeV, c0, c1, c2),
        exposure_s=(1.0 if exposure_days is None else exposure_days) * _SECONDS_PER_DAY,
        cross_section_e=lambda energy, grid: cc_cross_section_grid(energy, grid, context=params.context),
        cross_section_x=None, context=params.context, background_counts=background_counts,
    )


def es_detector(
    params: SNOParameters,
    bin_edges_MeV: Tensor,
    *,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    c0: Optional[float] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    background_counts: Optional[Tensor] = None,
) -> FoldedDetector:
    """SNO's elastic-scattering channel (nu + e- -> nu + e-) as a ``Detector``.

    Same arguments as ``cc_detector``; ``n_target`` defaults to ``params.n_target_electrons``.

    Returns:
        FoldedDetector with the electron target, the nu-e cross sections and the real response.
    """
    return FoldedDetector(
        n_target=params.n_target_electrons if n_target is None else n_target,
        response=_response(params, bin_edges_MeV, T_grid_MeV, Tprime_grid_MeV, c0, c1, c2),
        exposure_s=(1.0 if exposure_days is None else exposure_days) * _SECONDS_PER_DAY,
        cross_section_e=nue_cross_section_grid, cross_section_x=numutau_cross_section_grid,
        context=params.context, background_counts=background_counts,
    )


def nc_detector(
    params: SNOParameters,
    bin_edges_MeV: Tensor,
    *,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    c0: Optional[float] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    capture_energy_MeV: Optional[float] = None,
    capture_efficiency: Optional[float] = None,
    background_counts: Optional[Tensor] = None,
) -> FixedEnergyDetector:
    """SNO's neutral-current channel (nu_x + d -> p + n + nu_x, neutron capture) as a ``Detector``.

    Same arguments as ``nc_event_rate`` that describe the detector.

    Returns:
        FixedEnergyDetector: flavour-blind deuteron breakup seen as a capture gamma at ``capture_energy_MeV``.
    """
    if Tprime_grid_MeV is None:
        Tprime_grid_MeV = params.Tprime_grid_MeV
    if capture_energy_MeV is None:
        capture_energy_MeV = params.nc_capture_energy_MeV
    capture_T = torch.as_tensor([capture_energy_MeV], dtype=Tprime_grid_MeV.dtype, device=Tprime_grid_MeV.device)
    resolution = cc_energy_resolution_MeV(
        capture_T,
        c0=params.energy_resolution_c0 if c0 is None else c0,
        c1=params.energy_resolution_c1 if c1 is None else c1,
        c2=params.energy_resolution_c2 if c2 is None else c2,
    )
    return FixedEnergyDetector(
        n_target=params.n_target_deuterons if n_target is None else n_target,
        sigma_total=lambda energy: nc_cross_section(energy, context=params.context),
        visible_energy_MeV=capture_energy_MeV, resolution_MeV=resolution,
        efficiency=params.nc_capture_efficiency if capture_efficiency is None else capture_efficiency,
        reco_grid_MeV=Tprime_grid_MeV, bin_edges_MeV=bin_edges_MeV,
        exposure_s=(1.0 if exposure_days is None else exposure_days) * _SECONDS_PER_DAY,
        context=params.context, background_counts=background_counts,
    )


def cc_event_rate(
    params: SNOParameters,
    probabilities: Tensor,
    flux_tot_MeV: Tensor,
    bin_edges_MeV: Tensor,
    *,
    E_nu_grid_MeV: Optional[Tensor] = None,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    c0: Optional[float] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    background_counts: Optional[Tensor] = None,
) -> Tensor:
    """Predicted SNO CC counts per bin, for one period (day or night).

    Args:
        params: Loaded SNO parameters, supplying every grid/target/
            exposure/resolution default below.
        probabilities: nu_e survival probability on ``E_nu_grid_MeV``
            for this period (already Earth-regeneration-exposure
            -averaged for night, see
            ``tpeanuts.experiments.sno.inference_model.SNOModel``),
            shape ``(..., n_E, n_flavour>=3)``. Differentiable w.r.t.
            oscillation parameters.
        flux_tot_MeV: Total (flavour-summed) differential flux
            dPhi/dE on ``E_nu_grid_MeV``, shape ``(n_E,)``, cm^-2 s^-1
            MeV^-1.
        bin_edges_MeV: Observed-spectrum bin edges, shape
            ``(n_bins + 1,)``.
        E_nu_grid_MeV, T_grid_MeV, Tprime_grid_MeV: Override grids;
            None uses ``params.e_nu_grid_MeV``/``params.T_grid_MeV``/
            ``params.Tprime_grid_MeV``.
        n_target: Override target deuteron count; None uses
            ``params.n_target_deuterons``.
        exposure_days: Exposure in days for this period; None uses
            ``1.0``, matching v1_0's own default (a real exposure is
            always supplied explicitly by
            ``tpeanuts.experiments.sno.inference_model.SNOModel``).
        c0, c1, c2: Override real energy-resolution formula
            coefficients; None uses ``params.energy_resolution_c0/c1/c2``.
        background_counts: Real published background per bin for this
            period (see ``tpeanuts.experiments.sno.io.load_backgrounds``);
            None adds nothing.

    Returns:
        Predicted counts per bin, shape ``(n_bins,)``.

    Raises:
        ValueError: If ``probabilities`` does not have shape
            ``(..., n_E, n_flavour>=3)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")

    if E_nu_grid_MeV is None:
        E_nu_grid_MeV = params.e_nu_grid_MeV

    flux_e = flux_tot_MeV * probabilities[..., 0]
    flux_x = torch.zeros_like(flux_e)  # CC is nu_e-only: no nu_mu/nu_tau contribution.
    folded = cc_detector(
        params, bin_edges_MeV, T_grid_MeV=T_grid_MeV, Tprime_grid_MeV=Tprime_grid_MeV, n_target=n_target,
        exposure_days=exposure_days, c0=c0, c1=c1, c2=c2, background_counts=background_counts,
    )
    return folded.counts(E_nu_grid_MeV, flux_e, flux_x).counts


def es_event_rate(
    params: SNOParameters,
    probabilities: Tensor,
    flux_tot_MeV: Tensor,
    bin_edges_MeV: Tensor,
    *,
    E_nu_grid_MeV: Optional[Tensor] = None,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    c0: Optional[float] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    background_counts: Optional[Tensor] = None,
) -> Tensor:
    """Predicted SNO ES (elastic-scattering) counts per bin, for one period.

    nu + e- -> nu + e-, sensitive to every active flavour (with
    different couplings for nu_e vs. nu_mu/nu_tau, see
    ``tpeanuts.detector.neutrino_electron``) -- the same reaction and
    fold ``tpeanuts.experiments.borexino.event_rate.event_rate`` uses,
    with SNO's own D2O electron target and Cherenkov response in place
    of Borexino's scintillator ones. A genuinely separate channel from
    ``cc_event_rate``: call both and add the results if a comparison
    against real data that itself contains both contributions is
    needed (see module docstring).

    Args:
        params, probabilities, flux_tot_MeV, bin_edges_MeV,
        E_nu_grid_MeV, T_grid_MeV, Tprime_grid_MeV, exposure_days, c0,
        c1, c2, background_counts: Same meaning as ``cc_event_rate``.
        n_target: Override target electron count; None uses
            ``params.n_target_electrons``.

    Returns:
        Predicted counts per bin, shape ``(n_bins,)``.

    Raises:
        ValueError: If ``probabilities`` does not have shape
            ``(..., n_E, n_flavour>=3)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")

    if E_nu_grid_MeV is None:
        E_nu_grid_MeV = params.e_nu_grid_MeV

    flux_e = flux_tot_MeV * probabilities[..., 0]
    flux_x = flux_tot_MeV * probabilities[..., 1:3].sum(dim=-1)
    folded = es_detector(
        params, bin_edges_MeV, T_grid_MeV=T_grid_MeV, Tprime_grid_MeV=Tprime_grid_MeV, n_target=n_target,
        exposure_days=exposure_days, c0=c0, c1=c1, c2=c2, background_counts=background_counts,
    )
    return folded.counts(E_nu_grid_MeV, flux_e, flux_x).counts


def nc_event_rate(
    params: SNOParameters,
    flux_tot_MeV: Tensor,
    bin_edges_MeV: Tensor,
    *,
    probabilities: Optional[Tensor] = None,
    E_nu_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    c0: Optional[float] = None,
    c1: Optional[float] = None,
    c2: Optional[float] = None,
    capture_energy_MeV: Optional[float] = None,
    capture_efficiency: Optional[float] = None,
    background_counts: Optional[Tensor] = None,
) -> Tensor:
    """Predicted SNO NC (neutral-current) counts per bin, for one period.

    nu_x + d -> p + n + nu_x is flavour-blind (equal cross section for
    nu_e/nu_mu/nu_tau). With three active flavours the oscillated and
    unoscillated active fluxes coincide; when ``probabilities`` is
    supplied, sterile probability is explicitly excluded. This
    integrates ``tpeanuts.detector.deuteron.nc_cross_section`` against
    the total flux directly rather than folding a differential cross
    section against a flavour-split flux the way
    ``cc_event_rate``/``es_event_rate`` do:

        ``R_nc = n_target * integral dE_nu [Phi_tot(E_nu) sigma_NC(E_nu)]``
        (interactions/s)

    Unlike CC/ES, the observable here is not the outgoing particle's
    own kinetic energy: the produced neutron thermalizes and is later
    captured on a second deuteron, releasing a single fixed-energy
    gamma (``capture_energy_MeV``) independent of the original E_nu.
    That capture rate is scaled by the published Phase-I effective
    neutron detection efficiency (``capture_efficiency``) and
    distributed into a visible-energy spectrum with the same Gaussian
    response every other channel in this project uses, reusing
    ``tpeanuts.detector.common.response.gaussian_response_matrix``
    evaluated at the single true energy ``capture_energy_MeV`` (a
    one-column response matrix) rather than ``predicted_counts``'s
    full E_nu-dependent fold.

    Args:
        params: Loaded SNO parameters.
        flux_tot_MeV: Total produced differential flux dPhi/dE on
            ``E_nu_grid_MeV``, shape ``(n_E,)``, cm^-2 s^-1 MeV^-1.
        bin_edges_MeV: Observed-spectrum bin edges, shape
            ``(n_bins + 1,)``.
        probabilities: Optional full flavour probabilities, shaped
            ``(..., n_E, n_flavour>=3)``. If present, only indices 0:3
            contribute to NC; if absent, unit active probability is
            assumed.
        E_nu_grid_MeV: Override true neutrino energy grid; None uses
            ``params.e_nu_grid_MeV``.
        Tprime_grid_MeV: Override reconstructed visible-energy grid the
            capture-gamma response is evaluated on; None uses
            ``params.Tprime_grid_MeV``.
        n_target: Override target deuteron count; None uses
            ``params.n_target_deuterons``.
        exposure_days: Exposure in days for this period; None uses
            ``1.0``.
        c0, c1, c2: Override real energy-resolution formula
            coefficients; None uses ``params.energy_resolution_c0/c1/c2``.
        capture_energy_MeV: Override neutron-capture gamma energy;
            None uses ``params.nc_capture_energy_MeV``.
        capture_efficiency: Override Phase-I effective neutron
            detection efficiency; None uses
            ``params.nc_capture_efficiency``.
        background_counts: Optional background per bin; None adds
            nothing. Not the real published "neutron" background
            column (see module docstring) -- that is a CC/ES-window
            leakage correction, not a background to this channel's own
            dedicated peak.

    Returns:
        Predicted counts per bin, shape ``(n_bins,)``.

    Raises:
        ValueError: If ``probabilities`` is given and does not have
            shape ``(..., n_E, n_flavour>=3)``.
    """
    if E_nu_grid_MeV is None:
        E_nu_grid_MeV = params.e_nu_grid_MeV

    active_flux = flux_tot_MeV
    if probabilities is not None:
        if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
            raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
        active_flux = flux_tot_MeV * probabilities[..., :3].sum(dim=-1)
    folded = nc_detector(
        params, bin_edges_MeV, Tprime_grid_MeV=Tprime_grid_MeV, n_target=n_target, exposure_days=exposure_days,
        c0=c0, c1=c1, c2=c2, capture_energy_MeV=capture_energy_MeV, capture_efficiency=capture_efficiency,
        background_counts=background_counts,
    )
    return folded.counts(E_nu_grid_MeV, active_flux).counts
