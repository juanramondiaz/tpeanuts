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
Composed SNO event-rate functions: three separate, independently composable channels.

SNO measures solar neutrinos through three physically distinct reactions on
the same D2O target, and this module keeps them as three separate functions
(never auto-summed here) rather than one "total predicted spectrum", so a
caller can combine whichever subset a given analysis needs -- e.g. all
three summed against the real day/night candidate spectrum
(``tpeanuts.detector.sno.inference_model.SNODayNightModel``, since SNO's
own raw counts are not separated by reaction either), or ``cc_event_rate``
alone for a CC-only study:

    cc_event_rate(...)
        nu_e + d -> p + p + e-  (CC, nu_e-only). Wires
        ``detector.common.event_rate.predicted_counts`` with the real
        Nakamura et al. (2002) cross section
        (``detector.interaction.deuteron``).
    es_event_rate(...)
        nu + e- -> nu + e-  (ES, all flavours, different couplings). Same
        fold as ``detector.borexino.event_rate``, with SNO's own target/
        response/exposure -- added as a genuinely separate channel, not
        merged into ``cc_event_rate``'s prediction.
    nc_event_rate(...)
        nu_x + d -> p + n + nu_x  (NC, flavour-blind), followed by neutron
        capture on a second deuteron and its gamma's Compton-scattered
        electron -- see that function's own docstring for why this channel
        cannot reuse ``predicted_counts``'s continuum fold the way the
        other two do.

All three deliberately do not know about oscillation, the Sun, or the
Earth: they take flavour probabilities/``flux_tot_MeV`` as plain tensors, built upstream
(day or night, see ``tpeanuts.detector.sno.inference_model.SNODayNightModel``).

**Real neutron backgrounds stay backgrounds, not the NC signal above.** The
"neutron" column of the real published
``data/detector/sno/observation/backgrounds.csv``, loaded by
``detector.sno.backgrounds``/``detector.sno.io.load_backgrounds``, is
SNO's own measured NC-neutron-capture *leakage into the CC/ES electron-
energy analysis window* -- a background correction to the CC(+ES) spectrum
fit, unrelated to ``nc_event_rate`` above (a forward-modeled prediction for
NC's own dedicated visible-energy peak). Neither function reads from or
feeds into the other.

Module contents:
    cc_event_rate(...), es_event_rate(...), nc_event_rate(...)
        See above.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.event_rate import bin_counts, predicted_counts
from tpeanuts.detector.common.response import gaussian_response_matrix
from tpeanuts.detector.interaction.deuteron import cc_cross_section_grid, sigma_nc_total
from tpeanuts.detector.interaction.neutrino_electron import (
    nue_cross_section_grid,
    numutau_cross_section_grid,
)
from tpeanuts.detector.sno.parameters import (
    E_NU_GRID_MEV,
    ENERGY_RESOLUTION_C0,
    ENERGY_RESOLUTION_C1,
    ENERGY_RESOLUTION_C2,
    NC_CAPTURE_EFFICIENCY,
    NC_CAPTURE_ENERGY_MEV,
    N_TARGET_DEUTERONS,
    N_TARGET_ELECTRONS,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)
from tpeanuts.detector.sno.response import response_matrix, sigma_MeV as cc_sigma_MeV

_SECONDS_PER_DAY = 86_400.0


def cc_event_rate(
    probabilities: torch.Tensor,
    flux_tot_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    n_target: torch.Tensor = N_TARGET_DEUTERONS,
    exposure_days: float = 1.0,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predicted SNO CC counts per bin, for one period (day or night).

    Args:
        p_ee: nu_e survival probability on ``E_nu_grid_MeV`` for this
            period (already Earth-regeneration-averaged for night, see
            ``tpeanuts.detector.sno.inference_model.SNODayNightModel``), shape
            ``(n_E,)``. Differentiable w.r.t. oscillation parameters.
        flux_tot_MeV: Total (flavour-summed) differential flux dPhi/dE on
            ``E_nu_grid_MeV``, shape ``(n_E,)``, cm^-2 s^-1 MeV^-1.
        bin_edges_MeV: Observed-spectrum bin edges, shape ``(n_bins + 1,)``.
        E_nu_grid_MeV: True neutrino energy grid.
        T_grid_MeV: True electron-energy grid.
        Tprime_grid_MeV: Reconstructed electron-energy grid; must equal
            ``T_grid_MeV`` (see ``detector.common.event_rate
            .predicted_counts``).
        n_target: Target deuteron count (default: 1000 t D2O, see
            ``detector.sno.parameters``).
        exposure_days: Exposure in days for this period.
        c0, c1, c2: Real energy-resolution formula coefficients, see
            ``detector.sno.response``.
        background_counts: Real published background per bin for this
            period (see ``detector.sno.backgrounds``); None adds nothing.

    Returns:
        Predicted counts per bin, shape ``(n_bins,)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
    device, dtype = probabilities.device, probabilities.dtype
    E_nu_grid_MeV = E_nu_grid_MeV.to(device=device, dtype=dtype)
    T_grid_MeV = T_grid_MeV.to(device=device, dtype=dtype)
    Tprime_grid_MeV = Tprime_grid_MeV.to(device=device, dtype=dtype)
    bin_edges_MeV = bin_edges_MeV.to(device=device, dtype=dtype)
    flux_tot_MeV = flux_tot_MeV.to(device=device, dtype=dtype)
    flux_e = flux_tot_MeV * probabilities[..., 0]
    flux_x = torch.zeros_like(flux_e)  # CC is nu_e-only: no nu_mu/nu_tau contribution.

    cross_section_e = cc_cross_section_grid(E_nu_grid_MeV, T_grid_MeV)
    cross_section_x = torch.zeros_like(cross_section_e)

    R = response_matrix(T_grid_MeV, Tprime_grid_MeV, c0=c0, c1=c1, c2=c2)
    efficiency = torch.ones_like(Tprime_grid_MeV)

    return predicted_counts(
        E_nu_grid_MeV, flux_e, flux_x,
        cross_section_e, cross_section_x, n_target,
        T_grid_MeV, R, efficiency,
        bin_edges_MeV, exposure_days * _SECONDS_PER_DAY,
        background_counts=background_counts,
    )


def es_event_rate(
    probabilities: torch.Tensor,
    flux_tot_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    n_target: torch.Tensor = N_TARGET_ELECTRONS,
    exposure_days: float = 1.0,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predicted SNO ES (elastic-scattering) counts per bin, for one period.

    nu + e- -> nu + e-, sensitive to every active flavour (with different
    couplings for nu_e vs. nu_mu/nu_tau, see
    ``detector.interaction.neutrino_electron``) -- the same reaction and
    fold ``detector.borexino.event_rate`` uses, with SNO's own D2O electron
    target and Cherenkov response in place of Borexino's scintillator ones.
    A genuinely separate channel from ``cc_event_rate``: call both and add
    the results if a comparison against real data that itself contains
    both contributions is needed (see module docstring).

    Args:
        p_ee: nu_e survival probability on ``E_nu_grid_MeV`` for this
            period, shape ``(n_E,)``. Differentiable w.r.t. oscillation
            parameters.
        flux_tot_MeV: Total (flavour-summed) differential flux dPhi/dE on
            ``E_nu_grid_MeV``, shape ``(n_E,)``, cm^-2 s^-1 MeV^-1.
        bin_edges_MeV: Observed-spectrum bin edges, shape ``(n_bins + 1,)``.
        E_nu_grid_MeV: True neutrino energy grid.
        T_grid_MeV: True electron-recoil grid.
        Tprime_grid_MeV: Reconstructed electron-recoil grid; must equal
            ``T_grid_MeV`` (see ``detector.common.event_rate
            .predicted_counts``).
        n_target: Target electron count (default: 1000 t D2O, see
            ``detector.sno.parameters.N_TARGET_ELECTRONS``).
        exposure_days: Exposure in days for this period.
        c0, c1, c2: Real energy-resolution formula coefficients, see
            ``detector.sno.response``.
        background_counts: Optional background per bin; None adds nothing.

    Returns:
        Predicted counts per bin, shape ``(n_bins,)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
    device, dtype = probabilities.device, probabilities.dtype
    E_nu_grid_MeV = E_nu_grid_MeV.to(device=device, dtype=dtype)
    T_grid_MeV = T_grid_MeV.to(device=device, dtype=dtype)
    Tprime_grid_MeV = Tprime_grid_MeV.to(device=device, dtype=dtype)
    bin_edges_MeV = bin_edges_MeV.to(device=device, dtype=dtype)
    flux_tot_MeV = flux_tot_MeV.to(device=device, dtype=dtype)
    flux_e = flux_tot_MeV * probabilities[..., 0]
    flux_x = flux_tot_MeV * probabilities[..., 1:3].sum(dim=-1)

    cross_section_e = nue_cross_section_grid(E_nu_grid_MeV, T_grid_MeV)
    cross_section_x = numutau_cross_section_grid(E_nu_grid_MeV, T_grid_MeV)

    R = response_matrix(T_grid_MeV, Tprime_grid_MeV, c0=c0, c1=c1, c2=c2)
    efficiency = torch.ones_like(Tprime_grid_MeV)

    return predicted_counts(
        E_nu_grid_MeV, flux_e, flux_x,
        cross_section_e, cross_section_x, n_target,
        T_grid_MeV, R, efficiency,
        bin_edges_MeV, exposure_days * _SECONDS_PER_DAY,
        background_counts=background_counts,
    )


def nc_event_rate(
    flux_tot_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    *,
    probabilities: Optional[torch.Tensor] = None,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    n_target: torch.Tensor = N_TARGET_DEUTERONS,
    exposure_days: float = 1.0,
    c0: float = ENERGY_RESOLUTION_C0,
    c1: float = ENERGY_RESOLUTION_C1,
    c2: float = ENERGY_RESOLUTION_C2,
    capture_energy_MeV: float = NC_CAPTURE_ENERGY_MEV,
    capture_efficiency: float = NC_CAPTURE_EFFICIENCY,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predicted SNO NC (neutral-current) counts per bin, for one period.

    nu_x + d -> p + n + nu_x is flavour-blind (equal cross section for
    nu_e/nu_mu/nu_tau). With three active flavours the oscillated and
    unoscillated active fluxes coincide; when ``probabilities`` is supplied,
    sterile probability is explicitly excluded. This integrates
    ``detector.interaction.deuteron.sigma_nc_total`` against the total flux
    directly rather than folding a differential cross section against a
    flavour-split flux the way ``cc_event_rate``/``es_event_rate`` do:

        R_nc = n_target * integral dE_nu [Phi_tot(E_nu) sigma_NC(E_nu)]     (interactions/s)

    Unlike CC/ES, the observable here is not the outgoing particle's own
    kinetic energy: the produced neutron thermalizes and is later captured
    on a second deuteron, releasing a single fixed-energy gamma
    (``NC_CAPTURE_ENERGY_MEV``) independent of the original E_nu. That
    capture rate is scaled by the published Phase-I effective neutron
    detection efficiency (``NC_CAPTURE_EFFICIENCY``) and
    distributed into a visible-energy spectrum with the same Gaussian
    response every other channel in this project uses, reusing
    ``detector.common.response.gaussian_response_matrix`` evaluated at the
    single true energy ``capture_energy_MeV`` (a one-column response
    matrix) rather than ``predicted_counts``'s full E_nu-dependent fold.

    Args:
        flux_tot_MeV: Total produced differential flux dPhi/dE on
            ``E_nu_grid_MeV``, shape ``(n_E,)``, cm^-2 s^-1 MeV^-1.
        bin_edges_MeV: Observed-spectrum bin edges, shape ``(n_bins + 1,)``.
        probabilities: Optional full flavour probabilities, shaped
            ``(..., n_E, n_flavour>=3)``. If present, only indices 0:3
            contribute to NC; if absent, unit active probability is assumed.
        E_nu_grid_MeV: True neutrino energy grid.
        Tprime_grid_MeV: Reconstructed visible-energy grid the capture-gamma
            response is evaluated on.
        n_target: Target deuteron count (default: 1000 t D2O, see
            ``detector.sno.parameters``).
        exposure_days: Exposure in days for this period.
        c0, c1, c2: Real energy-resolution formula coefficients, see
            ``detector.sno.response``.
        capture_energy_MeV: Neutron-capture gamma energy, see
            ``detector.sno.parameters.NC_CAPTURE_ENERGY_MEV``.
        capture_efficiency: Phase-I effective neutron detection efficiency,
            see ``detector.sno.parameters.NC_CAPTURE_EFFICIENCY``.
        background_counts: Optional background per bin; None adds nothing.
            Not the real published "neutron" background column (see module
            docstring) -- that is a CC/ES-window leakage correction, not a
            background to this channel's own dedicated peak.

    Returns:
        Predicted counts per bin, shape ``(n_bins,)``.
    """
    device, dtype = flux_tot_MeV.device, flux_tot_MeV.dtype
    E_nu_grid_MeV = E_nu_grid_MeV.to(device=device, dtype=dtype)
    Tprime_grid_MeV = Tprime_grid_MeV.to(device=device, dtype=dtype)
    bin_edges_MeV = bin_edges_MeV.to(device=device, dtype=dtype)
    sigma_nc = sigma_nc_total(E_nu_grid_MeV)
    active_flux = flux_tot_MeV
    if probabilities is not None:
        probabilities = probabilities.to(device=device, dtype=dtype)
        if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
            raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
        active_flux = flux_tot_MeV * probabilities[..., :3].sum(dim=-1)
    interaction_rate_per_s = n_target * torch.trapezoid(active_flux * sigma_nc, x=E_nu_grid_MeV)
    detected_rate_per_s = interaction_rate_per_s * capture_efficiency

    capture_T = torch.as_tensor(
        [capture_energy_MeV], dtype=Tprime_grid_MeV.dtype, device=Tprime_grid_MeV.device,
    )
    sigma_capture = cc_sigma_MeV(capture_T, c0=c0, c1=c1, c2=c2)
    response_column = gaussian_response_matrix(capture_T, Tprime_grid_MeV, sigma_capture)[:, 0]

    reco_spectrum = detected_rate_per_s * response_column  # (n_Tp,), events / s / MeV
    counts = bin_counts(reco_spectrum, Tprime_grid_MeV, bin_edges_MeV, exposure_days * _SECONDS_PER_DAY)

    if background_counts is not None:
        counts = counts + background_counts
    return counts
