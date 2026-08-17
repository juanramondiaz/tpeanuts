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
Composed Borexino event-rate function: oscillated flux -> predicted counts.

Wires ``detector.common.event_rate.predicted_counts`` with Borexino's own
target/response/efficiency/background, following
``detector.common.event_rate``'s "compose, do not reimplement" principle.
Deliberately does not know about oscillation or the Sun: it takes the full
flavour-probability vector and ``flux_tot_MeV`` as plain tensors, built upstream from
``tpeanuts.medium.solar``/``tpeanuts.source.solar``/``tpeanuts.inference``
(see ``notebooks/inference/inference1_borexino.ipynb`` for how a caller
assembles them via ``SolarNeutrinoSource.flux``/``SolarNeutrinoSource
.spectrum`` and an oscillation model's ``predict_pee``-style method).

Module contents:
    event_rate(...)
        probabilities, flux_tot_MeV -> predicted counts per bin.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.borexino.backgrounds import backgrounds_MeV
from tpeanuts.detector.borexino.parameters import (
    E_NU_GRID_MEV,
    ENERGY_RESOLUTION_A,
    N_TARGET_ELECTRONS,
    REFERENCE_EXPOSURE_DAYS,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)
from tpeanuts.detector.borexino.response import response_matrix
from tpeanuts.detector.common.efficiency import step_efficiency
from tpeanuts.detector.common.event_rate import predicted_counts, predicted_counts_discrete
from tpeanuts.detector.interaction.neutrino_electron import (
    nue_cross_section_grid,
    numutau_cross_section_grid,
)

_SECONDS_PER_DAY = 86_400.0


def event_rate(
    probabilities: torch.Tensor,
    flux_tot_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    n_target: torch.Tensor = N_TARGET_ELECTRONS,
    exposure_days: float = REFERENCE_EXPOSURE_DAYS,
    resolution_a: float = ENERGY_RESOLUTION_A,
    threshold_MeV: Optional[float] = None,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Predicted Borexino counts from flavour probabilities and an incident flux.

    Args:
        probabilities: Flavour probabilities on ``E_nu_grid_MeV``, shape
            ``(..., n_E, n_flavour)``. Index 0 is electron flavour and
            indices 1 and 2 are the active muon and tau flavours. Any
            additional sterile components do not interact in this fold.
        flux_tot_MeV: Total (flavour-summed) differential flux dPhi/dE on
            ``E_nu_grid_MeV``, shape ``(n_E,)``, physical units
            cm^-2 s^-1 MeV^-1 (e.g. ``source.total_flux(source_name) *
            source.spectrum(source_name, E_nu_grid_MeV)``, see module
            docstring).
        bin_edges_MeV: Observed-spectrum bin edges, shape ``(n_bins + 1,)``.
        E_nu_grid_MeV: True neutrino energy grid matching ``p_ee``/
            ``flux_tot_MeV``.
        T_grid_MeV: True electron-recoil grid.
        Tprime_grid_MeV: Reconstructed electron-recoil grid; must currently
            equal ``T_grid_MeV`` (see ``detector.common.event_rate
            .predicted_counts``).
        n_target: Target electron count (default: 100 t of pseudocumene,
            matching the reference normalization -- see
            ``detector.borexino.parameters``).
        exposure_days: Exposure in days (default: 1 day, i.e. the returned
            counts are directly a rate, events/day, matching the reference
            normalization).
        resolution_a: Energy-resolution normalization, see
            ``detector.borexino.response``.
        threshold_MeV: Optional hard analysis energy threshold; None applies
            no threshold (efficiency = 1 everywhere).
        background_counts: Optional background per bin; None uses
            ``detector.borexino.backgrounds.backgrounds_MeV`` (currently
            always zero).

    Returns:
        Predicted counts per bin, shape ``(..., n_bins)``, in
        events/``exposure_days`` days.
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
    R = response_matrix(T_grid_MeV, Tprime_grid_MeV, resolution_a=resolution_a)

    if threshold_MeV is None:
        efficiency = torch.ones_like(Tprime_grid_MeV)
    else:
        efficiency = step_efficiency(Tprime_grid_MeV, threshold_MeV)

    n_bins = bin_edges_MeV.shape[0] - 1
    background = (
        backgrounds_MeV(n_bins, device=flux_tot_MeV.device, dtype=flux_tot_MeV.dtype)
        if background_counts is None else background_counts
    )

    return predicted_counts(
        E_nu_grid_MeV, flux_e, flux_x,
        cross_section_e, cross_section_x, n_target,
        T_grid_MeV, R, efficiency,
        bin_edges_MeV, exposure_days * _SECONDS_PER_DAY,
        background_counts=background,
    )


def line_event_rate(
    probabilities: torch.Tensor,
    total_flux: torch.Tensor,
    line_weights: torch.Tensor,
    line_energy_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    *,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    n_target: torch.Tensor = N_TARGET_ELECTRONS,
    exposure_days: float = REFERENCE_EXPOSURE_DAYS,
    resolution_a: float = ENERGY_RESOLUTION_A,
    threshold_MeV: Optional[float] = None,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fold monoenergetic solar-neutrino lines without trapezoidal E integration."""
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_lines, n_flavour>=3)")
    device, dtype = probabilities.device, probabilities.dtype
    energy = line_energy_MeV.to(device=device, dtype=dtype)
    weights = line_weights.to(device=device, dtype=dtype)
    total_flux = total_flux.to(device=device, dtype=dtype)
    T_grid_MeV = T_grid_MeV.to(device=device, dtype=dtype)
    Tprime_grid_MeV = Tprime_grid_MeV.to(device=device, dtype=dtype)
    bin_edges_MeV = bin_edges_MeV.to(device=device, dtype=dtype)
    flux_lines = total_flux * weights
    flux_e = flux_lines * probabilities[..., 0]
    flux_x = flux_lines * probabilities[..., 1:3].sum(dim=-1)
    cross_section_e = nue_cross_section_grid(energy, T_grid_MeV)
    cross_section_x = numutau_cross_section_grid(energy, T_grid_MeV)
    R = response_matrix(T_grid_MeV, Tprime_grid_MeV, resolution_a=resolution_a)
    efficiency = (
        torch.ones_like(Tprime_grid_MeV)
        if threshold_MeV is None else step_efficiency(Tprime_grid_MeV, threshold_MeV)
    )
    n_bins = bin_edges_MeV.shape[0] - 1
    background = (
        backgrounds_MeV(n_bins, device=device, dtype=dtype)
        if background_counts is None else background_counts.to(device=device, dtype=dtype)
    )
    return predicted_counts_discrete(
        flux_e, flux_x, cross_section_e, cross_section_x, n_target,
        T_grid_MeV, R, efficiency, bin_edges_MeV,
        exposure_days * _SECONDS_PER_DAY, background_counts=background,
    )
