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

"""Composed Borexino event-rate function: oscillated flux -> predicted counts.

Ported from v1_0's ``experiments/borexino/event_rate.py``. Wires
``tpeanuts.detector.common.event_rate.predicted_counts``/
``predicted_counts_discrete`` with Borexino's own target/response/
efficiency, following ``detector.common.event_rate``'s "compose, do
not reimplement" principle. Deliberately does not know about
oscillation or the Sun: it takes the full flavour-probability vector
and ``flux_tot_MeV``/line data as plain tensors, built upstream (see
``tpeanuts.experiments.borexino.inference_model.BorexinoModel``
for the assembly).

Unlike v1_0's own version (module-level constant defaults for
``E_nu_grid_MeV``/``T_grid_MeV``/``n_target``/``exposure_days``/
``resolution_a``, each a bare ``torch.float64`` value fixed at import
time -- incompatible with this project's ``RuntimeContext``
convention), every grid/target/exposure/resolution default here comes
from an explicit ``BorexinoParameters`` instance instead (same
params-object pattern as
``tpeanuts.experiments.dayabay.event_rate.ibd_event_rate``).

v1_0 always adds a zero background via
``experiments.borexino.backgrounds.backgrounds_MeV``; that module is not
ported here -- ``background_counts=None`` already means "no
background" without a placeholder tensor (the same choice round 1
made for the IBD detector).

Module contents:
    es_detector(...)
        The ``FoldedDetector`` of Borexino: electron target, nu-e elastic scattering, Gaussian response, optional
        threshold. ``es_detector(...).observe(flux)`` is the chain's last link, for a ``Flux`` per unit area.
    event_rate(...)
        probabilities, flux_tot_MeV -> predicted counts per bin
        (continuous spectrum).
    line_event_rate(...)
        probabilities, discrete line flux -> predicted counts per bin
        (monoenergetic solar lines).
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.efficiency import step_efficiency
from tpeanuts.detector.common.response import DetectorResponse
from tpeanuts.detector.folded import FoldedDetector
from tpeanuts.detector.neutrino_electron import nue_cross_section_grid, numutau_cross_section_grid
from tpeanuts.experiments.borexino.parameters import BorexinoParameters
from tpeanuts.experiments.borexino.response import response_matrix
from tpeanuts.util.tensor import TensorLike

Tensor = torch.Tensor

_SECONDS_PER_DAY = 86_400.0


def es_detector(
    params: BorexinoParameters,
    bin_edges_MeV: Tensor,
    *,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    resolution_a: Optional[TensorLike] = None,
    threshold_MeV: Optional[TensorLike] = None,
    background_counts: Optional[Tensor] = None,
) -> FoldedDetector:
    """Borexino as a ``Detector``: ``es_detector(params, bin_edges_MeV).observe(flux)`` gives the counts per bin.

    The arguments are those of ``event_rate`` that describe the detector (not the flux); None takes the default from
    ``params``. The grids must coincide (``Tprime_grid_MeV`` is ``T_grid_MeV``).

    Returns:
        FoldedDetector: electron target, nu-e elastic scattering, Gaussian response, optional hard threshold.
    """
    if T_grid_MeV is None:
        T_grid_MeV = params.T_grid_MeV
    if Tprime_grid_MeV is None:
        Tprime_grid_MeV = params.Tprime_grid_MeV
    if n_target is None:
        n_target = params.n_target_electrons
    if exposure_days is None:
        exposure_days = params.exposure_days
    if resolution_a is None:
        resolution_a = params.energy_resolution_a

    efficiency = (
        torch.ones_like(Tprime_grid_MeV)
        if threshold_MeV is None else step_efficiency(Tprime_grid_MeV, threshold_MeV)
    )
    response = DetectorResponse(
        T_grid_MeV=T_grid_MeV,
        response_matrix=response_matrix(T_grid_MeV, Tprime_grid_MeV, resolution_a=resolution_a),
        efficiency=efficiency,
        bin_edges_MeV=bin_edges_MeV,
    )
    return FoldedDetector(
        n_target=n_target, response=response, exposure_s=exposure_days * _SECONDS_PER_DAY,
        cross_section_e=nue_cross_section_grid, cross_section_x=numutau_cross_section_grid,
        context=params.context, background_counts=background_counts,
    )


def event_rate(
    params: BorexinoParameters,
    probabilities: Tensor,
    flux_tot_MeV: Tensor,
    bin_edges_MeV: Tensor,
    *,
    E_nu_grid_MeV: Optional[Tensor] = None,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    resolution_a: Optional[TensorLike] = None,
    threshold_MeV: Optional[TensorLike] = None,
    background_counts: Optional[Tensor] = None,
) -> Tensor:
    """Predicted Borexino counts from flavour probabilities and an incident flux.

    Args:
        params: Loaded Borexino parameters, supplying every grid/
            target/exposure/resolution default below.
        probabilities: Flavour probabilities on ``E_nu_grid_MeV``,
            shape ``(..., n_E, n_flavour)``. Index 0 is electron
            flavour and indices 1/2 are the active muon/tau flavours;
            any additional sterile components do not interact in this
            fold.
        flux_tot_MeV: Total (flavour-summed) differential flux
            dPhi/dE on ``E_nu_grid_MeV``, shape ``(n_E,)``, physical
            units cm^-2 s^-1 MeV^-1 (e.g.
            ``source.total_flux(source_name) * source.spectrum(
            source_name, E_nu_grid_MeV)``).
        bin_edges_MeV: Observed-spectrum bin edges, shape
            ``(n_bins + 1,)``.
        E_nu_grid_MeV, T_grid_MeV, Tprime_grid_MeV: Override grids;
            None uses ``params.e_nu_grid_MeV``/``params.T_grid_MeV``/
            ``params.Tprime_grid_MeV``.
        n_target: Override target electron count; None uses
            ``params.n_target_electrons``.
        exposure_days: Override exposure in days; None uses
            ``params.exposure_days``.
        resolution_a: Override resolution coefficient; None uses
            ``params.energy_resolution_a``.
        threshold_MeV: Optional hard analysis energy threshold; None
            applies no threshold (efficiency = 1 everywhere).
        background_counts: Optional background per bin; None applies
            no background.

    Returns:
        Predicted counts per bin, shape ``(..., n_bins)``, in
        events/``exposure_days`` days.

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
        exposure_days=exposure_days, resolution_a=resolution_a, threshold_MeV=threshold_MeV,
        background_counts=background_counts,
    )
    return folded.counts(E_nu_grid_MeV, flux_e, flux_x).counts


def line_event_rate(
    params: BorexinoParameters,
    probabilities: Tensor,
    total_flux: Tensor,
    line_weights: Tensor,
    line_energy_MeV: Tensor,
    bin_edges_MeV: Tensor,
    *,
    T_grid_MeV: Optional[Tensor] = None,
    Tprime_grid_MeV: Optional[Tensor] = None,
    n_target: Optional[Tensor] = None,
    exposure_days: Optional[float] = None,
    resolution_a: Optional[TensorLike] = None,
    threshold_MeV: Optional[TensorLike] = None,
    background_counts: Optional[Tensor] = None,
) -> Tensor:
    """Fold monoenergetic solar-neutrino lines without trapezoidal E integration.

    Args:
        params, bin_edges_MeV, T_grid_MeV, Tprime_grid_MeV, n_target,
        exposure_days, resolution_a, threshold_MeV, background_counts:
            Same meaning as ``event_rate``.
        probabilities: Flavour probabilities on ``line_energy_MeV``,
            shape ``(..., n_lines, n_flavour>=3)``.
        total_flux: Total integrated flux for the line source (e.g.
            ``source.total_flux(source_name)``), scalar.
        line_weights: Normalized per-line weights, shape ``(n_lines,)``
            (see ``tpeanuts.source.solar.solar.SolarLineSpectrum``).
        line_energy_MeV: Line energies, shape ``(n_lines,)``.

    Returns:
        Predicted counts per bin, shape ``(..., n_bins)``.

    Raises:
        ValueError: If ``probabilities`` does not have shape
            ``(..., n_lines, n_flavour>=3)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_lines, n_flavour>=3)")

    flux_lines = total_flux * line_weights
    flux_e = flux_lines * probabilities[..., 0]
    flux_x = flux_lines * probabilities[..., 1:3].sum(dim=-1)
    folded = es_detector(
        params, bin_edges_MeV, T_grid_MeV=T_grid_MeV, Tprime_grid_MeV=Tprime_grid_MeV, n_target=n_target,
        exposure_days=exposure_days, resolution_a=resolution_a, threshold_MeV=threshold_MeV,
        background_counts=background_counts,
    )
    return folded.counts(line_energy_MeV, flux_e, flux_x, discrete=True).counts
