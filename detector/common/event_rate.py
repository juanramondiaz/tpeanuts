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
The single event-rate folding assembly: flux -> predicted per-bin counts.

This module contains the *only* event-rate folding code in the project --
mirroring ``tpeanuts.core.common.hamiltonian``'s "only Hamiltonian assembly
code" principle -- so every detector's ``event_rate.py`` composes these same
four steps rather than reimplementing the folding formula:

    dR/dT(T)   = N_target * integral dE_nu [Phi_e(E_nu) dsigma_e/dT(E_nu,T)
                                           + Phi_x(E_nu) dsigma_x/dT(E_nu,T)]
    dR/dT'(T') = integral dT R(T'|T) dR/dT(T)                    (response)
    dR_det/dT' = eps(T') * dR/dT'(T')                             (efficiency)
    N_i        = exposure * integral_{bin i} dT' dR_det/dT'(T')  (+ N_i^bkg)

``true_observable_spectrum`` takes ``flux_e``/``flux_x`` rather than a
flavour-resolved flux tensor directly: only the nu_e vs. (nu_mu + nu_tau)
split matters for a cross section that treats mu/tau identically (true for
every interaction currently in ``tpeanuts.detector.interaction``), so a
caller with a full ``(..., n_flavours)`` flux from
``tpeanuts.core.common.flux.flux_state`` splits it once
(``flux_e = flux[..., 0]``, ``flux_x = flux[..., 1] + flux[..., 2]``) before
calling in.

Every intermediate tensor here carries the gradient from whichever upstream
oscillation parameters produced ``flux_e``/``flux_x``; every other argument
(cross sections, response matrix, efficiency, target, exposure) is a
grid-only constant multiplied in, so autograd differentiates this exactly
like any other product of a differentiable factor and detached constants.

Module contents:
    true_observable_spectrum(...)
        Fold flux and cross sections, integrate over neutrino energy.
    apply_response(...)
        Convolve with a response/migration matrix.
    bin_counts(...)
        Integrate a reconstructed-energy spectrum into per-bin counts.
    predicted_counts(...)
        Top-level orchestration: the four steps above, plus
        ``tpeanuts.detector.common.efficiency.apply_efficiency`` and an
        optional background.
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.efficiency import apply_efficiency
from tpeanuts.util.type import TensorLike, as_tensor


def true_observable_spectrum(
    E_nu_grid_MeV: torch.Tensor,
    flux_e: torch.Tensor,
    flux_x: torch.Tensor,
    cross_section_e: torch.Tensor,
    cross_section_x: torch.Tensor,
    n_target: TensorLike,
) -> torch.Tensor:
    """Fold an incident flux with interaction cross sections, integrating over E_nu.

    dR/dT(T) = N_target * integral dE_nu [Phi_e(E_nu) dsigma_e/dT(E_nu,T)
                                         + Phi_x(E_nu) dsigma_x/dT(E_nu,T)]

    Args:
        E_nu_grid_MeV: True neutrino energy grid, shape ``(n_E,)``.
        flux_e: Differential nu_e flux dPhi_e/dE, shape ``(..., n_E)``.
        flux_x: Differential (nu_mu + nu_tau) flux dPhi_x/dE, broadcastable
            with ``flux_e``.
        cross_section_e: dsigma_e/dT for nu_e, pre-evaluated on
            ``(E_nu_grid_MeV, T_grid_MeV)``, shape ``(n_E, n_T)`` (see
            ``tpeanuts.detector.interaction.neutrino_electron``).
        cross_section_x: dsigma_x/dT for nu_mu/nu_tau, same shape.
        n_target: Scalar number of target particles (electrons or nucleons,
            depending on the interaction), e.g. from
            ``tpeanuts.detector.common.target.n_electrons``.

    Returns:
        dR/dT, shape ``(..., n_T)`` (``flux_e``'s leading batch shape, cross
        sections' ``n_T``).

    Raises:
        ValueError: If ``cross_section_e``/``cross_section_x`` do not share
            ``cross_section_e.shape``, or their leading dimension does not
            match ``E_nu_grid_MeV``.
    """
    if cross_section_e.shape != cross_section_x.shape:
        raise ValueError(
            "cross_section_e and cross_section_x must share the same shape, "
            f"got {tuple(cross_section_e.shape)} and {tuple(cross_section_x.shape)}."
        )
    if cross_section_e.shape[0] != E_nu_grid_MeV.shape[0]:
        raise ValueError(
            "cross_section_e/cross_section_x's leading dimension "
            f"({cross_section_e.shape[0]}) must match E_nu_grid_MeV "
            f"({E_nu_grid_MeV.shape[0]})."
        )

    n_target_t = as_tensor(n_target, device=flux_e.device, dtype=flux_e.dtype)

    # (..., n_E, 1) * (n_E, n_T) -> (..., n_E, n_T), summed over the two
    # flavour contributions before integrating out n_E.
    integrand = (
        flux_e[..., :, None] * cross_section_e
        + flux_x[..., :, None] * cross_section_x
    )
    return n_target_t * torch.trapezoid(integrand, x=E_nu_grid_MeV, dim=-2)


def true_observable_spectrum_discrete(
    flux_e: torch.Tensor,
    flux_x: torch.Tensor,
    cross_section_e: torch.Tensor,
    cross_section_x: torch.Tensor,
    n_target: TensorLike,
) -> torch.Tensor:
    """Fold discrete neutrino lines without applying an energy quadrature."""
    if cross_section_e.shape != cross_section_x.shape:
        raise ValueError("cross_section_e and cross_section_x must share the same shape.")
    if flux_e.shape[-1] != cross_section_e.shape[0]:
        raise ValueError("The number of line fluxes must match the cross-section energy axis.")
    n_target_t = as_tensor(n_target, device=flux_e.device, dtype=flux_e.dtype)
    return n_target_t * (
        flux_e[..., :, None] * cross_section_e
        + flux_x[..., :, None] * cross_section_x
    ).sum(dim=-2)


def apply_response(
    true_spectrum: torch.Tensor,
    T_grid_MeV: torch.Tensor,
    response_matrix: torch.Tensor,
) -> torch.Tensor:
    """Convolve a true-observable spectrum with a response/migration matrix.

    dR/dT'(T') = integral dT R(T'|T) dR/dT(T)

    Args:
        true_spectrum: dR/dT, shape ``(..., n_T)``.
        T_grid_MeV: True-observable grid, shape ``(n_T,)``, matching
            ``true_spectrum``'s final axis.
        response_matrix: R(T'|T), shape ``(n_Tp, n_T)`` (see
            ``tpeanuts.detector.common.response.gaussian_response_matrix``).

    Returns:
        dR/dT', shape ``(..., n_Tp)``.

    Raises:
        ValueError: If ``response_matrix``'s second dimension does not
            match ``T_grid_MeV``/``true_spectrum``'s final axis.
    """
    if response_matrix.shape[1] != T_grid_MeV.shape[0]:
        raise ValueError(
            f"response_matrix's second dimension ({response_matrix.shape[1]}) "
            f"must match T_grid_MeV ({T_grid_MeV.shape[0]})."
        )
    if true_spectrum.shape[-1] != T_grid_MeV.shape[0]:
        raise ValueError(
            f"true_spectrum's final dimension ({true_spectrum.shape[-1]}) "
            f"must match T_grid_MeV ({T_grid_MeV.shape[0]})."
        )

    integrand = true_spectrum[..., None, :] * response_matrix  # (..., n_Tp, n_T)
    return torch.trapezoid(integrand, x=T_grid_MeV, dim=-1)


def bin_counts(
    reco_spectrum: torch.Tensor,
    Tprime_grid_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    exposure: TensorLike,
) -> torch.Tensor:
    """Integrate a reconstructed-energy spectrum into per-bin counts.

    N_i = exposure * integral_{bin i} dT' dR_det/dT'(T'), each bin integral
    evaluated by trapezoidal quadrature after linearly interpolating the
    spectrum at both exact bin edges. The result is therefore independent
    of whether those edges coincide with grid samples.

    Args:
        reco_spectrum: dR_det/dT' (after response and efficiency), shape
            ``(..., n_Tp)``.
        Tprime_grid_MeV: Reconstructed-observable grid, shape ``(n_Tp,)``,
            strictly increasing.
        bin_edges_MeV: Observed-spectrum bin edges, shape ``(n_bins + 1,)``,
            strictly increasing, spanning a sub-range of ``Tprime_grid_MeV``.
        exposure: Scalar exposure (e.g. live time, or live time already
            combined with a per-100-tons normalization to match a published
            rate table's own units).

    Returns:
        Predicted counts per bin, shape ``(..., n_bins)``.

    Raises:
        ValueError: If ``bin_edges_MeV`` is not strictly increasing, or
            lies outside ``Tprime_grid_MeV``'s range.
    """
    if Tprime_grid_MeV.ndim != 1 or Tprime_grid_MeV.numel() < 2:
        raise ValueError("Tprime_grid_MeV must be one-dimensional with at least two points.")
    if torch.any(torch.diff(Tprime_grid_MeV) <= 0):
        raise ValueError("Tprime_grid_MeV must be strictly increasing.")
    if torch.any(torch.diff(bin_edges_MeV) <= 0):
        raise ValueError("bin_edges_MeV must be strictly increasing.")
    if bin_edges_MeV[0] < Tprime_grid_MeV[0] or bin_edges_MeV[-1] > Tprime_grid_MeV[-1]:
        raise ValueError(
            "bin_edges_MeV must lie within Tprime_grid_MeV's range "
            f"[{float(Tprime_grid_MeV[0])}, {float(Tprime_grid_MeV[-1])}]; got "
            f"[{float(bin_edges_MeV[0])}, {float(bin_edges_MeV[-1])}]."
        )

    exposure_t = as_tensor(exposure, device=reco_spectrum.device, dtype=reco_spectrum.dtype)
    n_bins = bin_edges_MeV.shape[0] - 1
    counts = []
    for i in range(n_bins):
        lo, hi = bin_edges_MeV[i], bin_edges_MeV[i + 1]
        mask = (Tprime_grid_MeV > lo) & (Tprime_grid_MeV < hi)

        edge_values = []
        for edge in (lo, hi):
            upper = torch.searchsorted(Tprime_grid_MeV, edge).clamp(
                min=1, max=Tprime_grid_MeV.numel() - 1,
            )
            lower = upper - 1
            x0, x1 = Tprime_grid_MeV[lower], Tprime_grid_MeV[upper]
            y0, y1 = reco_spectrum[..., lower], reco_spectrum[..., upper]
            weight = (edge - x0) / (x1 - x0)
            edge_values.append(y0 + weight * (y1 - y0))

        T_sub = torch.cat((lo.reshape(1), Tprime_grid_MeV[mask], hi.reshape(1)))
        spectrum_sub = torch.cat(
            (edge_values[0][..., None], reco_spectrum[..., mask], edge_values[1][..., None]),
            dim=-1,
        )
        counts.append(torch.trapezoid(spectrum_sub, x=T_sub, dim=-1))

    return exposure_t * torch.stack(counts, dim=-1)


def predicted_counts(
    E_nu_grid_MeV: torch.Tensor,
    flux_e: torch.Tensor,
    flux_x: torch.Tensor,
    cross_section_e: torch.Tensor,
    cross_section_x: torch.Tensor,
    n_target: TensorLike,
    T_grid_MeV: torch.Tensor,
    response_matrix: torch.Tensor,
    efficiency: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    exposure: TensorLike,
    *,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full forward fold: oscillated flux -> predicted per-bin event counts.

    Composes ``true_observable_spectrum`` -> ``apply_response`` ->
    ``tpeanuts.detector.common.efficiency.apply_efficiency`` -> ``bin_counts``,
    then adds ``background_counts`` if given. This is the function a
    detector's own ``event_rate.py`` calls with its target/response/
    efficiency/exposure already plugged in (see
    ``tpeanuts.detector.borexino.event_rate``).

    Args:
        E_nu_grid_MeV: True neutrino energy grid, shape ``(n_E,)``.
        flux_e: Differential nu_e flux, shape ``(..., n_E)``.
        flux_x: Differential (nu_mu + nu_tau) flux, broadcastable with
            ``flux_e``.
        cross_section_e: dsigma_e/dT, shape ``(n_E, n_T)``.
        cross_section_x: dsigma_x/dT, shape ``(n_E, n_T)``.
        n_target: Scalar target particle count.
        T_grid_MeV: True-observable grid, shape ``(n_T,)``.
        response_matrix: R(T'|T), shape ``(n_Tp, n_T)``.
        efficiency: eps(T'), shape ``(n_Tp,)``.
        bin_edges_MeV: Observed-spectrum bin edges, shape ``(n_bins + 1,)``.
        exposure: Scalar exposure.
        background_counts: Optional background per bin, shape
            ``(n_bins,)``, added to the folded signal counts (see
            ``tpeanuts.detector.common.background``).

    Returns:
        Predicted total (signal [+ background]) counts per bin, shape
        ``(..., n_bins)``.
    """
    Tprime_grid_MeV = T_grid_MeV if response_matrix.shape[0] == T_grid_MeV.shape[0] else None
    if Tprime_grid_MeV is None:
        raise ValueError(
            "predicted_counts assumes the reconstructed-observable grid "
            "equals T_grid_MeV (response_matrix.shape[0] == T_grid_MeV's "
            "length); build response_matrix on that same grid, or bin/"
            "resample it first."
        )

    true_spectrum = true_observable_spectrum(
        E_nu_grid_MeV, flux_e, flux_x, cross_section_e, cross_section_x, n_target,
    )
    reco_spectrum = apply_response(true_spectrum, T_grid_MeV, response_matrix)
    detected_spectrum = apply_efficiency(reco_spectrum, efficiency)
    counts = bin_counts(detected_spectrum, Tprime_grid_MeV, bin_edges_MeV, exposure)

    if background_counts is not None:
        counts = counts + background_counts

    return counts


def predicted_counts_discrete(
    flux_e: torch.Tensor,
    flux_x: torch.Tensor,
    cross_section_e: torch.Tensor,
    cross_section_x: torch.Tensor,
    n_target: TensorLike,
    T_grid_MeV: torch.Tensor,
    response_matrix: torch.Tensor,
    efficiency: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    exposure: TensorLike,
    *,
    background_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward-fold discrete neutrino lines into reconstructed-bin counts."""
    if response_matrix.shape[0] != T_grid_MeV.shape[0]:
        raise ValueError("The reconstructed-observable grid must equal T_grid_MeV.")
    true_spectrum = true_observable_spectrum_discrete(
        flux_e, flux_x, cross_section_e, cross_section_x, n_target,
    )
    reco_spectrum = apply_response(true_spectrum, T_grid_MeV, response_matrix)
    detected_spectrum = apply_efficiency(reco_spectrum, efficiency)
    counts = bin_counts(detected_spectrum, T_grid_MeV, bin_edges_MeV, exposure)
    return counts if background_counts is None else counts + background_counts
