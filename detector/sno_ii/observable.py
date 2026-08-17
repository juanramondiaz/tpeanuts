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
#      August 2026
# =============================================================================

"""
SNO salt-phase (Phase II) equivalent-flux observables: CC spectrum, NC, ES.

Implements Eq. A1 of the primary source (see ``tpeanuts.detector.sno_ii``'s
package docstring) exactly:

    Y_i = Phi_tot * Numerator_i(Pee) / Denominator(Pee=1)

where ``Phi_tot = integral dE_nu flux_tot_MeV(E_nu)`` and ``Numerator_i``/
``Denominator`` are the same detector fold (cross section x response,
``tpeanuts.detector.common.event_rate.predicted_counts``), evaluated with
``n_target=1``/``exposure=1`` since SNO's own "equivalent flux" convention
already divides out the detector's absolute normalization -- ``Numerator_i``
is folded with the oscillated flux and integrated over bin i's range only;
``Denominator`` is folded with the *unoscillated* flux (Pee=1, i.e. the
plain ``flux_tot_MeV`` with no survival-probability weighting) and
integrated over the full analysis range above threshold. Both share the
same (arbitrary) n_target/exposure=1 convention, which cancels exactly in
the ratio -- this is why ``predicted_counts`` can be reused unmodified with
those placeholder values instead of needing a dedicated "un-normalized
fold" helper.

A direct, useful consequence: summing ``cc_equivalent_flux_spectrum``'s
output over all bins at Pee=1 recovers ``Phi_tot`` exactly (the ratio's
numerator, summed over every bin, equals the same integral as its
denominator) -- the closure property this module's tests below verify.

Module contents:
    cc_equivalent_flux_spectrum(...)
        The 17-bin (or caller-supplied) CC observable.
    es_equivalent_flux(...)
        The single integrated ES equivalent flux (pure-nu_e reference, Eq.
        A1's structure with the ES cross sections and 1-Pee weighting for
        nu_mu/nu_tau).
    nc_equivalent_flux(...)
        The single integrated NC equivalent flux -- Eq. 10 of the primary
        source: no detector fold at all, just the active-flavour flux
        integral (SNO's own signal extraction already un-folds the
        neutron-capture response for this channel).
"""

from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.detector.common.event_rate import predicted_counts
from tpeanuts.detector.interaction.deuteron import cc_cross_section_grid
from tpeanuts.detector.interaction.neutrino_electron import (
    nue_cross_section_grid,
    numutau_cross_section_grid,
)
from tpeanuts.detector.sno_ii.parameters import (
    ANALYSIS_THRESHOLD_MEV,
    E_NU_GRID_MEV,
    T_GRID_MEV,
    TPRIME_GRID_MEV,
)
from tpeanuts.detector.sno_ii.response import response_matrix as _response_matrix

__all__ = ["cc_equivalent_flux_spectrum", "es_equivalent_flux", "nc_equivalent_flux"]


def cc_equivalent_flux_spectrum(
    probabilities: torch.Tensor,
    flux_tot_MeV: torch.Tensor,
    bin_edges_MeV: torch.Tensor,
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    threshold_MeV: float = ANALYSIS_THRESHOLD_MEV,
) -> torch.Tensor:
    """Predicted CC equivalent-flux spectrum, Eq. A1 of the primary source.

    Args:
        probabilities: Flavour probabilities on ``E_nu_grid_MeV``, shape
            ``(n_E, n_flavour>=3)``, index 0 = nu_e survival Pee.
            Differentiable w.r.t. oscillation parameters.
        flux_tot_MeV: Total (flavour-summed) differential 8B flux
            dPhi/dE, shape ``(n_E,)``, cm^-2 s^-1 MeV^-1. Its own
            integral over ``E_nu_grid_MeV`` is Eq. A1's ``Phi_tot``
            prefactor.
        bin_edges_MeV: CC spectrum bin edges, shape ``(n_bins + 1,)``
            (typically ``tpeanuts.detector.sno_ii.parameters
            .CC_BIN_EDGES_MEV``).
        E_nu_grid_MeV: True neutrino energy grid.
        T_grid_MeV: True electron kinetic-energy grid.
        Tprime_grid_MeV: Reconstructed electron kinetic-energy grid; must
            equal ``T_grid_MeV`` (see ``predicted_counts``).
        threshold_MeV: Lower edge of the denominator's integration range
            (Eq. A1: ``integral_{5.5}^infinity``).

    Returns:
        Predicted CC equivalent flux per bin, shape ``(n_bins,)``, same
        units as ``flux_tot_MeV``'s own integral (1e6 cm^-2 s^-1 when
        ``flux_tot_MeV`` is expressed that way).

    Raises:
        ValueError: If ``probabilities`` does not have shape
            ``(..., n_E, n_flavour>=3)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
    device, dtype = probabilities.device, probabilities.dtype
    E_nu_grid_MeV = E_nu_grid_MeV.to(device=device, dtype=dtype)
    T_grid_MeV = T_grid_MeV.to(device=device, dtype=dtype)
    Tprime_grid_MeV = Tprime_grid_MeV.to(device=device, dtype=dtype)
    bin_edges_MeV = bin_edges_MeV.to(device=device, dtype=dtype)
    flux_tot_MeV = flux_tot_MeV.to(device=device, dtype=dtype)

    cross_section_e = cc_cross_section_grid(E_nu_grid_MeV, T_grid_MeV)
    cross_section_x = torch.zeros_like(cross_section_e)  # CC is nu_e-only.
    R = _response_matrix(T_grid_MeV, Tprime_grid_MeV)
    efficiency = torch.ones_like(Tprime_grid_MeV)

    flux_e_oscillated = flux_tot_MeV * probabilities[..., 0]
    flux_x_zero = torch.zeros_like(flux_e_oscillated)
    numerator = predicted_counts(
        E_nu_grid_MeV, flux_e_oscillated, flux_x_zero,
        cross_section_e, cross_section_x, 1.0,
        T_grid_MeV, R, efficiency, bin_edges_MeV, 1.0,
    )

    full_range_edges = torch.stack([
        torch.as_tensor(threshold_MeV, device=device, dtype=dtype),
        Tprime_grid_MeV[-1],
    ])
    denominator = predicted_counts(
        E_nu_grid_MeV, flux_tot_MeV, flux_x_zero,
        cross_section_e, cross_section_x, 1.0,
        T_grid_MeV, R, efficiency, full_range_edges, 1.0,
    )[0]

    phi_tot = torch.trapezoid(flux_tot_MeV, x=E_nu_grid_MeV)
    return phi_tot * numerator / denominator


def es_equivalent_flux(
    probabilities: torch.Tensor,
    flux_tot_MeV: torch.Tensor,
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
    T_grid_MeV: torch.Tensor = T_GRID_MEV,
    Tprime_grid_MeV: torch.Tensor = TPRIME_GRID_MEV,
    threshold_MeV: float = ANALYSIS_THRESHOLD_MEV,
) -> torch.Tensor:
    """Predicted integrated ES equivalent flux, mirroring Eq. A1's structure.

    The numerator folds the full elastic-scattering cross section (nu_e
    with Pee weighting, nu_mu/nu_tau with 1-Pee weighting -- see
    ``tpeanuts.detector.interaction.neutrino_electron``); the denominator
    is the pure-nu_e (Pee=1) reference, matching the primary source's own
    "equivalent electron-neutrino flux" convention for this channel.

    Args:
        probabilities: Flavour probabilities on ``E_nu_grid_MeV``, shape
            ``(n_E, n_flavour>=3)``.
        flux_tot_MeV: Total differential 8B flux, shape ``(n_E,)``.
        E_nu_grid_MeV, T_grid_MeV, Tprime_grid_MeV, threshold_MeV: See
            ``cc_equivalent_flux_spectrum``.

    Returns:
        Scalar tensor (0-dimensional), same units as ``flux_tot_MeV``'s
        own integral.

    Raises:
        ValueError: If ``probabilities`` does not have shape
            ``(..., n_E, n_flavour>=3)``.
    """
    if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
        raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
    device, dtype = probabilities.device, probabilities.dtype
    E_nu_grid_MeV = E_nu_grid_MeV.to(device=device, dtype=dtype)
    T_grid_MeV = T_grid_MeV.to(device=device, dtype=dtype)
    Tprime_grid_MeV = Tprime_grid_MeV.to(device=device, dtype=dtype)
    flux_tot_MeV = flux_tot_MeV.to(device=device, dtype=dtype)

    cross_section_e = nue_cross_section_grid(E_nu_grid_MeV, T_grid_MeV)
    cross_section_x = numutau_cross_section_grid(E_nu_grid_MeV, T_grid_MeV)
    R = _response_matrix(T_grid_MeV, Tprime_grid_MeV)
    efficiency = torch.ones_like(Tprime_grid_MeV)

    full_range_edges = torch.stack([
        torch.as_tensor(threshold_MeV, device=device, dtype=dtype),
        Tprime_grid_MeV[-1],
    ])

    flux_e_oscillated = flux_tot_MeV * probabilities[..., 0]
    flux_x_oscillated = flux_tot_MeV * probabilities[..., 1:3].sum(dim=-1)
    numerator = predicted_counts(
        E_nu_grid_MeV, flux_e_oscillated, flux_x_oscillated,
        cross_section_e, cross_section_x, 1.0,
        T_grid_MeV, R, efficiency, full_range_edges, 1.0,
    )[0]

    flux_x_zero = torch.zeros_like(flux_tot_MeV)
    denominator = predicted_counts(
        E_nu_grid_MeV, flux_tot_MeV, flux_x_zero,
        cross_section_e, cross_section_x, 1.0,
        T_grid_MeV, R, efficiency, full_range_edges, 1.0,
    )[0]

    phi_tot = torch.trapezoid(flux_tot_MeV, x=E_nu_grid_MeV)
    return phi_tot * numerator / denominator


def nc_equivalent_flux(
    probabilities: Optional[torch.Tensor],
    flux_tot_MeV: torch.Tensor,
    *,
    E_nu_grid_MeV: torch.Tensor = E_NU_GRID_MEV,
) -> torch.Tensor:
    """Predicted integrated NC equivalent flux, Eq. 10 of the primary source.

    No detector fold: SNO's own signal extraction already un-folds the
    neutron-capture response for this channel, so the NC equivalent flux
    is simply the active-flavour flux integral -- in the Standard Model
    (``probabilities[..., :3]`` summing to 1) this equals ``flux_tot_MeV``'s
    own integral regardless of the oscillation parameters, exactly
    reproducing the primary source's statement that the NC flux measures
    the total active flux independent of mixing.

    Args:
        probabilities: Optional flavour probabilities, shape
            ``(n_E, n_flavour>=3)``. If given, only indices 0:3 (active
            flavours) contribute, for future sterile-neutrino extensions
            (mirroring ``tpeanuts.detector.sno.event_rate.nc_event_rate``'s
            own convention); if None, unit active probability is assumed.
        flux_tot_MeV: Total differential 8B flux, shape ``(n_E,)``.
        E_nu_grid_MeV: True neutrino energy grid.

    Returns:
        Scalar tensor (0-dimensional), same units as ``flux_tot_MeV``'s
        own integral.
    """
    device, dtype = flux_tot_MeV.device, flux_tot_MeV.dtype
    E_nu_grid_MeV = E_nu_grid_MeV.to(device=device, dtype=dtype)
    active_flux = flux_tot_MeV
    if probabilities is not None:
        probabilities = probabilities.to(device=device, dtype=dtype)
        if probabilities.ndim < 2 or probabilities.shape[-1] < 3:
            raise ValueError("probabilities must have shape (..., n_E, n_flavour>=3)")
        active_flux = flux_tot_MeV * probabilities[..., :3].sum(dim=-1)
    return torch.trapezoid(active_flux, x=E_nu_grid_MeV)
