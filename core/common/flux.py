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
Generic flux utilities built on top of probability tensors.

This module contains medium-independent helpers that convert flavour
probabilities into flavour-resolved fluxes by applying source normalizations
and optional spectral weights. It does not construct probabilities or know
about solar, Atmosphere, Earth, or vacuum propagation models.

Module functions:
    flux_state(...)
        Multiply final flavour probabilities by a flux normalization and an
        optional spectrum, preserving the final flavour dimension.
    flux_integrated(...)
        Integrate a differential (energy-resolved) flux over energy via the
        trapezoidal rule, obtaining a neutrino rate resolved by every
        remaining axis (e.g. angle) and by flavour.
    flux_integrated_angular(...)
        Integrate a differential (angle-resolved) flux over the full zenith
        range via the trapezoidal rule, weighted by the geometric
        solid-angle element sin(theta) (assuming azimuthal symmetry).
"""



from __future__ import annotations

import torch


def _lift_to_probability_shape(value: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    """Append singleton axes so value broadcasts against a probability tensor."""
    if value.ndim > probability.ndim:
        raise ValueError("value cannot have more dimensions than probability.")

    return value.reshape(*value.shape, *((1,) * (probability.ndim - value.ndim)))


def flux_state(
    probability: torch.Tensor,
    flux: torch.Tensor | float,
    spectrum: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Build a flavour-resolved flux from final flavour probabilities.

    Args:
        probability: Final flavour probabilities shaped ``(..., 3)`` or
            ``(..., 4)`` (4 for the 3+1 sterile extension).
        flux: Flux normalization broadcastable with the leading probability
            dimensions. Scalars, one value per source, or full grids are
            accepted.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.

    Returns:
        Flavour-resolved flux with the same shape as ``probability``.

    Raises:
        ValueError: If ``probability`` does not have final flavour dimension
            3 or 4, or if ``flux``/``spectrum`` have too many dimensions.
    """
    if probability.shape[-1] not in (3, 4):
        raise ValueError("probability must have final flavour dimension 3 or 4.")

    flux_t = torch.as_tensor(flux, device=probability.device, dtype=probability.dtype)
    out = probability * _lift_to_probability_shape(flux_t, probability)

    if spectrum is not None:
        spectrum_t = torch.as_tensor(
            spectrum,
            device=probability.device,
            dtype=probability.dtype,
        )
        out = out * _lift_to_probability_shape(spectrum_t, probability)

    return out


def flux_transition(
    probability_matrix: torch.Tensor,
    initial_flux: torch.Tensor,
) -> torch.Tensor:
    """Apply a full transition matrix to an incoherent flavour flux.

    Uses ``P[..., beta, alpha]`` and returns
    ``Phi[..., beta] = sum_alpha P[..., beta, alpha] Phi[..., alpha]``.
    """
    if probability_matrix.shape[-1] not in (3, 4):
        raise ValueError("probability_matrix must have size 3 or 4 on its last axis.")
    if probability_matrix.shape[-2:] != (
        probability_matrix.shape[-1],
        probability_matrix.shape[-1],
    ):
        raise ValueError("probability_matrix must be square on its final axes.")
    flux = torch.as_tensor(
        initial_flux,
        device=probability_matrix.device,
        dtype=probability_matrix.dtype,
    )
    if flux.shape[-1] != probability_matrix.shape[-1]:
        raise ValueError("initial_flux flavour dimension must match probability_matrix.")
    return torch.einsum("...ba,...a->...b", probability_matrix, flux)


def flux_integrated_coordinate(
    flux: torch.Tensor,
    coordinate: torch.Tensor,
    *,
    dim: int = -2,
) -> torch.Tensor:
    """Integrate a flavour-resolved flux over an arbitrary coordinate."""
    if flux.shape[-1] not in (3, 4):
        raise ValueError("flux must have final flavour dimension 3 or 4.")
    axis = dim % flux.ndim
    if axis == flux.ndim - 1:
        raise ValueError("dim must not select the flavour axis.")
    x = torch.as_tensor(coordinate, device=flux.device, dtype=flux.dtype)
    if x.ndim != 1 or flux.shape[axis] != x.numel():
        raise ValueError("coordinate must be one-dimensional and match flux along dim.")
    return torch.trapezoid(flux, x=x, dim=axis)


def flux_integrated(
    flux: torch.Tensor,
    E_grid_MeV: torch.Tensor,
    *,
    energy_dim: int = -2,
) -> torch.Tensor:
    """Integrate a differential flux over energy to obtain a neutrino rate.

    Args:
        flux: Differential flux ``dPhi/dE``, flavour-resolved (final
            dimension 3 or 4, 4 for the 3+1 sterile extension) with energy
            along ``energy_dim``. The default layout is ``(..., E, 3)``; a
            leading angle axis, e.g. ``(..., E, angle, 3)`` with
            ``energy_dim=-3``, is handled the same way, keeping the flux
            resolved by angle and flavour once energy is integrated out.
        E_grid_MeV: One-dimensional energy grid in MeV, matching
            ``flux.shape[energy_dim]``.
        energy_dim: Axis of ``flux`` holding the energy grid. Must not be
            the final (flavour) axis.

    Returns:
        Flux integrated over energy (a rate, in the units of ``flux``
        times ``E_grid_MeV``), with the energy axis removed and every other
        axis -- including flavour, and angle if present -- preserved.

    Raises:
        ValueError: If ``flux`` does not have final flavour dimension 3 or
            4, if ``E_grid_MeV`` is not one-dimensional, if ``energy_dim``
            selects the flavour axis, or if the two do not match in size.
    """
    if flux.shape[-1] not in (3, 4):
        raise ValueError("flux must have final flavour dimension 3 or 4.")

    if energy_dim % flux.ndim == flux.ndim - 1:
        raise ValueError("energy_dim must not select the flavour axis.")

    E = torch.as_tensor(E_grid_MeV, device=flux.device, dtype=flux.dtype)
    if E.ndim != 1:
        raise ValueError("E_grid_MeV must be one-dimensional.")
    if flux.shape[energy_dim] != E.numel():
        raise ValueError("flux.shape[energy_dim] must match E_grid_MeV.")

    return flux_integrated_coordinate(flux, E, dim=energy_dim)


def flux_integrated_angular(
    flux: torch.Tensor,
    theta_deg: torch.Tensor,
    *,
    angular_dim: int = -2,
) -> torch.Tensor:
    """Integrate a differential flux over the full zenith range (solid angle).

    Computes the total flux over the sky, assuming azimuthal symmetry,

        Phi_total = 2*pi * integral (dPhi/dOmega) sin(theta) dtheta,

    the geometric solid-angle integral, evaluated by the trapezoidal rule.
    Unlike ``probability_integrated_angular`` (``core.common.probability``),
    this does **not** normalize: the result is the physically total,
    un-normalized rate summed over the whole angular range, in the same
    spirit as ``flux_integrated`` for energy.

    Args:
        flux: Differential flux ``dPhi/dOmega``, flavour-resolved (final
            dimension 3 or 4, 4 for the 3+1 sterile extension), with the
            zenith angle along ``angular_dim``.
        theta_deg: One-dimensional zenith-angle grid in degrees, spanning
            the range to be integrated over, matching
            ``flux.shape[angular_dim]``.
        angular_dim: Axis of ``flux`` holding the angle grid. Must not be
            the final (flavour) axis.

    Returns:
        Flux integrated over the full zenith range (a rate, in the units
        of ``flux``), with the angular axis removed and every other axis
        -- including flavour -- preserved.

    Raises:
        ValueError: If ``flux`` does not have final flavour dimension 3 or
            4, if ``theta_deg`` is not one-dimensional, if ``angular_dim``
            selects the flavour axis, or if the two do not match in size.
    """
    if flux.shape[-1] not in (3, 4):
        raise ValueError("flux must have final flavour dimension 3 or 4.")

    if angular_dim % flux.ndim == flux.ndim - 1:
        raise ValueError("angular_dim must not select the flavour axis.")

    theta = torch.as_tensor(theta_deg, device=flux.device, dtype=flux.dtype)
    if theta.ndim != 1:
        raise ValueError("theta_deg must be one-dimensional.")
    if flux.shape[angular_dim] != theta.numel():
        raise ValueError("flux.shape[angular_dim] must match theta_deg.")

    theta_rad = torch.deg2rad(theta)
    sin_theta = torch.sin(theta_rad)

    trailing = flux.ndim - (angular_dim % flux.ndim) - 1
    sin_theta_b = sin_theta
    for _ in range(trailing):
        sin_theta_b = sin_theta_b.unsqueeze(-1)

    return 2.0 * torch.pi * torch.trapezoid(flux * sin_theta_b, x=theta_rad, dim=angular_dim)
