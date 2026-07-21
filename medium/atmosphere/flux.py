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
Atmosphere flux helpers.

This module composes atmosphere flavour probabilities with the
medium-independent flux utilities from ``core.common.flux``. It does not add
new propagation physics: ``atmosphere_probability_state`` computes final
flavour probabilities at the Earth surface and ``flux_state``/
``flux_integrated``/``flux_integrated_angular`` apply flux normalizations,
optional spectra, and energy/angular integration.

Detector-composition helpers (weighting a surface/detector flux by production
data, integrating over production height, and summing produced-flavour
contributions) do not live here: the atmosphere medium only propagates
production -> surface, and surface -> detector composition is a pipeline
concern (see ``pipeline.atmosphere_earth``).

Module functions:
    atmosphere_flux_state(...)
        Compute final flavour-resolved atmosphere flux at the Earth surface
        from an initial state, flux normalization, and optional spectral
        weight.
    atmosphere_flux_integrated(...)
        Integrate the atmosphere flux over energy (always), optionally
        chaining height and/or angular integration.
    atmosphere_flux_integrated_angular(...)
        Integrate the atmosphere flux over the full zenith range.
    atmosphere_flux_integrated_height(...)
        Integrate the atmosphere flux over production height using the
        height-resolved production flux tables. Atmosphere-specific: unlike
        the other functions in this module, it does not call any
        ``core.common`` function directly.
"""



from __future__ import annotations

from typing import Literal, Optional

import torch

from tpeanuts.core.common.flux import (
    flux_integrated,
    flux_integrated_angular,
    flux_integrated_coordinate,
    flux_state,
)
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.atmosphere.probability import (
    atmosphere_probability_integrated_height,
    atmosphere_probability_state,
)
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


@torch.no_grad()
def atmosphere_flux_state(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    flux: TensorLike,
    spectrum: TensorLike | None = None,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Compute final flavour-resolved atmosphere flux at the Earth surface.

    Args:
        nustate: Initial state passed to ``atmosphere_probability_state``.
            Interpreted as a coherent flavour-basis amplitude vector when
            ``massbasis=False`` and as incoherent mass weights when
            ``massbasis=True``.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle in degrees.
        flux: Flux normalization broadcastable with the leading probability
            dimensions.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method passed to
            ``atmosphere_probability_state``.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.

    Returns:
        Flavour-resolved atmosphere flux at the Earth surface, with final
        flavour dimension 3.
    """
    probabilities = atmosphere_probability_state(
        nustate=nustate,
        oscillation=oscillation,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        massbasis=massbasis,
        method=method,
        atmosphere=atmosphere,
        context=context,
    )

    return flux_state(probabilities, flux, spectrum)


@torch.no_grad()
def atmosphere_flux_integrated(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    flux: TensorLike,
    spectrum: TensorLike | None = None,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    energy_dim: int = -2,
    integrate_angular: bool = False,
    angular_dim: int = -2,
    integrate_height: bool = False,
    height_dim: int = -2,
) -> torch.Tensor:
    """Integrate the atmosphere flux over energy, with optional extra axes.

    Always integrates over energy (a physical rate, unnormalized -- see
    ``core.common.flux.flux_integrated``). Height and/or angular integration
    are optional, independent reductions chained by hand afterward when
    requested. The height reduction is a plain trapezoidal integral: ``flux``
    already carries whatever height dependence it has (it broadcasts against
    the leading probability dimensions, same as ``flux_state``), so no
    separate weight is needed here. See the module docstring in
    ``medium.atmosphere.probability`` for the grid axis-ordering convention
    each successive ``*_dim=-2`` reduction assumes.

    Args:
        nustate: Initial state passed to ``atmosphere_flux_state``.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV.
        h_km: Production altitude grid in km.
        theta_deg: Atmosphere zenith angle grid in degrees.
        flux: Flux normalization broadcastable with the leading probability
            dimensions.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype.
        energy_dim: Axis of the resulting flux tensor holding the energy
            grid.
        integrate_angular: If True, additionally integrate over the full
            zenith range after the (optional) height reduction.
        angular_dim: Axis holding the zenith-angle grid at the time the
            angular reduction runs.
        integrate_height: If True, additionally integrate over production
            height right after the energy reduction.
        height_dim: Axis holding the height grid at the time the height
            reduction runs.

    Returns:
        Flux integrated over energy (a rate), with the height and/or angular
        axes also removed when their flags are set.
    """
    flux_grid = atmosphere_flux_state(
        nustate=nustate,
        oscillation=oscillation,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        flux=flux,
        spectrum=spectrum,
        depth_km=depth_km,
        massbasis=massbasis,
        method=method,
        atmosphere=atmosphere,
        context=context,
    )

    result = flux_integrated(flux_grid, E_MeV, energy_dim=energy_dim)

    if integrate_height:
        result = flux_integrated_coordinate(result, h_km, dim=height_dim)

    if integrate_angular:
        result = flux_integrated_angular(result, theta_deg, angular_dim=angular_dim)

    return result


@torch.no_grad()
def atmosphere_flux_integrated_angular(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    flux: TensorLike,
    spectrum: TensorLike | None = None,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    angular_dim: int = -2,
) -> torch.Tensor:
    """Integrate the atmosphere flux over the full zenith range.

    Builds the angle-resolved flux with ``atmosphere_flux_state`` and
    integrates it with ``core.common.flux.flux_integrated_angular``.

    Args:
        nustate: Initial state passed to ``atmosphere_flux_state``.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle grid in degrees, spanning the
            range to be integrated over.
        flux: Flux normalization broadcastable with the leading probability
            dimensions.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype.
        angular_dim: Axis of the resulting flux tensor holding the angle
            grid.

    Returns:
        Flux integrated over the full zenith range (a rate), with the
        angular axis removed.
    """
    flux_grid = atmosphere_flux_state(
        nustate=nustate,
        oscillation=oscillation,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        flux=flux,
        spectrum=spectrum,
        depth_km=depth_km,
        massbasis=massbasis,
        method=method,
        atmosphere=atmosphere,
        context=context,
    )

    return flux_integrated_angular(flux_grid, theta_deg, angular_dim=angular_dim)


@torch.no_grad()
def atmosphere_flux_integrated_height(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    production_flux: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
    height_dim: int = -2,
) -> torch.Tensor:
    """Integrate the atmosphere flux over production height.

    Atmosphere-specific: works directly from the tabulated
    production-flux-by-height data and, unlike the other functions in this
    module, does not call any ``core.common`` function directly. It combines
    two pieces:

        1. ``atmosphere_probability_integrated_height`` -- the
           production-flux-weighted average probability over height,
           ``<P>_h``.
        2. The plain trapezoidal integral of ``production_flux`` itself over
           height, ``Phi_h = integral production_flux(h) dh``.

    The product ``<P>_h * Phi_h`` is exactly
    ``integral P(h) production_flux(h) dh``, i.e. the height-integrated flux,
    by the definition of the weighted average.

    Args:
        nustate: Initial state passed to
            ``atmosphere_probability_integrated_height``.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude grid in km, spanning the range to be
            integrated over.
        theta_deg: Atmosphere zenith angle in degrees.
        production_flux: Height-resolved production flux, e.g. the
            ``phi_Eh``/``f_Eh`` tables selected by
            ``pipeline.atmosphere.select_production_flux``.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.
        height_dim: Axis of the resulting probability/flux tensors holding
            the height grid.

    Returns:
        Flux integrated over production height (a rate), with the height
        axis removed.
    """
    mean_probability = atmosphere_probability_integrated_height(
        nustate=nustate,
        oscillation=oscillation,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        production_flux=production_flux,
        depth_km=depth_km,
        massbasis=massbasis,
        method=method,
        atmosphere=atmosphere,
        context=context,
        legacy_precision=legacy_precision,
        height_dim=height_dim,
    )

    production_flux_t = torch.as_tensor(
        production_flux,
        device=mean_probability.device,
        dtype=mean_probability.dtype,
    )
    h = torch.as_tensor(h_km, device=mean_probability.device, dtype=mean_probability.dtype)
    # production_flux carries no flavour axis, so its own height axis is its
    # last dimension -- the same convention core.common.probability_integrated
    # already assumes for the "spectrum" argument (see
    # atmosphere_probability_integrated_height / probability_integrated).
    total_flux = torch.trapezoid(production_flux_t, x=h, dim=-1)

    return mean_probability * total_flux.unsqueeze(-1)
