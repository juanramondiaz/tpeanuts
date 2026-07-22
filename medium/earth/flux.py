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
Earth flux helpers.

This module composes Earth matter-regeneration probabilities with the
medium-independent flux utilities from ``core.common.flux``. It does not add
new Earth propagation physics: ``earth_probability_state`` computes final
flavour probabilities and ``flux_state``/``flux_integrated`` apply flux
normalizations, optional spectra, and energy integration.

Module functions:
    earth_flux_state(...)
        Compute final flavour-resolved Earth flux from an initial state, flux
        normalization, and optional spectral weight.
    earth_flux_integrated(...)
        Integrate the energy-resolved Earth flux over energy to obtain a
        neutrino rate.
    earth_flux_exposure(...)
        Compute exposure-integrated Earth flux from exposure-integrated Earth
        probabilities.
"""



from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

import tpeanuts.config.default as default
from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.numerical.geometry import OdeMethod
from tpeanuts.medium.earth.exposure_integration import earth_probability_exposure
from tpeanuts.medium.earth.exposure_table import ExposureParameters
from tpeanuts.medium.earth.probability import PearthMethod, earth_probability_state
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


@torch.no_grad()
def earth_flux_state(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    flux: TensorLike,
    spectrum: TensorLike | None = None,
    *,
    method: PearthMethod = default.earth_method,
    massbasis: bool = default.earth_massbasis,
    full_oscillation: bool = default.earth_full_oscillation,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = None,
    context: Optional[RuntimeContext] = None,
    reunitarize: bool = default.earth_reunitarize,
    include_matter_nc: Optional[bool] = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute final flavour-resolved Earth flux.

    Args:
        nustate: Initial state passed to ``earth_probability_state``.
            Interpreted as mass-basis incoherent weights when
            ``massbasis=True`` and as flavour-basis coherent amplitudes
            otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        eta: Detector nadir angle in radians.
        depth_m: Detector depth in metres.
        flux: Flux normalization broadcastable with the leading probability
            dimensions.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.
        method: "analytical" or "numerical" Earth probability pipeline.
        massbasis: Selects the interpretation of ``nustate``.
        full_oscillation: For numerical mode, return the full path evolution
            and x grid instead of only the final probability.
        nsteps: Numerical integration steps for method="numerical".
        ode_method: Numerical profile sampling rule.
        context: Runtime device/dtype for method="numerical"; analytical
            infers from inputs.
        reunitarize: For method="analytical", project evolution operators to
            the nearest unitary matrix.
        include_matter_nc: If True/False, applied/not applied (see
            ``medium.earth.probability.earth_probability_state``). If
            ``None`` (the default), auto-resolved per-call.

    Returns:
        Flavour-resolved Earth flux. If ``full_oscillation=True`` in numerical
        mode, returns ``(flux_along_path, x_grid)``.
    """
    method_name = str(method).lower().strip()

    probabilities = earth_probability_state(
        nustate=nustate,
        profile_earth=profile_earth,
        oscillation=oscillation,
        E_MeV=E_MeV,
        eta=eta,
        depth_m=depth_m,
        method=method,
        massbasis=massbasis,
        full_oscillation=full_oscillation,
        nsteps=nsteps,
        ode_method=ode_method,
        context=context,
        reunitarize=reunitarize,
        include_matter_nc=include_matter_nc,
    )

    if full_oscillation and method_name == "numerical":
        probabilities, x = probabilities
        return flux_state(probabilities, flux, spectrum), x

    return flux_state(probabilities, flux, spectrum)


@torch.no_grad()
def earth_flux_integrated(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    eta: TensorLike,
    depth_m: float,
    flux: TensorLike,
    spectrum: TensorLike,
    *,
    method: PearthMethod = default.earth_method,
    massbasis: bool = default.earth_massbasis,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = None,
    context: Optional[RuntimeContext] = None,
    reunitarize: bool = default.earth_reunitarize,
    energy_dim: int = -2,
    include_matter_nc: Optional[bool] = None,
) -> Tensor:
    """Integrate the energy-resolved Earth flux over energy.

    Builds the flavour-resolved Earth flux with ``earth_flux_state`` at a
    fixed nadir angle ``eta`` and integrates it over energy with
    ``core.common.flux.flux_integrated``, obtaining a physical rate
    (unnormalized, unlike ``earth_probability_integrated``). This is
    distinct from ``earth_flux_exposure``, which integrates over the nadir
    angle instead of energy.

    Args:
        nustate: Initial state passed to ``earth_flux_state``.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional.
        eta: Detector nadir angle in radians.
        depth_m: Detector depth in metres.
        flux: Flux normalization broadcastable with the leading probability
            dimensions, resolved on the same energy grid as ``E_MeV``.
        spectrum: Spectral weight broadcastable with the leading probability
            dimensions, required (no default).
        method: "analytical" or "numerical" Earth probability pipeline.
        massbasis: Selects the interpretation of ``nustate``.
        nsteps: Numerical integration steps for method="numerical".
        ode_method: Numerical profile sampling rule.
        context: Runtime device/dtype for method="numerical"; analytical
            infers from inputs.
        reunitarize: For method="analytical", project evolution operators to
            the nearest unitary matrix.
        energy_dim: Axis of the resulting flux tensor holding the energy
            grid. Must not be the final (flavour) axis.
        include_matter_nc: If True/False, applied/not applied (see
            ``earth_flux_state``). If ``None`` (the default), auto-resolved
            per-call.

    Returns:
        Flux integrated over energy (a rate), with the energy axis removed.
    """
    flux_grid = earth_flux_state(
        nustate,
        profile_earth,
        oscillation,
        E_MeV,
        eta,
        depth_m,
        flux,
        spectrum,
        method=method,
        massbasis=massbasis,
        full_oscillation=False,
        nsteps=nsteps,
        ode_method=ode_method,
        context=context,
        reunitarize=reunitarize,
        include_matter_nc=include_matter_nc,
    )

    return flux_integrated(flux_grid, E_MeV, energy_dim=energy_dim)


@torch.no_grad()
def earth_flux_exposure(
    nustate: Tensor,
    profile_earth: object,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    depth_m: float,
    flux: TensorLike,
    spectrum: TensorLike | None = None,
    *,
    method: PearthMethod = default.earth_method,
    massbasis: bool = default.earth_massbasis,
    exposure: ExposureParameters = ExposureParameters(),
    normalized_exposure: bool = default.earth_normalized_exposure,
    context: RuntimeContext = RuntimeContext.resolve(default.earth_device, default.dtype),
    chunk_eta: Optional[int] = default.earth_chunk_eta,
    reunitarize: bool = default.earth_reunitarize,
    nsteps: int = default.earth_probability_nsteps,
    ode_method: OdeMethod | None = default.earth_numerical_method,
    include_matter_nc: Optional[bool] = None,
) -> Tensor:
    """Compute exposure-integrated final flavour-resolved Earth flux.

    This function composes ``earth_probability_exposure`` with
    ``flux_state``. The angular exposure integration is performed
    at the probability level, and the resulting flavour probabilities are then
    applied to the input flux normalization and optional spectral weight.

    Args:
        nustate: Initial state passed to ``earth_probability_exposure``.
            Interpreted as mass-basis incoherent weights when
            ``massbasis=True`` and as flavour-basis coherent amplitudes
            otherwise.
        profile_earth: EarthProfile-compatible profile object.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Scalar or vector of neutrino energies in MeV.
        depth_m: Detector depth in metres.
        flux: Flux normalization broadcastable with the leading probability
            dimensions.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.
        method: "analytical" or "numerical" Earth probability pipeline.
        massbasis: Selects the interpretation of ``nustate``.
        exposure: Exposure-table construction settings. The default
            ``ExposureParameters()`` selects ``exposure_source="math"``.
        normalized_exposure: Normalize the exposure weights before
            integration.
        context: Runtime device/dtype used by the integration.
        chunk_eta: Number of eta samples evaluated per batch.
        reunitarize: For analytical propagation, project evolution operators
            to the nearest unitary matrix.
        nsteps: Number of numerical trajectory samples for numerical mode.
        ode_method: Numerical profile sampling rule for numerical mode.
        include_matter_nc: If True/False, applied/not applied (see
            ``medium.earth.exposure_integration.earth_probability_exposure``).
            If ``None`` (the default), auto-resolved per-call.

    Returns:
        Exposure-integrated flavour-resolved Earth flux.
    """
    probabilities = earth_probability_exposure(
        nustate=nustate,
        profile_earth=profile_earth,
        oscillation=oscillation,
        E_MeV=E_MeV,
        depth_m=depth_m,
        method=method,
        massbasis=massbasis,
        exposure=exposure,
        normalized_exposure=normalized_exposure,
        context=context,
        chunk_eta=chunk_eta,
        reunitarize=reunitarize,
        nsteps=nsteps,
        ode_method=ode_method,
        include_matter_nc=include_matter_nc,
    )

    return flux_state(probabilities, flux, spectrum)
