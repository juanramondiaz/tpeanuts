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
Atmosphere neutrino oscillation probabilities.

This module converts the atmosphere evolution operator into flavour
probabilities. It sits above ``medium.atmosphere.evolutor``: the evolutor
builds either the perturbative analytical or segmented numerical operator from
production altitude to the Earth surface, while this module projects that
operator into probabilities for a supplied initial state.

Two input-state conventions are supported by ``atmosphere_probability_state``:

    massbasis=False
        ``nustate`` is a coherent flavour-basis amplitude vector. The final
        state is ``psi_surface = S_atm psi_initial`` and
        ``P_alpha = |psi_surface,alpha|^2``.

    massbasis=True
        ``nustate`` is an incoherent mass-basis weight vector ``w_i``. The
        final flavour probability is
        ``P_alpha = sum_i |(S_atm U_PMNS)_{alpha i}|^2 w_i``.

Integration axis convention
----------------------------
``atmosphere_probability_integrated`` always integrates over energy (using an
explicit production ``spectrum``, matching the other media) and can
additionally chain height and/or angular integration by hand via
``integrate_height``/``integrate_angular``. Every ``*_dim`` parameter across
this module defaults to ``-2``, i.e. "the axis immediately before flavour" --
this is the same convention ``core.common.probability`` already uses. For the
standalone ``atmosphere_probability_integrated_angular``/
``atmosphere_probability_integrated_height``, that just means putting the
axis you are integrating over last before flavour. For the composite
``atmosphere_probability_integrated``, which reduces energy, then
(optionally) height, then (optionally) angle, in that fixed sequence, callers
who want all three defaults to line up must lay out the input grid as
``(..., theta, h, E, flavour)``: each reduction removes the axis then
immediately before flavour, which promotes the next one into that slot for
the following step.

Module functions:
    atmosphere_probability_transition(...)
        Compute the full atmosphere flavour-transition probability matrix.
    atmosphere_probability_state(...)
        Compute final atmosphere-surface flavour probabilities for an initial
        coherent flavour state or incoherent mass mixture.
    atmosphere_probability_integrated(...)
        Average final flavour probabilities over energy (always), optionally
        chaining height and/or angular integration.
    atmosphere_probability_integrated_angular(...)
        Average final flavour probabilities over the full zenith range.
    atmosphere_probability_integrated_height(...)
        Average final flavour probabilities over production height, weighted
        by the height-resolved production flux.
"""



from __future__ import annotations

from typing import Literal, Optional

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import (
    probability_integrated,
    probability_integrated_angular,
    probability_weighted_average,
    probability_state,
    probability_transition,
)
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import default_device, resolve_dtype
from tpeanuts.util.type import (
    TensorLike,
    broadcast_flavour_vector,
    cdtype_from_real,
    state_tensor,
)


@torch.no_grad()
def atmosphere_probability_transition(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute all atmosphere flavour-transition probabilities.

    Args:
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below the Earth surface in km.
        atmosphere: Atmosphere density profile construction settings.
        method: Atmosphere evolution method.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.

    Returns:
        Real tensor ``P=|S_atm|^2`` with shape ``(..., N, N)``, N in {3, 4}.
        The final two dimensions are final flavour and initial flavour.
    """
    S_atm, _ = atmosphere_evolutor(
        oscillation,
        E_MeV,
        h_km,
        theta_deg,
        depth_km,
        atmosphere=atmosphere,
        method=method,
        context=context,
        legacy_precision=legacy_precision,
    )

    return probability_transition(S_atm, real_dtype=context.dtype if context is not None else None)


@torch.no_grad()
def atmosphere_probability_state(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute final atmosphere-surface flavour probabilities.

    Args:
        nustate: Initial state with final dimension matching
            ``oscillation.pmns.n_flavours`` (3, or 4 for the 3+1 sterile
            extension). When massbasis=False, this is a coherent
            flavour-basis amplitude vector. When massbasis=True, this is
            interpreted as incoherent mass-basis weights.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: ``"analytical"`` for perturbative polynomial propagation or
            ``"numerical"`` for sampled matrix-exponential propagation.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.

    Returns:
        Real tensor of final flavour probabilities at the Earth surface, with
        final dimension matching ``oscillation.pmns.n_flavours``.
    """
    if context is not None:
        device, dtype = context.device, context.dtype
    else:
        device = default_device(None)
        dtype = resolve_dtype(None, E_MeV, h_km, theta_deg, depth_km)
    resolved_context = RuntimeContext(device=device, dtype=dtype)

    S_atm, _ = atmosphere_evolutor(
        oscillation,
        E_MeV,
        h_km,
        theta_deg,
        depth_km,
        atmosphere=atmosphere,
        method=method,
        context=resolved_context,
        legacy_precision=legacy_precision,
    )

    if massbasis:
        state = state_tensor(nustate, device=device, dtype=dtype)
    else:
        state = state_tensor(
            nustate,
            device=device,
            dtype=cdtype_from_real(dtype),
        )
    state = broadcast_flavour_vector(state, S_atm.shape[:-2])

    return probability_state(
        S_atm,
        state,
        pmns=oscillation.pmns,
        massbasis=massbasis,
        antinu=oscillation.antinu,
        real_dtype=dtype,
    )


@torch.no_grad()
def atmosphere_probability_integrated(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    spectrum: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
    energy_dim: int = -2,
    integrate_angular: bool = False,
    angular_dim: int = -2,
    integrate_height: bool = False,
    height_weight: TensorLike | None = None,
    height_dim: int = -2,
) -> torch.Tensor:
    """Average final atmosphere flavour probabilities over energy.

    Always integrates over energy, weighted by an explicit production
    ``spectrum`` (see ``core.common.probability.probability_integrated``).
    Height and/or angular integration are optional, independent reductions
    chained by hand afterward when requested -- each one calls its own
    lower-level core function rather than being folded into a single generic
    reducer. See the module docstring for the grid axis-ordering convention
    each successive ``*_dim=-2`` reduction assumes.

    Args:
        nustate: Initial state passed to ``atmosphere_probability_state``.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy grid in MeV.
        h_km: Production altitude grid in km.
        theta_deg: Atmosphere zenith angle grid in degrees.
        spectrum: Spectral weight w(E), required (no default).
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.
        energy_dim: Axis of the resulting probability tensor holding the
            energy grid.
        integrate_angular: If True, additionally average over the full
            zenith range after the (optional) height reduction.
        angular_dim: Axis holding the zenith-angle grid at the time the
            angular reduction runs.
        integrate_height: If True, additionally average over production
            height (weighted by ``height_weight``) right after the energy
            reduction.
        height_weight: Height weight w(h), required when
            ``integrate_height=True``.
        height_dim: Axis holding the height grid at the time the height
            reduction runs.

    Returns:
        Probability tensor with the energy axis removed, and the height
        and/or angular axes also removed when their flags are set.
    """
    result = atmosphere_probability_state(
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
        legacy_precision=legacy_precision,
    )

    result = probability_integrated(result, E_MeV, spectrum, energy_dim=energy_dim)

    if integrate_height:
        if height_weight is None:
            raise ValueError("height_weight is required when integrate_height=True.")
        result = probability_weighted_average(
            result, h_km, height_weight, dim=height_dim
        )

    if integrate_angular:
        result = probability_integrated_angular(result, theta_deg, angular_dim=angular_dim)

    return result


@torch.no_grad()
def atmosphere_probability_integrated_angular(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    method: Literal["analytical", "numerical"] = "numerical",
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    legacy_precision: bool = False,
    angular_dim: int = -2,
) -> torch.Tensor:
    """Average final atmosphere flavour probabilities over the full zenith range.

    Builds the angle-resolved probabilities with ``atmosphere_probability_state``
    and averages them with
    ``core.common.probability.probability_integrated_angular``, weighted by
    the geometric solid-angle element sin(theta).

    Args:
        nustate: Initial state passed to ``atmosphere_probability_state``.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle grid in degrees, spanning the
            range to be averaged over.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.
        angular_dim: Axis of the resulting probability tensor holding the
            angle grid.

    Returns:
        Solid-angle-weighted average probability, with the angular axis
        removed.
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
        legacy_precision=legacy_precision,
    )

    return probability_integrated_angular(probabilities, theta_deg, angular_dim=angular_dim)


@torch.no_grad()
def atmosphere_probability_integrated_height(
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
    """Average final atmosphere flavour probabilities over production height.

    Builds the height-resolved probabilities with
    ``atmosphere_probability_state`` and averages them with
    ``core.common.probability.probability_integrated``, weighted by the
    height-resolved production flux. Height is not a universal core concept
    (only the atmosphere medium propagates over a production-altitude axis),
    so this wrapper lives here rather than in ``core.common.probability``,
    even though it reuses that module's generic weighted-average primitive.

    Args:
        nustate: Initial state passed to ``atmosphere_probability_state``.
        oscillation: Built pmns object plus mass splittings, antinu
            selection, and the optional ``nsi`` (NSIConfig) attribute.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude grid in km, spanning the range to be
            averaged over.
        theta_deg: Atmosphere zenith angle in degrees.
        production_flux: Height-resolved production flux weight, e.g. the
            ``phi_Eh``/``f_Eh`` tables selected by
            ``pipeline.atmosphere.select_production_flux``.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        method: Atmosphere propagation method.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.
        height_dim: Axis of the resulting probability tensor holding the
            height grid.

    Returns:
        Production-flux-weighted average probability, with the height axis
        removed.
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
        legacy_precision=legacy_precision,
    )

    return probability_weighted_average(
        probabilities, h_km, production_flux, dim=height_dim
    )
