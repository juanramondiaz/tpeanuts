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
builds the segmented numerical operator from production altitude to the Earth
surface, while this module projects that operator into either a full transition
matrix or final flavour probabilities for a supplied initial state.

Two input-state conventions are supported by ``patmosphere``:

    massbasis=False
        ``nustate`` is a coherent flavour-basis amplitude vector. The final
        state is ``psi_surface = S_atm psi_initial`` and
        ``P_alpha = |psi_surface,alpha|^2``.

    massbasis=True
        ``nustate`` is an incoherent mass-basis weight vector ``w_i``. The
        final flavour probability is
        ``P_alpha = sum_i |(S_atm U_PMNS)_{alpha i}|^2 w_i``.

Module functions:
    atmosphere_probability(...)
        Compute the full atmosphere flavour-transition probability matrix.
    patmosphere(...)
        Compute final atmosphere-surface flavour probabilities for an initial
        coherent flavour state or incoherent mass mixture.
"""



from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.core.common.probability import (
    probability_from_evolutor,
    probability_transition,
)
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.torch_util import default_device, resolve_dtype
from tpeanuts.util.type import (
    TensorLike,
    broadcast_last3,
    cdtype_from_real,
    state_tensor,
)


@torch.no_grad()
def atmosphere_probability(
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute all atmosphere flavour-transition probabilities.

    Args:
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below the Earth surface in km.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.
        epsilon: Optional NSI matrix passed to the atmosphere evolutor.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.

    Returns:
        Real tensor ``P=|S_atm|^2`` with shape ``(..., 3, 3)``. The final two
        dimensions are final flavour and initial flavour.
    """
    S_atm, _ = atmosphere_evolutor(
        oscillation,
        E_MeV,
        h_km,
        theta_deg,
        depth_km,
        atmosphere=atmosphere,
        context=context,
        epsilon=epsilon,
        legacy_precision=legacy_precision,
    )

    return probability_transition(S_atm, real_dtype=context.dtype if context is not None else None)


@torch.no_grad()
def patmosphere(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    h_km: TensorLike,
    theta_deg: TensorLike,
    depth_km: TensorLike = 0.0,
    *,
    massbasis: bool = False,
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
    epsilon: Optional[torch.Tensor] = None,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """
    Compute final atmosphere-surface flavour probabilities.

    Args:
        nustate: Initial state with final dimension 3. When massbasis=False,
            this is a coherent flavour-basis amplitude vector. When
            massbasis=True, this is interpreted as incoherent mass-basis
            weights.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        h_km: Production altitude in km.
        theta_deg: Atmosphere zenith angle in degrees.
        depth_km: Detector depth below the Earth surface in km.
        massbasis: Selects the interpretation of ``nustate``.
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.
        epsilon: Optional NSI matrix passed to the atmosphere evolutor.
        legacy_precision: If True, use the legacy peanuts matter-potential
            prefactor in atmosphere propagation.

    Returns:
        Real tensor of final flavour probabilities at the Earth surface, with
        final dimension 3.
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
        context=resolved_context,
        epsilon=epsilon,
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
    state = broadcast_last3(state, S_atm.shape[:-2])

    return probability_from_evolutor(
        S_atm,
        state,
        pmns=oscillation.pmns,
        massbasis=massbasis,
        antinu=oscillation.antinu,
        real_dtype=dtype,
    )
