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
new propagation physics: ``patmosphere`` computes final flavour probabilities
at the Earth surface and ``flux_from_probability`` applies flux normalizations
and optional spectra.

Module functions:
    atmosphere_flux(...)
        Compute final flavour-resolved atmosphere flux at the Earth surface
        from an initial state, flux normalization, and optional spectral
        weight.
"""



from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.flux import flux_from_probability
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.atmosphere.probability import patmosphere
from tpeanuts.medium.atmosphere.profile import AtmosphereParameters
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


@torch.no_grad()
def atmosphere_flux(
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
    atmosphere: Optional[AtmosphereParameters] = None,
    context: Optional[RuntimeContext] = None,
) -> torch.Tensor:
    """Compute final flavour-resolved atmosphere flux at the Earth surface.

    Args:
        nustate: Initial state passed to ``patmosphere``. Interpreted as a
            coherent flavour-basis amplitude vector when ``massbasis=False``
            and as incoherent mass weights when ``massbasis=True``.
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
        atmosphere: Atmosphere density profile construction settings.
        context: Optional runtime device/dtype. If omitted, both are inferred
            from the tensor inputs.

    Returns:
        Flavour-resolved atmosphere flux at the Earth surface, with final
        flavour dimension 3.
    """
    probabilities = patmosphere(
        nustate=nustate,
        oscillation=oscillation,
        E_MeV=E_MeV,
        h_km=h_km,
        theta_deg=theta_deg,
        depth_km=depth_km,
        massbasis=massbasis,
        atmosphere=atmosphere,
        context=context,
    )

    return flux_from_probability(probabilities, flux, spectrum)
