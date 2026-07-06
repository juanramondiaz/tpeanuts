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
Vacuum flux helpers.

This module composes vacuum flavour probabilities with the medium-independent
flux utilities from ``core.common.flux``. It does not add new vacuum physics:
``pvacuum`` computes the final flavour probabilities and
``flux_from_probability`` applies flux normalizations and optional spectra.

Module functions:
    vacuum_flux(...)
        Compute final flavour-resolved vacuum flux from an initial state, flux
        normalization, and optional spectral weight.
"""



from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.flux import flux_from_probability
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.vacuum.probability import pvacuum
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


@torch.no_grad()
def vacuum_flux(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    flux: TensorLike,
    spectrum: TensorLike | None = None,
    *,
    massbasis: bool = True,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
) -> torch.Tensor:
    """Compute final flavour-resolved vacuum flux.

    Args:
        nustate: Initial state passed to ``pvacuum``. Interpreted as
            mass-basis incoherent weights when ``massbasis=True`` and as
            flavour-basis coherent amplitudes otherwise.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        L_km: Propagation baseline in km.
        flux: Flux normalization broadcastable with the leading probability
            dimensions.
        spectrum: Optional spectral weight broadcastable with the leading
            probability dimensions.
        massbasis: Selects the interpretation of ``nustate``.
        context: Optional runtime device/dtype.
        evolution_scale_m: Positive scale in metres used for kinetic phases.

    Returns:
        Flavour-resolved vacuum flux with final flavour dimension 3.
    """
    probabilities = pvacuum(
        nustate,
        oscillation,
        E_MeV,
        L_km,
        massbasis=massbasis,
        context=context,
        evolution_scale_m=evolution_scale_m,
    )

    return flux_from_probability(probabilities, flux, spectrum)
