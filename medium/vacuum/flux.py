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
``vacuum_probability_state`` computes the final flavour probabilities and
``flux_state``/``flux_integrated`` apply flux normalizations, optional
spectra, and energy integration.

Module functions:
    vacuum_flux_state(...)
        Compute final flavour-resolved vacuum flux from an initial state, flux
        normalization, and optional spectral weight.
    vacuum_flux_integrated(...)
        Integrate the energy-resolved vacuum flux over energy to obtain a
        neutrino rate.
"""



from __future__ import annotations

from typing import Optional

import torch

from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.vacuum.probability import vacuum_probability_state
from tpeanuts.util.constant import R_E
from tpeanuts.util.context import RuntimeContext
from tpeanuts.util.type import TensorLike


@torch.no_grad()
def vacuum_flux_state(
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
        nustate: Initial state passed to ``vacuum_probability_state``. Interpreted as
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
    probabilities = vacuum_probability_state(
        nustate,
        oscillation,
        E_MeV,
        L_km,
        massbasis=massbasis,
        context=context,
        evolution_scale_m=evolution_scale_m,
    )

    return flux_state(probabilities, flux, spectrum)


@torch.no_grad()
def vacuum_flux_integrated(
    nustate: TensorLike,
    oscillation: OscillationParameters,
    E_MeV: TensorLike,
    L_km: TensorLike,
    flux: TensorLike,
    *,
    massbasis: bool = True,
    context: Optional[RuntimeContext] = None,
    evolution_scale_m: TensorLike = R_E,
    energy_dim: int = -2,
) -> torch.Tensor:
    """Integrate the energy-resolved vacuum flux over energy.

    Builds the flavour-resolved vacuum flux with ``vacuum_flux_state`` and
    integrates it over energy with ``core.common.flux.flux_integrated``,
    obtaining a physical rate (unnormalized, unlike
    ``vacuum_probability_integrated``).

    Args:
        nustate: Initial state passed to ``vacuum_flux_state``.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional.
        L_km: Propagation baseline in km.
        flux: Flux normalization broadcastable with the leading probability
            dimensions, resolved on the same energy grid as ``E_MeV``.
        massbasis: Selects the interpretation of ``nustate``.
        context: Optional runtime device/dtype.
        evolution_scale_m: Positive scale in metres used for kinetic phases.
        energy_dim: Axis of the resulting flux tensor holding the energy
            grid. Must not be the final (flavour) axis.

    Returns:
        Flux integrated over energy (a rate), with the energy axis removed.
    """
    flux_grid = vacuum_flux_state(
        nustate,
        oscillation,
        E_MeV,
        L_km,
        flux,
        massbasis=massbasis,
        context=context,
        evolution_scale_m=evolution_scale_m,
    )

    return flux_integrated(flux_grid, E_MeV, energy_dim=energy_dim)
