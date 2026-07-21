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
Solar source flux helpers.

This module sits above the adiabatic solar-probability functions. It selects
one or more production sources from a SolarProfile, computes their final
flavour probabilities with ``solar_probability_state``, and delegates the
generic probability-to-flux multiplication and energy integration to
``core.common.flux``.

Module functions:
    solar_flux_state(...)
        Weight one or several source probabilities by their total fluxes and
        optional spectra.
    solar_flux_integrated(...)
        Integrate the energy-resolved solar flux over energy to obtain a
        neutrino rate.
"""



from __future__ import annotations

import torch

from tpeanuts.core.common.flux import flux_integrated, flux_state
from tpeanuts.core.common.oscillation import OscillationParameters
from tpeanuts.medium.solar.probability import solar_probability_state


def solar_flux_state(
    sources: str | list[str] | tuple[str, ...],
    profile,
    oscillation: OscillationParameters,
    E_MeV,
    source_spectrum: torch.Tensor | None = None,
    *,
    legacy_precision: bool = False,
) -> torch.Tensor:
    """Compute flavour-resolved solar flux for one or more sources.

    Args:
        sources: Solar source key or ordered source keys available in
            ``profile``.
        profile: SolarProfile-like object exposing source fluxes and
            production fractions.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy in MeV.
        source_spectrum: Optional spectral weight broadcastable with the
            probability tensor. For several sources, include the leading source
            dimension when needed.
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).

    Returns:
        Flavour-resolved flux with optional leading source dimensions and final
        flavour dimension 3.
    """
    probabilities = solar_probability_state(
        oscillation,
        E_MeV,
        profile,
        sources,
        legacy_precision=legacy_precision,
    )

    if isinstance(sources, str):
        fluxes = profile.flux(sources)
    else:
        fluxes = torch.stack(
            [
                profile.flux(source)
                for source in sources
            ],
            dim=0,
        )

    return flux_state(probabilities, fluxes, source_spectrum)


def solar_flux_integrated(
    sources: str | list[str] | tuple[str, ...],
    profile,
    oscillation: OscillationParameters,
    E_MeV,
    *,
    legacy_precision: bool = False,
    energy_dim: int = -2,
) -> torch.Tensor:
    """Integrate the energy-resolved solar flux over energy.

    Builds the flavour-resolved solar flux with ``solar_flux_state`` and
    integrates it over energy with ``core.common.flux.flux_integrated``,
    obtaining a physical rate (unnormalized, unlike
    ``solar_probability_integrated``).

    Args:
        sources: Solar source key or ordered source keys available in
            ``profile``.
        profile: SolarProfile-like object exposing source fluxes and
            production fractions.
        oscillation: Built pmns object plus mass splittings and antinu
            selection.
        E_MeV: Neutrino energy grid in MeV, one-dimensional.
        legacy_precision: If True, evaluate the underlying matter-mixing
            angles with the legacy peanuts ``Vk`` prefactor for
            bit-comparable validation (see ``medium.solar.matter_mixing``).
        energy_dim: Axis of the resulting flux tensor holding the energy
            grid. Must not be the final (flavour) axis.

    Returns:
        Flux integrated over energy (a rate), with the energy axis removed.
    """
    flux_grid = solar_flux_state(
        sources,
        profile,
        oscillation,
        E_MeV,
        legacy_precision=legacy_precision,
    )

    return flux_integrated(flux_grid, E_MeV, energy_dim=energy_dim)
