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
solar source and spectrum integration helpers.
"""



from __future__ import annotations

import torch

from tpeanuts.solar.probabilities import psolar, psolar_sources


def psolar_source(
    source: str,
    profile,
    pmns,
    dm21_eV2,
    dm3l_eV2,
    E_MeV,
    *,
    antinu: bool | torch.Tensor = False,
) -> torch.Tensor:
    return psolar(
        pmns,
        dm21_eV2,
        dm3l_eV2,
        E_MeV,
        profile.radius,
        profile.density,
        profile.production_fraction(source),
        antinu=antinu,
    )


def psolar_source_batch(
    sources: list[str] | tuple[str, ...],
    profile,
    pmns,
    dm21_eV2,
    dm3l_eV2,
    E_MeV,
    *,
    antinu: bool | torch.Tensor = False,
) -> torch.Tensor:
    fractions = torch.stack(
        [
            profile.production_fraction(source)
            for source in sources
        ],
        dim=0,
    )

    return psolar_sources(
        pmns,
        dm21_eV2,
        dm3l_eV2,
        E_MeV,
        profile.radius,
        profile.density,
        fractions,
        antinu=antinu,
    )


def solar_flux_by_flavour(
    source: str,
    profile,
    pmns,
    dm21_eV2,
    dm3l_eV2,
    E_MeV,
    source_spectrum: torch.Tensor | None = None,
    *,
    antinu: bool | torch.Tensor = False,
) -> torch.Tensor:
    probabilities = psolar_source(
        source,
        profile,
        pmns,
        dm21_eV2,
        dm3l_eV2,
        E_MeV,
        antinu=antinu,
    )
    flux = profile.flux(source)

    out = probabilities * flux

    if source_spectrum is not None:
        out = out * source_spectrum[..., None]

    return out


def solar_flux_by_flavour_batch(
    sources: list[str] | tuple[str, ...],
    profile,
    pmns,
    dm21_eV2,
    dm3l_eV2,
    E_MeV,
    source_spectra: torch.Tensor | None = None,
    *,
    antinu: bool | torch.Tensor = False,
) -> torch.Tensor:
    probabilities = psolar_source_batch(
        sources,
        profile,
        pmns,
        dm21_eV2,
        dm3l_eV2,
        E_MeV,
        antinu=antinu,
    )

    flux = torch.stack(
        [
            profile.flux(source)
            for source in sources
        ],
        dim=0,
    )

    out = probabilities * flux.reshape(-1, *((1,) * (probabilities.ndim - 2)), 1)

    if source_spectra is not None:
        out = out * source_spectra[..., None]

    return out
