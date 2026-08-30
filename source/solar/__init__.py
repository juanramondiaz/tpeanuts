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
Solar-neutrino source: production, total flux, and production spectra.

Provider-neutral physics describing what the Sun produces (radial
production distributions per source, total per-source fluxes, and
per-source energy spectra), independent of solar propagation physics
(``medium.solar``, which consumes a built ``SolarNeutrinoSource`` alongside
its own ``SolarMediumProfile``).

Package contents:
    io
        CSV loaders for the configured production, flux, and spectrum
        tables, and their provider tables.
    model
        SolarNeutrinoSource container, SolarSourceParameters construction
        settings, and build_solar_source.
"""

from tpeanuts.source.solar.io import (
    available_solar_spectrum_sources,
    load_solar_fluxes,
    load_solar_production,
    load_solar_spectrum,
    load_spectrum_csv,
    solar_source_provider_path,
    solar_spectrum_path,
)
from tpeanuts.source.solar.model import (
    ContinuousSolarSpectrum,
    SolarLineSpectrum,
    SolarNeutrinoSource,
    SolarSourceParameters,
    SolarSpectrum,
    build_solar_source,
)

__all__ = [
    "available_solar_spectrum_sources",
    "load_solar_fluxes",
    "load_solar_production",
    "load_solar_spectrum",
    "load_spectrum_csv",
    "solar_source_provider_path",
    "solar_spectrum_path",
    "SolarNeutrinoSource",
    "ContinuousSolarSpectrum",
    "SolarLineSpectrum",
    "SolarSourceParameters",
    "SolarSpectrum",
    "build_solar_source",
]
