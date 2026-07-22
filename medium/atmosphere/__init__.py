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
Torch-native atmosphere neutrino utilities.

Package contents:
    geometry
        Angle conversions, spherical path lengths, altitude grids, and
        peanuts-compatible nadir coordinates.
    density
        Atmosphere mass-density and electron-density sources, including
        exponential, nuSQuIDS EarthAtm, file, MCEq, and external pymsis/MSIS
        backends.
    profile
        Numerical trajectory samples and electron-density profiles for
        atmosphere propagation.
    evolutor
        Segmented atmosphere evolution operators from production height to
        Earth surface.
    probabilities
        Transition matrices and final state probabilities at the Earth
        surface, plus energy/angular/height-integrated variants.
    flux
        Flux normalization helpers built on top of atmosphere probabilities,
        plus energy/angular/height-integrated variants.
    io
        Readers and writers for Atmosphere height-flux datasets.

Atmosphere-plus-Earth flux workflows, including surface-to-detector
composition, live in tpeanuts.pipeline (see
``pipeline.atmosphere_earth``) rather than in this package.
"""

from tpeanuts.medium.atmosphere.density import (
    atmosphere_density,
    atmosphere_mass_density_profile_exponential,
    atmosphere_mass_density_profile_from_file,
)
from tpeanuts.medium.atmosphere.depth import (
    alpha_deg_to_cos,
    atmosphere_slant_depth,
    atmosphere_vertical_depth,
    compute_dXdh,
    interpolate_flux_at_Xobs,
)
from tpeanuts.external.nusquids.density import (
    atmosphere_density_nusquids,
)
from tpeanuts.medium.atmosphere.profile import AtmosphereProfile
from tpeanuts.medium.atmosphere.evolutor import (
    atmosphere_evolutor,
    atmosphere_evolutor_analytical,
    atmosphere_evolutor_numerical,
)
from tpeanuts.medium.atmosphere.probability import (
    atmosphere_probability_transition,
    atmosphere_probability_state,
    atmosphere_probability_integrated,
    atmosphere_probability_integrated_angular,
    atmosphere_probability_integrated_height,
)
from tpeanuts.medium.atmosphere.flux import (
    atmosphere_flux_state,
    atmosphere_flux_integrated,
    atmosphere_flux_integrated_angular,
    atmosphere_flux_integrated_height,
)
from tpeanuts.medium.atmosphere.io import (
    AtmosphericFluxTable,
    OutputConfig,
    load_atmospheric_flux,
    load_directory,
)

__all__ = [
    "atmosphere_density",
    "atmosphere_mass_density_profile_exponential",
    "atmosphere_mass_density_profile_from_file",
    "alpha_deg_to_cos",
    "atmosphere_slant_depth",
    "atmosphere_vertical_depth",
    "compute_dXdh",
    "interpolate_flux_at_Xobs",
    "atmosphere_density_nusquids",
    "AtmosphereProfile",
    "atmosphere_evolutor",
    "atmosphere_evolutor_analytical",
    "atmosphere_evolutor_numerical",
    "atmosphere_probability_transition",
    "atmosphere_probability_state",
    "atmosphere_probability_integrated",
    "atmosphere_probability_integrated_angular",
    "atmosphere_probability_integrated_height",
    "atmosphere_flux_state",
    "atmosphere_flux_integrated",
    "atmosphere_flux_integrated_angular",
    "atmosphere_flux_integrated_height",
    "OutputConfig",
    "AtmosphericFluxTable",
    "load_atmospheric_flux",
    "load_directory",
]
