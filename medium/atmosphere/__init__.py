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
        surface.
    flux
        Flux normalization helpers built on top of atmosphere probabilities.
    io
        Readers and writers for Atmosphere height-flux datasets.

Atmosphere-plus-Earth flux workflows live in tpeanuts.pipeline.
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
from tpeanuts.medium.atmosphere.evolutor import atmosphere_evolutor
from tpeanuts.medium.atmosphere.probability import (
    atmosphere_probability,
    patmosphere,
)
from tpeanuts.medium.atmosphere.flux import atmosphere_flux
from tpeanuts.medium.atmosphere.io import OutputConfig, load_directory

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
    "atmosphere_probability",
    "patmosphere",
    "atmosphere_flux",
    "OutputConfig",
    "load_directory",
]
