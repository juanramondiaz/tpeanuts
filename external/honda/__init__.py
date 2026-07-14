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
Honda/HKKM Atmosphere flux generation utilities.

The Honda (also called HKKM, after Honda-Kajita-Kasahara-Midorikawa) tables
are an external set of Monte Carlo calculations of the atmospheric neutrino
flux produced by cosmic-ray air showers at a given experimental site. They
tabulate the differential flux Phi(E, cosZ) in (m^2 s sr GeV)^-1 for nue,
antinue, numu and antinumu, together with the production-height distribution
of those neutrinos as a function of energy and zenith angle. This package
is a thin tpeanuts-native wrapper: it does not implement the Honda
calculation itself (which lives in the external .d.gz data files), but reads
those files (``tables``) and turns them, by interpolation onto a tpeanuts
energy/height grid, into the height-differential source flux
Phi(E,h) = Phi(E; X_obs) * f(h|E,theta) used by the atmosphere propagation
pipeline (``generator``).

Module contents:
    generate_flux_for_particle_angle(...)
        Build one Honda-derived height-differential flux file for a single
        particle and zenith/detector angle.
    generate_flux_for_particles_angle_grid(...)
        Run generate_flux_for_particle_angle over a grid of particles and
        angles, optionally in parallel.
    HondaTableSelection
        Dataclass selecting which Honda table variant (site, season, solar
        activity, mountain profile, angular binning) to read.
    find_honda_data_dir(...)
        Locate the local directory containing the Honda .d.gz table files.
    load_honda_tables(...)
        Load one Honda flux table plus per-particle production-height tables.
    honda_cosz_centers(...)
        Return the fixed cos(zenith) bin centers used by the standard
        20-bin Honda zenith binning.
"""

from tpeanuts.external.honda.generator import (
    generate_flux_for_particles_angle_grid,
    generate_flux_for_particle_angle,
)
from tpeanuts.external.honda.tables import (
    HondaTableSelection,
    find_honda_data_dir,
    honda_cosz_centers,
    load_honda_tables,
)

__all__ = [
    "HondaTableSelection",
    "find_honda_data_dir",
    "generate_flux_for_particle_angle",
    "generate_flux_for_particles_angle_grid",
    "honda_cosz_centers",
    "load_honda_tables",
]
