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
coherent solar-to-earth propagation utilities.
"""



from tpeanuts.coherent.coordinates import (
    distance_to_solar_radius_fraction,
    production_to_surface_path_length,
    solar_path_grid,
    solar_radius_fraction_to_distance,
    solar_shell_widths,
)
from tpeanuts.coherent.evolution import (
    solar_radius_fraction_to_core_x,
    solar_surface_evolutor,
    solar_surface_state,
    solar_to_earth_probabilities,
    solar_to_earth_state,
)

__all__ = [
    "distance_to_solar_radius_fraction",
    "production_to_surface_path_length",
    "solar_radius_fraction_to_core_x",
    "solar_path_grid",
    "solar_radius_fraction_to_distance",
    "solar_shell_widths",
    "solar_surface_evolutor",
    "solar_surface_state",
    "solar_to_earth_probabilities",
    "solar_to_earth_state",
]
