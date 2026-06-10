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
Honda/HKKM atmospheric flux generation utilities.
"""

from tpeanuts.external.honda.generator import (
    generate_flux_for_particles_angle_grid,
    generate_flux_for_particle_angle,
)
from tpeanuts.external.honda.tables import (
    HondaTableSelection,
    find_honda_data_dir,
    honda_cosz_centers,
)

__all__ = [
    "HondaTableSelection",
    "find_honda_data_dir",
    "generate_flux_for_particle_angle",
    "generate_flux_for_particles_angle_grid",
    "honda_cosz_centers",
]
