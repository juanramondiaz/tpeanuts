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
Neutrino flux propagation workflows.
"""



from tpeanuts.flux_propagation.pipeline_atmosphere import (
    select_particle_angle_flux,
    propagate_atmosphere_coherent,
    propagate_earth_coherent,
    integrate_initial_and_surface_fluxes,
    integrate_height_and_sum_flavours,
)
from tpeanuts.flux_propagation.pipeline_coherent import (
    detector_conversion_spectra_from_coherent_result,
    load_coherent_solar_detector_result,
    propagate_solar_to_detector_coherent,
    run_and_save_solar_to_detector_coherent,
    save_coherent_solar_detector_result,
)
from tpeanuts.flux_propagation.pipeline_incoherent import (
    load_incoherent_solar_detector_result,
    propagate_solar_to_detector_incoherent,
    run_and_save_solar_to_detector_incoherent,
    save_incoherent_solar_detector_result,
)
from tpeanuts.flux_propagation.pipeline_legacypeanuts import (
    load_legacypeanuts_solar_detector_result,
    propagate_solar_to_detector_legacypeanuts,
    run_and_save_solar_to_detector_legacypeanuts,
    save_legacypeanuts_solar_detector_result,
)
from tpeanuts.io.io_flux_propagation import (
    aggregate_detector_conversion_by_mode,
    aggregate_detector_flux_by_mode,
    build_detector_flux_filename,
    build_detector_flux_path,
    detector_initial_flavour,
    detector_particle_mode,
    load_detector_flux_directory,
    load_detector_flux_result,
    save_detector_flux_result,
)
from tpeanuts.util.math import relative_error_summary


__all__ = [
    "select_particle_angle_flux",
    "propagate_atmosphere_coherent",
    "propagate_earth_coherent",
    "integrate_initial_and_surface_fluxes",
    "integrate_height_and_sum_flavours",
    "detector_conversion_spectra_from_coherent_result",
    "load_coherent_solar_detector_result",
    "propagate_solar_to_detector_coherent",
    "run_and_save_solar_to_detector_coherent",
    "save_coherent_solar_detector_result",
    "load_incoherent_solar_detector_result",
    "propagate_solar_to_detector_incoherent",
    "run_and_save_solar_to_detector_incoherent",
    "save_incoherent_solar_detector_result",
    "load_legacypeanuts_solar_detector_result",
    "propagate_solar_to_detector_legacypeanuts",
    "run_and_save_solar_to_detector_legacypeanuts",
    "save_legacypeanuts_solar_detector_result",
    "aggregate_detector_conversion_by_mode",
    "aggregate_detector_flux_by_mode",
    "build_detector_flux_filename",
    "build_detector_flux_path",
    "detector_initial_flavour",
    "detector_particle_mode",
    "load_detector_flux_directory",
    "load_detector_flux_result",
    "relative_error_summary",
    "save_detector_flux_result",
]
