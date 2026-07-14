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
High-level neutrino propagation pipelines.

This package contains user-facing orchestration workflows that compose the
medium-specific solar, vacuum, Earth, and atmosphere modules with common
probability and flux utilities. The package is intentionally above ``core``
and ``medium``: it prepares inputs, calls the physics blocks, integrates or
aggregates observables, and saves pipeline outputs.

Module groups:
    pipeline_common
        Shared setup helpers for PMNS objects, profiles, exposure, and devices.
    pipeline_incoherent, pipeline_legacypeanuts
        Solar-to-detector workflows with incoherent and legacy peanuts
        treatments.
    pipeline_atmosphere, atmosphere_flux
        Atmosphere and atmosphere-plus-Earth flux workflows.
"""



from tpeanuts.pipeline.pipeline_atmosphere import (
    select_particle_angle_flux,
    build_atmosphere_trajectories,
    propagate_atmosphere_coherent,
    propagate_earth_coherent,
    integrate_initial_and_surface_fluxes,
    integrate_height_and_sum_flavours,
)
from tpeanuts.pipeline.atmosphere_flux import (
    build_probability_matrix,
    integrate_detector_flux_over_height,
    integrate_flux_over_height,
    propagate_flux_E_h,
    propagate_flux_vector,
)
from tpeanuts.pipeline.pipeline_incoherent import (
    load_incoherent_solar_detector_result,
    propagate_solar_to_detector_incoherent,
    run_and_save_solar_to_detector_incoherent,
    save_incoherent_solar_detector_result,
)
from tpeanuts.pipeline.pipeline_legacypeanuts import (
    load_legacypeanuts_solar_detector_result,
    propagate_solar_to_detector_legacypeanuts,
    run_and_save_solar_to_detector_legacypeanuts,
    save_legacypeanuts_solar_detector_result,
)
from tpeanuts.pipeline.io import (
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
    "build_atmosphere_trajectories",
    "propagate_atmosphere_coherent",
    "propagate_earth_coherent",
    "integrate_initial_and_surface_fluxes",
    "integrate_height_and_sum_flavours",
    "build_probability_matrix",
    "integrate_detector_flux_over_height",
    "integrate_flux_over_height",
    "propagate_flux_E_h",
    "propagate_flux_vector",
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
