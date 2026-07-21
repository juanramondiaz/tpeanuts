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
    solar, earth, solar_earth
        Solar production, pure Earth propagation, and their incoherent
        composition.
    legacy
        Explicit legacy validation backend, imported from
        ``tpeanuts.pipeline.legacy`` rather than re-exported here.
    atmosphere, atmosphere_earth
        Production-to-surface and surface-to-detector atmosphere workflows.
"""



from tpeanuts.pipeline.atmosphere import (
    AtmosphereSurfaceResult,
    propagate_atmosphere_to_surface,
    select_production_flux,
)
from tpeanuts.pipeline.atmosphere_earth import (
    AtmosphereEarthDetectorResult,
    AtmosphereDetectorGridResult,
    propagate_atmosphere_grid_to_detector,
    propagate_surface_to_detector,
    detector_flux_from_production,
    integrate_detector_flux_over_height,
    sum_detected_flavours,
)
from tpeanuts.pipeline.solar import SolarSurfaceResult, propagate_solar_to_surface
from tpeanuts.pipeline.earth import (
    EarthDetectorResult,
    propagate_earth_to_detector,
    propagate_earth_to_detector_exposure,
)
from tpeanuts.pipeline.solar_earth import (
    SolarEarthDetectorResult,
    propagate_solar_to_earth_detector,
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
    "AtmosphereSurfaceResult",
    "AtmosphereEarthDetectorResult",
    "AtmosphereDetectorGridResult",
    "select_production_flux",
    "propagate_atmosphere_to_surface",
    "propagate_surface_to_detector",
    "propagate_atmosphere_grid_to_detector",
    "detector_flux_from_production",
    "integrate_detector_flux_over_height",
    "sum_detected_flavours",
    "SolarSurfaceResult",
    "EarthDetectorResult",
    "SolarEarthDetectorResult",
    "propagate_solar_to_surface",
    "propagate_earth_to_detector",
    "propagate_earth_to_detector_exposure",
    "propagate_solar_to_earth_detector",
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
