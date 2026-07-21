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
Earth density, geometry, exposure, and propagation utilities.

This package implements the Earth matter-regeneration layer of the
simulation: the Earth electron-density profile (``medium.earth.profile``),
day/night Earth-crossing trajectory geometry (``medium.earth.geometry``),
the Earth evolution operator and matter-regeneration probability pipelines
(``medium.earth.evolutor``, ``medium.earth.probability``), nadir-angle
exposure tables used to time-average probabilities over a detector's annual
exposure (``medium.earth.exposure_math``, ``medium.earth.exposure_table``,
``medium.earth.exposure_io``, ``medium.earth.exposure_integration``), flux
helpers (``medium.earth.flux``), and validation utilities comparing this
implementation against the legacy NumPy/Numba ``peanuts`` package
(``medium.earth.validation``).

Re-exported names:
    EarthProfile: Torch representation of the Earth electron-density profile
        and trajectory geometry (see ``medium.earth.profile``).
    compare_earth_probability_state_with_legacy: Compare pointwise Earth
        probabilities against legacy peanuts (see ``medium.earth.validation``).
    compare_earth_probability_exposure_with_legacy: Compare exposure-integrated
        Earth probabilities against legacy peanuts (see
        ``medium.earth.validation``).
"""

from tpeanuts.medium.earth.profile import EarthProfile, build_earth_profile
from tpeanuts.medium.earth.geometry import build_atmosphere_trajectories
from tpeanuts.medium.earth.validation import (
    compare_earth_probability_exposure_with_legacy,
    compare_earth_probability_state_with_legacy,
)

__all__ = [
    "EarthProfile",
    "build_earth_profile",
    "build_atmosphere_trajectories",
    "compare_earth_probability_state_with_legacy",
    "compare_earth_probability_exposure_with_legacy",
]
