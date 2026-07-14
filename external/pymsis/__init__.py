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
Optional PyMSIS atmosphere-density backend.

This package wraps the optional third-party ``pymsis`` Python package,
which implements NRLMSIS 2.0 (and earlier MSIS versions): an empirical,
data-driven model of Earth's neutral atmosphere density and composition as
a function of date/time, geographic position, altitude, and space-weather
indices (solar radio flux F10.7 and geomagnetic activity Ap). tpeanuts does
not reimplement this model; it only calls ``pymsis.calculate(...)`` to get
mass density vs. altitude and converts the result (density.py) into the
electron density and vertical atmospheric depth needed by the matter
oscillation propagation code. This is a more realistic alternative to the
simple exponential atmosphere approximation used elsewhere in tpeanuts.

Module contents:
    PyMSISatmosphereConfig
        Dataclass with the date, geographic location, solar/geomagnetic
        activity indices, MSIS version, and electron fraction used to call
        pymsis.
    atmosphere_density_pymsis(...)
        Call pymsis on an altitude grid and return mass density, electron
        density, and atmospheric depth as torch tensors.
"""

from tpeanuts.external.pymsis.density import (
    PyMSISatmosphereConfig,
    atmosphere_density_pymsis,
)

__all__ = [
    "PyMSISatmosphereConfig",
    "atmosphere_density_pymsis",
]
