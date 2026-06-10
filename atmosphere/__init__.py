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
Torch-native atmospheric neutrino utilities.

Package contents:
    geometry
        Angle conversions, spherical path lengths, altitude grids, and
        peanuts-compatible nadir coordinates.
    density, density_pymsis
        Atmospheric mass-density and electron-density sources, including
        exponential, file, MCEq, and pymsis/MSIS backends.
    propagation
        Atmospheric Hamiltonians, segmented evolution operators, and coherent
        propagation from production height to Earth surface.
    earth
        Surface-to-detector Earth matter evolution for atmospheric zenith
        angles.
    flux, io
        Flavour-flux loading, propagation, probability matrices, and
        height-integration utilities.
"""


