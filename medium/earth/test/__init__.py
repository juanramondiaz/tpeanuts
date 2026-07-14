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
Tests for tpeanuts.medium.earth.

Package modules:
    test1_density
        Tests for the EarthProfile electron-density module.
    test2_geometry
        Tests for Earth trajectory geometry utilities.
    test3_evolutor
        Tests for the earth_evolutor flavour-basis evolution operator.
    test4_probabilities
        Tests for the pearth Earth matter-regeneration probability pipeline.
    test5_flux
        Tests for the earth_flux / earth_flux_integrated flux pipeline.
    test6_exposure
        Tests for the exposure_math, exposure_table, exposure_io, and
        exposure_integration modules.
    test7_legacy_validation
        Validation of pearth / pearth_integrated against the legacy peanuts
        NumPy implementation.
"""
