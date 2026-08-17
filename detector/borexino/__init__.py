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
Borexino-specific detector wiring, built from ``detector.common``/``detector.interaction``.

Package modules:
    parameters
        Scintillator composition, reference target mass/exposure, and the
        (illustrative, see its own docstring) energy-resolution
        parametrization.
    response
        Borexino's sigma(T) resolution curve and response matrix.
    backgrounds
        Background placeholder (see ``detector.common.background``).
    event_rate
        Composed Borexino event-rate function.
    io
        Loader for the real Borexino Nature 2018 low-energy rate spectrum.
"""
