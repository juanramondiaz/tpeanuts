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

"""Shared detector utilities.

Module contents:
    observation
        Container for pointwise and binned measurements.
    target
        Target-particle counts derived from material composition.
    response
        Energy-response matrices and grid redistribution.
    efficiency
        Detection-efficiency utilities.
    background
        Construction of background-count vectors.
    event_rate
        Spectrum folding, detector response and bin integration.
"""
