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
Detector-agnostic event-rate building blocks.

Package modules:
    observation
        Observation: a small container for a published binned/pointwise
        measurement (value, uncertainty, bin/energy grid), detector-agnostic.
    target
        Number of target electrons/nucleons from material stoichiometry.
    response
        Gaussian energy-response (migration) matrix construction.
    efficiency
        Detection/selection efficiency application.
    background
        Background-rate placeholder interface.
    event_rate
        The single event-rate folding assembly: true spectrum -> response ->
        efficiency -> binned, exposure-scaled counts (+ optional background).
"""
