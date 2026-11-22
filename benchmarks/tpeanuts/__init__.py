#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#
# TPeanuts — Fast Simulation of Neutrino Oscillations in Matter.
#
# Developed as part of the Master's Thesis for the MSc in Advanced Physics,
# specialization in Theoretical Physics, at the Universitat de València.
#
# Author
# ------
# Juan Ramón Díaz Santos
# diazjuan@alumni.uv.es
#
# Supervisors
# -----------
# Roberto Ruiz de Austri Bazán
# rruiz@ific.uv.es
#
# Michele Lucente
# michele.lucente@unibo.it
#
# Universitat de València, Spain
# September 2026
# =============================================================================

"""The tpeanuts side of the benchmarks: every scenario of the catalogue, built through the public API (plan step B2).

The scenarios are timed alone here (the baseline); the comparisons with nuSQuIDS and the original peanuts reuse the same
sizes and record the same schema, from their own packages and environments.

Module contents:
    ``cases``
        The scenarios as ``Case`` objects: build, evaluate, chunked to a memory budget.
    ``run``
        Command-line runner: times the scenarios and writes a ``ResultSet`` JSON.
"""
