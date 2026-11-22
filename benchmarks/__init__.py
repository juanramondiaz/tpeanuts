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

"""Comparative performance benchmarks of tpeanuts against the reference packages.

This is a repository-level package, outside ``src/``: it is not part of the installed wheel and carries dependencies
the library must not require (nuSQuIDS, the original peanuts). ``validation`` answers whether the results agree;
``benchmarks`` answers how long they take and how that scales. Each reference package runs in its own process and
environment and exchanges only JSON files (see ``benchmarks/README.md`` for the measurement protocol).

Module contents:
    ``common``
        Shared harness: timer, environment record, result schema, scenarios.
    ``tpeanuts``
        The tpeanuts scenarios, timed alone (the baseline) through the public API.
    ``peanuts``
        tpeanuts against the original peanuts.
    ``nusquids``
        tpeanuts against nuSQuIDS.
    ``report``
        Reads the result files: times, speed-ups, scaling exponents, absences, tables and figures.
"""
