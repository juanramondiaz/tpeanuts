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

"""Benchmarks of tpeanuts against the original PEANUTS, run in the original's own process and environment (step B4).

The original is the separate repository next to this one (``PEANUTS_PATH`` or a sibling ``PEANUTS`` directory), not the
modified fork in ``v1_0/peanuts``. Each case is checked against the tpeanuts reference before it is timed; scenarios the
original does not support are recorded as unavailable, never as zero.

Module contents:
    ``adapter``
        The original loaded with the shared parameters, and its scenarios over a grid.
    ``run``
        Command-line runner: checks and times the scenarios and writes a ``ResultSet`` JSON.
"""
