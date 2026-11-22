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

"""Code shared by the external validations of tpeanuts, whichever package it is compared with.

``validation.common.reference`` is the independent NumPy reference (no ``tpeanuts``, no external solver);
``validation.common.tpeanuts_side`` evaluates the same segments, mixings and weights with tpeanuts. The package-specific
adapters are in ``validation.nusquids`` (and, for the original PEANUTS, ``validation.peanuts``).

Module contents:
    ``validation.common.reference``
        Plain-NumPy mixing, vacuum, segment propagation, adiabatic weights and path geometry.
    ``validation.common.tpeanuts_side``
        The tpeanuts evaluation of the same quantities, with the same signatures where possible.
"""

from __future__ import annotations
