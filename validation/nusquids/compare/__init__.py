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

"""tpeanuts-vs-nuSQuIDS precision comparisons, one module per physics scenario.

``grid.py`` holds the shared, backend-agnostic comparison machinery
(build a grid, call two arbitrary probability functions over it,
compute error metrics, check unitarity, export a CSV) -- it does not
import ``tpeanuts`` or nuSQuIDS itself, only plain numpy, so it is
reusable by any future comparison (not specific to this pair of
backends at all). Every other module in this package (``vacuum``,
``earth``, ``atmosphere``, ``solar``, ``solar_earth``,
``bsm_sterile`` -- added one at a time, see
``validation.nusquids``'s own package docstring) is the part that
actually imports both ``tpeanuts`` and
``validation.nusquids.adapter`` and wires one concrete scenario's pair
of probability functions into ``grid.py``'s generic harness.

Module contents:
    ``grid``
        Backend-agnostic comparison/error/export utilities.
    ``chain``
        ``nusquids_propagate(medium, flux, model)``: nuSQuIDS behind the same shape as ``medium.propagate``.
"""

from __future__ import annotations
