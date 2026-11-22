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

"""Independent, optional cross-validation/benchmarking backends -- never a tpeanuts dependency.

Deliberately outside ``src/tpeanuts/`` (not installed as part of the
``tpeanuts`` distribution): each subpackage here wraps one external
tool used purely to validate or benchmark tpeanuts's own physics
against an independent reference, never the other way around -- see
each subpackage's own module docstring for how that independence is
enforced structurally, not just documented.

Module contents:
    ``validation.common``
        What every external validation shares: the independent NumPy reference (mixing, vacuum, segments,
        adiabatic weights, path geometry) and its tpeanuts counterpart.
    ``validation.nusquids``
        Cross-validation and benchmarking against the nuSQuIDS solver.
"""

from __future__ import annotations
