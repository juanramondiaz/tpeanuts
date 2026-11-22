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

"""tpeanuts-vs-nuSQuIDS wall-clock performance comparisons.

The shared, backend-agnostic timing harness lives in ``benchmarks.common.timing``.
``run_benchmark.py`` (added once ``validation.nusquids.adapter`` and at
least one ``validation.nusquids.compare`` scenario exist) is the part
that actually times tpeanuts's own calls against nuSQuIDS's.

Module contents:
    ``run_benchmark``
        The scenarios timed against nuSQuIDS.
"""

from __future__ import annotations
