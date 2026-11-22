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

"""What a cluster run needs: the wrapper, the submit files generated from the YAML specifications, and ``render.py``.

Nothing here is imported by the benchmarks themselves; the package exists so that ``render.py`` can be tested.

Module contents:
    ``render``
        Turns a specification of ``benchmarks/suites/`` into an HTCondor submit file, and writes ``submit_all.sh``.
"""
