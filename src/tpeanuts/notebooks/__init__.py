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

"""Support modules of the tutorial notebooks in the repository's ``notebooks/`` directory.

They are not part of the stable API: they exist so that every notebook configures paths, runtime and plot style, and
saves its figures, in the same way. They need the ``notebooks`` extra (matplotlib, pandas).

Module contents:
    notebookConfig
        Output paths, runtime and display style: ``load_notebook_config``.
    notebooks_helper
        Array conversion, error measures and ``save_and_show``.
    validation_helper
        The common structure of the validation notebooks: cases with a tolerance, residual statistics by region.
"""

from __future__ import annotations
