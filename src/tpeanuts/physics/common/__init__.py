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

"""Model-agnostic physics infrastructure.

Everything under ``tpeanuts.physics.common`` is shared, unconditionally, by
every scenario (Standard Model, sterile 3+N, NSI, Dirac or Majorana, with or
without a magnetic-moment/decoherence extension). Nothing here imports
``tpeanuts.physics.extensions`` -- extensions depend on this layer, never the
other way around.
"""

from __future__ import annotations
