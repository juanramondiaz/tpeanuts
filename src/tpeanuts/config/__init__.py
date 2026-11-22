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

"""User-tunable default values, as opposed to fixed physical constants.

``tpeanuts.config.default`` is the single source of truth for values a
caller could reasonably want to override for their own run (an algorithm's
hyperparameters, a provider choice, a numerical tolerance) -- the
counterpart of ``tpeanuts.util.constant`` (facts of nature: particle
masses, coupling constants, geometric radii), which is never something a
caller "configures". Every dataclass field or function parameter that
defaults to a ``tpeanuts.config.default`` value can still be overridden
explicitly by the caller; the module only fixes what happens when they
don't.
"""

from __future__ import annotations
