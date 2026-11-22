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

"""The physics engine: model-, medium-, detector-, and source-agnostic.

``tpeanuts.physics`` never imports ``tpeanuts.source``, ``tpeanuts.medium``,
or ``tpeanuts.detector`` -- data flows the other way, those layers consume
this one. There is no separate "Standard Model" subpackage here: the SM is
simply the state with every optional extension (sterile, NSI, magnetic
moment, decoherence) switched off, so ``physics.extensions`` holds every
pluggable ingredient in one flat registry instead of forking SM/BSM code
paths.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "FlavourSpace": "tpeanuts.physics.common.space",
    "Dirac": "tpeanuts.physics.common.nature",
    "Majorana": "tpeanuts.physics.common.nature",
    "StandardMixingParameters": "tpeanuts.physics.common.mixing",
    "SterileMixingParameters": "tpeanuts.physics.common.mixing",
    "MixingMatrix": "tpeanuts.physics.common.mixing",
    "MassSpectrum": "tpeanuts.physics.common.mass_spectrum",
    "DensityMatrix": "tpeanuts.physics.common.state",
    "PropagationModel": "tpeanuts.physics.model",
    "PropagationMethod": "tpeanuts.physics.evolution.method",
    "effective_majorana_mass": "tpeanuts.physics.common.majorana_mass",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
