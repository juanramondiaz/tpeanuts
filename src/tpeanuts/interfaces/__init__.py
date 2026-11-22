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

"""Tiny data contracts exchanged between source, medium, and detector.

``tpeanuts.source``, ``tpeanuts.medium``, and ``tpeanuts.detector`` never
import each other directly -- every value that crosses one of those
boundaries is one of the plain, physics-*light* dataclasses defined here
instead. A source produces a ``Flux``; a medium consumes/produces a
``MediumProfile``; a detector produces an ``ObservableSpectrum`` (a rate over
true energy) or a ``BinnedSpectrum`` (counts per observed-energy bin) from the ``Flux`` that reached it.
Nothing here knows how to build any of the three (that is each domain's own job) or how to
propagate anything (that is ``tpeanuts.physics``'s job) -- this package only
fixes the shape of the data that flows between them.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "Flux": "tpeanuts.interfaces.flux",
    "ProductionDistribution": "tpeanuts.interfaces.flux",
    "SourceFlux": "tpeanuts.interfaces.flux",
    "MediumProfile": "tpeanuts.interfaces.medium",
    "ObservableSpectrum": "tpeanuts.interfaces.observable",
    "BinnedSpectrum": "tpeanuts.interfaces.observable",
    "Detector": "tpeanuts.interfaces.observable",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
