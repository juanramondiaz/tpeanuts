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

"""Concrete sources: whatever emits the neutrinos a scenario propagates.

``tpeanuts.source`` never imports ``tpeanuts.medium`` or
``tpeanuts.detector`` directly, and knows nothing about propagation itself
(that is ``tpeanuts.physics``'s job) -- mirroring the same isolation
``tpeanuts.medium`` keeps from ``tpeanuts.physics.model`` (see
``tpeanuts.medium.earth.exposure``'s own module docstring).

``tpeanuts.source.reactor.reactor.ReactorSource`` emits pure,
incoherent electron antineutrinos at a caller-supplied energy grid, as a
``SourceFlux`` with the single component ``"nue_bar"`` -- a single source
with a single, already-known flavour composition needs nothing more.
``tpeanuts.source.atmosphere`` is different in kind, not degree: an
atmospheric flux table genuinely mixes several particle species across an
(energy, angle) grid with data-dependent relative weights, so it exposes
the raw tabulated data itself (``tpeanuts.source.atmosphere.io.
AtmosphericFluxTable``) rather than a single ready-built ``Flux`` --
picking one bin and species to build a concrete ``Flux`` from is a
pipeline-level concern (a future ``tpeanuts.pipeline.atmosphere``), not
this package's own job. No shared base class or protocol exists across
either source yet, for the same reason none exists in ``tpeanuts.medium``.
"""

from __future__ import annotations

from tpeanuts.util.lazy import lazy_exports

_EXPORTS = {
    "SolarSource": "tpeanuts.source.solar.solar",
    "ReactorSource": "tpeanuts.source.reactor.reactor",
    "AtmosphericSource": "tpeanuts.source.atmosphere.atmosphere",
}
__all__ = list(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
